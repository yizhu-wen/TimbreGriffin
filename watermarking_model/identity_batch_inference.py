import numpy as np
import random
import torch


def set_random_seed(seed: int):
    random.seed(seed)  # For Python random module
    np.random.seed(seed)  # For NumPy
    torch.manual_seed(seed)  # For PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # For PyTorch (GPU)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables optimization for reproducibility


set_random_seed(999)
import warnings

warnings.filterwarnings("ignore")

import torch.nn as nn
from silero_vad import load_silero_vad
from torch.nn import LeakyReLU
from watermarking_model.model.blocks import (
    FCBlock,
    Conv2Encoder,
    WatermarkEmbedder,
    WatermarkExtracter,
    ReluBlock,
)
from distortions.frequency import TacotronSTFT, fixed_STFT, tacotron_mel
import yaml
import torch.nn.functional as F
from dataset.data import WavDataset as MyDataset
from dataset.data import collate_fn
from torch.utils.data import DataLoader
import torch
import torchaudio
from rich.progress import track
from torch.nn.functional import mse_loss

# Optional: set up a small constant
EPS = 1e-9


def generate_random_msg(batch_size, msg_length, device):
    # random [0, 1], mapped to [-1, 1]
    return (
        torch.randint(0, 2, (batch_size, 1, msg_length), device=device).float() * 2
    ) - 1


class Encoder(nn.Module):
    def __init__(self, process_config, model_config, train_config, msg_length):
        super(Encoder, self).__init__()
        self.name = "conv2"
        self.win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["hidden_dim"] + 3
        self.layers_EM = model_config["conv2"]["layers_EM"]
        self.n_fft = process_config["mel"]["n_fft"]
        self.hop_length = process_config["mel"]["hop_length"]
        self.win_length = process_config["mel"]["win_length"]
        self.sampling_rate = process_config["audio"]["or_sample_rate"]
        self.voice_prefilling = (
            int(
                (
                    (process_config["audio"]["audio_prefilling"] + 0.05)
                    * self.sampling_rate
                )
                // self.hop_length
            )
            - 1
        )  # 204
        self.delay_amt = int(
            (train_config["watermark"]["delay_amt_second"] * self.sampling_rate)
            // self.hop_length
            + 1
        )  # 51
        self.future_amt = (
            int(
                (train_config["watermark"]["future_amt_second"] * self.sampling_rate)
                // self.hop_length
                + 1
            )
            - 1
        )  # 50
        self.power = 1.0

        self.smooth_chunks = train_config["optimize"]["smooth_chunks"]
        self.dilate_chunks = train_config["optimize"]["dilate_chunks"]
        self.target_smooth_ms = train_config["optimize"]["target_smooth_ms"]
        self.target_dilate_ms = train_config["optimize"]["target_dilate_ms"]
        self.floor_eps = train_config["optimize"]["floor_eps"]
        self.tau = train_config["optimize"]["tau"]
        self.vad = load_silero_vad()
        self.vad_threshold = 0.50

        self.vocoder_step = model_config["structure"]["vocoder_step"]
        # MLP for the input wm
        # self.msg_linear_in = FCBlock(msg_length, self.win_dim, activation=LeakyReLU(inplace=True))
        self.msg_linear_in = FCBlock(
            msg_length, self.win_dim // 2, activation=LeakyReLU(inplace=True)
        )

        # stft transform
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        )

        self.ENc = Conv2Encoder(
            input_channel=2,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
            n_layers=self.layers_CE,
        )

        self.EM = WatermarkEmbedder(
            input_channel=self.EM_input_dim,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
            n_layers=self.layers_EM,
        )

    def pad_w_zero_stft(self, input_stft, watermark_stft, voice_prefilling):
        """
        Pad the watermarked stft output with zeros on the left + right,
        respecting future_amt and chunk-based offsets.
        """

        zeros_right_len = (
            input_stft.shape[3]
            - watermark_stft.shape[3]
            - (voice_prefilling + self.future_amt)
        )
        if zeros_right_len < 0:
            # Edge case: won't happen if chunking logic is correct, but just to be safe
            zeros_right_len = 0

        zeros_left = torch.zeros_like(
            input_stft[:, :, :, : voice_prefilling + self.future_amt]
        )
        zeros_right = torch.zeros_like(input_stft[:, :, :, :zeros_right_len])

        actual_watermark = (
            torch.cat([zeros_left, watermark_stft, zeros_right], dim=3) + EPS
        )
        return actual_watermark, zeros_right

    def forward(self, x, msg, global_step):
        self.stft.num_samples = x.shape[-1]
        _, _, stft_result = self.stft.transform(x)
        # Evaluate how many chunks we can process
        # 2s input + 0.5s calculation delay
        # 2.00*16000 = 32000
        # 32800 // hop_length + 1 = 201 center=True
        # 0.5s*16000 = 8000
        # 8000 // hop_length + 1 = 51 center=True
        # Predict future 0.5s watermark
        # 0.5*16000 = 8000
        # 8000 // hop_length + 1 =51
        if (
            int(
                stft_result.shape[-1]
                - (self.voice_prefilling + self.future_amt) / self.delay_amt
            )
            <= 0
        ):
            return None  # Not enough frames for a chunk

        list_of_watermarks = []
        for i in range(
            int(
                (stft_result.shape[-1] - (self.voice_prefilling + self.future_amt))
                / self.delay_amt
            )
        ):
            carrier_encoded = self.ENc(
                stft_result[
                    :,
                    :,
                    :,
                    i * self.delay_amt : self.voice_prefilling + i * self.delay_amt,
                ]
            )
            # torch.Size([B, 1, 81])
            # torch.Size([B, 81, 1])
            # torch.Size([B, 1, 81, 1])
            # torch.Size([B, 1, 162, 201])
            watermark_encoded = (
                self.msg_linear_in(msg)
                .transpose(1, 2)
                .unsqueeze(1)
                .repeat(1, 1, 2, carrier_encoded.shape[3])
            )
            # watermark_encoded = self.msg_linear_in(msg).transpose(1, 2).unsqueeze(1).repeat(1, 1, 1,
            #                                                                                 carrier_encoded.shape[3])

            concatenated_feature = torch.cat(
                (
                    carrier_encoded,
                    stft_result[
                        :,
                        :,
                        :,
                        i * self.delay_amt : self.voice_prefilling + i * self.delay_amt,
                    ],
                    watermark_encoded,
                ),
                dim=1,
            )
            # [B, 2, bins, length]
            # Embed the watermark
            carrier_watermarked = self.EM(concatenated_feature)
            # Append both the watermark chunk and the pilot segment
            list_of_watermarks.append(carrier_watermarked)

        if len(list_of_watermarks) > 0:
            watermark = torch.cat(list_of_watermarks, dim=-1)
            all_watermark_stft, zeros_right = self.pad_w_zero_stft(
                stft_result, watermark, self.voice_prefilling
            )
            del list_of_watermarks
            mask = stft_result != 0
            all_watermark_stft = all_watermark_stft * mask + 0.0000001

            # Recompute magnitude & phase
            real_part = all_watermark_stft[:, 0, :, :]
            imag_part = all_watermark_stft[:, 1, :, :]
            spect = torch.sqrt(real_part**2 + imag_part**2)
            phase = torch.atan2(imag_part, real_part)

            y = self.stft.inverse(spect, phase).squeeze(1)
            del spect, phase, real_part, imag_part, all_watermark_stft

            with torch.no_grad():
                # Get chunk-level speech probabilities for the batch.
                # The output shape should be [batch, num_chunks]
                batch_chunk_probs = self.vad.audio_forward(x, sr=self.sampling_rate)
            p = batch_chunk_probs.to(device=y.device, dtype=y.dtype)  # [B, C]
            C = p.shape[-1]

            # --- infer hop in ms from C, T, sr ---
            # chunks_per_sec = C * sr / T; hop_ms = 1000 / chunks_per_sec
            hop_ms = 1000.0 * self.stft.num_samples / (C * self.sampling_rate)

            # If caller didn't fix counts, compute them from target ms
            if self.smooth_chunks is None:
                smooth_chunks = max(1, int(round(self.target_smooth_ms / hop_ms)))
            if self.dilate_chunks is None:
                dilate_chunks = max(0, int(round(self.target_dilate_ms / hop_ms)))
                # in practice keep at least 1 for robustness at edges
                if dilate_chunks == 0:
                    dilate_chunks = 1

            # 2) Soft step around the threshold
            m_chunk = torch.sigmoid(
                (p - self.vad_threshold) / self.tau
            )  # [B, C] in (0, 1)

            # 3) Smooth in chunk space (moving average)
            if smooth_chunks > 1:
                k = (
                    torch.ones(1, 1, smooth_chunks, device=y.device, dtype=y.dtype)
                    / smooth_chunks
                )
                z = m_chunk.unsqueeze(1)  # [B,1,C]
                pad = smooth_chunks // 2
                z = F.pad(z, (pad, pad), mode="replicate")
                m_chunk = F.conv1d(z, k, stride=1).squeeze(1)  # [B, C]

            # 4) Dilate voiced regions by max-pool
            if dilate_chunks > 0:
                z = m_chunk.unsqueeze(1)  # [B,1,C]
                pad = dilate_chunks
                z = F.pad(z, (pad, pad), mode="replicate")
                m_chunk = F.max_pool1d(z, kernel_size=2 * pad + 1, stride=1).squeeze(
                    1
                )  # [B, C]

            # 5) Upsample to sample grid
            m_up = F.interpolate(
                m_chunk.unsqueeze(1),
                size=self.stft.num_samples,
                mode="linear",
                align_corners=True,
            ).squeeze(
                1
            )  # [B, T]

            # After computing m_up ...
            frame_size = 512
            rms = x.unfold(
                -1, frame_size, frame_size // 2
            )  # [B, num_frames, frame_size]
            rms = rms.pow(2).mean(dim=-1).sqrt()  # [B, num_frames]
            # Normalize RMS into [0,1] (prevent divide-by-zero)
            rms = rms / (rms.max(dim=1, keepdim=True).values + 1e-8)

            # Upsample RMS back to sample level and map to floor in [floor_min,floor_max].
            dynamic_floor = F.interpolate(
                rms.unsqueeze(1),
                size=self.stft.num_samples,
                mode="linear",
                align_corners=True,
            ).squeeze(1)
            floor_min, floor_max = 0.05, 0.2
            dynamic_floor = floor_min + (floor_max - floor_min) * dynamic_floor

            # Now build the mask using this per-sample floor
            soft_sample_masks = (dynamic_floor + (1.0 - dynamic_floor) * m_up).clamp_(
                0.0, 1.0
            )

            # # 6) Floor ε so mask ∈ [ε, 1]
            # soft_sample_masks = (self.floor_eps + (1.0 - self.floor_eps) * m_up).clamp_(0.0, 1.0)  # [B, T]

            masked_y = y * soft_sample_masks
            # # Threshold the probabilities to obtain a binary mask per chunk.
            # batch_chunk_mask = (batch_chunk_probs > self.vad_threshold).float()
            #
            # # Upsample the chunk-level mask to a sample-level mask.
            # # Each chunk's decision is repeated for chunk_size samples.
            # sample_masks = torch.repeat_interleave(batch_chunk_mask, 512, dim=1).to(y.device)
            #
            # # Since the upsampled mask might be longer than the actual audio length,
            # # slice the mask to match the original number of samples.
            # sample_masks = sample_masks[:, :self.stft.num_samples]

            # # Apply the mask to the original audio to zero out non-speech regions.
            # masked_y = y * sample_masks

            return masked_y, zeros_right.shape[-1]
        else:
            print("Not enough watermarking!!!!")
            return None


class Decoder(nn.Module):
    def __init__(self, process_config, model_config, train_config, msg_length):
        super(Decoder, self).__init__()
        self.robust = model_config["robust"]
        # if self.robust:
        #     self.dl = distortion(process_config, train_config)
        self.mel_transform = TacotronSTFT(
            filter_length=process_config["mel"]["n_fft"],
            hop_length=process_config["mel"]["hop_length"],
            win_length=process_config["mel"]["win_length"],
        )
        # self.vocoder = get_vocoder(device)
        self.vocoder_step = model_config["structure"]["vocoder_step"]
        self.win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.hop_length = process_config["mel"]["hop_length"]
        self.block = model_config["conv2"]["block"]
        self.EX = WatermarkExtracter(
            input_channel=2,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
        )
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        )

        # self.msg_linear_out = FCBlock(self.win_dim, msg_length, activation=LeakyReLU(inplace=True))
        self.msg_linear_out = FCBlock(
            self.win_dim // 2, msg_length, activation=LeakyReLU(inplace=True)
        )

    def forward(self, y, global_step):
        y_identity = y
        # if global_step > self.vocoder_step:
        #     y_mel = self.mel_transform.mel_spectrogram(y.squeeze(1))
        #     # y = self.vocoder(y_mel)
        #     y_d = (self.mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
        # else:
        #     y_d = y
        y_d = y

        spect, phase, stft_result = self.stft.transform(y_d.squeeze(1))
        extracted_wm = self.EX(stft_result).squeeze(1)  # (B, win_dim, length)
        # Explicitly split the 162-dim vector into two halves of 81-dim each
        low, high = extracted_wm.chunk(
            2, dim=1
        )  # each has shape [B, win_dim / 2, length]
        low_msg = torch.mean(low, dim=2, keepdim=True).transpose(1, 2)
        high_msg = torch.mean(high, dim=2, keepdim=True).transpose(1, 2)
        msg_avg = (
            low_msg + high_msg
        ) / 2  # Average the two halves -> shape: [B, 1, 81]
        # msg = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1,2)
        # msg = self.msg_linear_out(msg)
        msg = self.msg_linear_out(msg_avg)

        _, _, stft_result_identity = self.stft.transform(y_identity)
        extracted_wm_identity = self.EX(stft_result_identity).squeeze(1)
        low_identity, high_identity = extracted_wm_identity.chunk(
            2, dim=1
        )  # each has shape [B, win_dim / 2, length]
        low_msg_identity = torch.mean(low_identity, dim=2, keepdim=True).transpose(1, 2)
        high_msg_identity = torch.mean(high_identity, dim=2, keepdim=True).transpose(
            1, 2
        )
        msg_avg_identity = (
            low_msg_identity + high_msg_identity
        ) / 2  # Average the two halves -> shape: [B, 1, 81]
        # msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
        # msg_identity = self.msg_linear_out(msg_identity)
        msg_identity = self.msg_linear_out(msg_avg_identity)
        del stft_result, stft_result_identity, extracted_wm, extracted_wm_identity
        return msg, msg_identity


process_config = yaml.load(open("./config/process.yaml", "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open("./config/model.yaml", "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open("./config/train.yaml", "r"), Loader=yaml.FullLoader)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    process_config, model_config, train_config, train_config["watermark"]["length"]
).to(device)
decoder = Decoder(
    process_config, model_config, train_config, train_config["watermark"]["length"]
).to(device)
checkpoint_path = "results/ckpt/pth/MSE_loudness_split_frequency_adaptive_soft_vad_ep_30_2025-09-17_06_50_13.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

train_config["path"]["raw_path"] = "/home/yizhu/Data/VoxCeleb_wav"
dev_audios = MyDataset(
    process_config=process_config, train_config=train_config, flag="test"
)
dev_audios_loader = DataLoader(
    dev_audios,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=20,
    persistent_workers=True,
)


with torch.inference_mode():
    encoder.eval()
    decoder.eval()
    test_avg_acc = [0, 0]
    test_avg_snr = 0
    total_samples = 0
    count = 0
    for sample in track(dev_audios_loader):
        count += 1
        # ---------------- build watermark
        wav_matrix = sample["matrix"].to(device)
        msg = generate_random_msg(
            wav_matrix.size(0), train_config["watermark"]["length"], device
        )
        watermark, carrier_wateramrked = encoder(wav_matrix, msg, 1)
        y_wm = wav_matrix + watermark

        # add a for loop for each distortion and print out different accuracy

        decoded = decoder(y_wm, 1)
        decoder_acc = [
            ((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
            ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
        ]
        zero_tensor = torch.zeros(wav_matrix.shape).to(device)
        snr = 10 * torch.log10(
            mse_loss(wav_matrix.detach(), zero_tensor)
            / mse_loss(wav_matrix.detach(), y_wm.detach())
        )

        test_avg_snr += snr
        test_avg_acc[0] += decoder_acc[0]
        test_avg_acc[1] += decoder_acc[1]

    test_avg_acc[0] /= count
    test_avg_acc[1] /= count
    test_avg_snr /= count

    print("Test Average SNR:", test_avg_snr)
    print("Test Average Accuracy:", test_avg_acc[0])
    print("The number samples:", len(dev_audios))
