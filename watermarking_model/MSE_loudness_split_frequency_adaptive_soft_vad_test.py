import numpy as np
import matplotlib.pyplot as plt
import torch
import random


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


def generate_random_msg(batch_size, msg_length, device):
    # random [0, 1], mapped to [-1, 1]
    return (
        torch.randint(0, 2, (batch_size, 1, msg_length), device=device).float() * 2
    ) - 1


def normalize_audio(y: torch.Tensor) -> torch.Tensor:
    """Normalize an audio tensor so its maximum absolute value is 1."""
    peak = torch.max(torch.abs(y))
    if peak.item() > 1e-8:
        y = y / peak
    return y


def save_spectrogram(signal, filepath, sample_rate=16000):
    plt.figure(figsize=(10, 4))
    plt.specgram(
        np.maximum(signal.cpu().detach().numpy(), 1e-10),
        Fs=sample_rate,
        NFFT=322,
        noverlap=160,
        window=np.hanning(322),
        cmap="magma",
        vmin=-100,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(filepath, format="png", bbox_inches="tight", pad_inches=0.0)
    plt.close()


def save_audio(signal, filepath="output.wav", sample_rate=16000):
    # signal shape: (1, length)
    torchaudio.save(filepath, signal.cpu(), sample_rate)
    return filepath


def splice_with_irrelevant(watermarked, irrelevant_paths, sample_rate=16000):
    """
    Replace 50% of each sample in a batch with random unrelated audio of same length.

    Args:
        watermarked (Tensor): (B, 1, L) batch of watermarked audio
        irrelevant_paths (List[str]): list of paths to unrelated .wav files
        sample_rate (int): target sample rate to match
    Returns:
        Tensor: spliced audio of same shape
    """
    B, C, L = watermarked.shape
    cut_len = L // 2
    mixed = watermarked.clone()

    for i in range(B):
        # Load a random unrelated audio
        rand_path = random.choice(irrelevant_paths)
        irre_audio, sr = torchaudio.load(rand_path)

        # Convert to mono and resample if needed
        if irre_audio.shape[0] > 1:
            irre_audio = irre_audio.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            irre_audio = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=sample_rate
            )(irre_audio)

        # Trim or pad irrelevant audio to at least cut_len
        if irre_audio.shape[1] < cut_len:
            pad_len = cut_len - irre_audio.shape[1]
            irre_audio = torch.nn.functional.pad(irre_audio, (0, pad_len))
        irre_audio = irre_audio[:, :cut_len]

        # Pick splice position
        start = random.randint(0, L - cut_len)
        end = start + cut_len

        # Replace middle 50% with irrelevant audio
        mixed[i, :, start:end] = irre_audio

    return mixed


def splice_with_irrelevant_scattered(
    watermarked, irrelevant_paths, sample_rate=16000, num_segments=10
):
    """
    Replaces 50% of each audio in a batch with unrelated audio, scattered over multiple random segments.

    Args:
        watermarked (Tensor): (B, 1, L) batch of watermarked audio
        irrelevant_paths (List[str]): list of paths to unrelated .wav files
        sample_rate (int): sample rate of the audio
        num_segments (int): number of scattered segments to use (default: 10)

    Returns:
        Tensor: Corrupted audio of shape (B, 1, L)
    """
    B, C, L = watermarked.shape
    target_total = L // 2  # Replace 50% of audio
    mixed = watermarked.clone()

    for i in range(B):
        # Randomly divide 50% length into `num_segments` random positive integers
        # Ensures segments have variable lengths but sum to target_total
        segment_lengths = torch.randint(
            low=1, high=L // (2 * num_segments), size=(num_segments,)
        )
        scale = target_total / segment_lengths.sum().item()
        segment_lengths = (segment_lengths.float() * scale).long()
        segment_lengths[-1] += (
            target_total - segment_lengths.sum()
        )  # Adjust rounding error

        # Keep track of inserted regions to avoid overlaps
        used_ranges = []
        positions = []

        for seg_len in segment_lengths:
            for _ in range(20):  # Try 20 times to find non-overlapping segment
                start = random.randint(0, L - seg_len)
                end = start + seg_len
                if all(end <= s or start >= e for s, e in used_ranges):
                    used_ranges.append((start, end))
                    positions.append((start, end, seg_len))
                    break

        for start, end, seg_len in positions:
            rand_path = random.choice(irrelevant_paths)
            irre_audio, sr = torchaudio.load(rand_path)

            # Convert to mono
            if irre_audio.shape[0] > 1:
                irre_audio = irre_audio.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != sample_rate:
                irre_audio = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=sample_rate
                )(irre_audio)

            # Ensure it's long enough
            if irre_audio.shape[1] < seg_len:
                irre_audio = torch.nn.functional.pad(
                    irre_audio, (0, seg_len - irre_audio.shape[1])
                )

            irre_audio = irre_audio[:, :seg_len]
            assert irre_audio.shape == (
                1,
                seg_len,
            ), f"Shape mismatch: got {irre_audio.shape}, expected (1, {seg_len})"

            mixed[i, :, start:end] = irre_audio

    return mixed


import torch.nn as nn
from silero_vad import load_silero_vad
from torch.nn import LeakyReLU
from model.blocks import (
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
from torchaudio.functional import resample as tf_resample
from torchaudio.functional import fftconvolve, add_noise
import julius
from typing import Dict, Tuple

# Optional: set up a small constant
EPS = 1e-9

_PHONE_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}  # sr -> (rir, noise)


def _get_phone_assets(target_sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Download/load demo RIR/noise and resample to target_sr. Returns mono [1, T] tensors."""
    if target_sr in _PHONE_CACHE:
        return _PHONE_CACHE[target_sr]

    try:
        from torchaudio.utils import download_asset

        SAMPLE_RIR = download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav"
        )
        SAMPLE_NOISE = download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav"
        )
        rir_raw, rir_sr = torchaudio.load(SAMPLE_RIR)
        noise_raw, noise_sr = torchaudio.load(SAMPLE_NOISE)
    except Exception as e:
        raise RuntimeError(f"failed to load phone assets: {e}")

    if rir_sr != target_sr:
        rir_raw = tf_resample(rir_raw, rir_sr, target_sr)
    if noise_sr != target_sr:
        noise = tf_resample(noise_raw, noise_sr, target_sr)

    rir = rir_raw[:, int(target_sr * 1.01) : int(target_sr * 1.3)]
    rir = rir / torch.linalg.vector_norm(rir, ord=2)

    _PHONE_CACHE[target_sr] = (rir, noise)
    return rir, noise


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

datasets = [
    "/dataset/CommonVoice",
    "/dataset/LibriSpeech_wav",
    "/dataset/LJSpeech-1.1",
    "/dataset/VoxCeleb_wav",
]

for dataset in datasets:
    train_config["path"]["raw_path"] = dataset
    dev_audios = MyDataset(
        process_config=process_config, train_config=train_config, flag="test"
    )
    dev_audios_loader = DataLoader(
        dev_audios,
        batch_size=4,
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

    import csv
    from pathlib import Path

    results_dir = Path("results/logs")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        results_dir / "MSE_loudness_split_frequency_adaptive_soft_vad_clean_eval.csv"
    )

    # write header once if file does not exist yet
    file_exists = csv_path.exists()

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "dataset",
                    "avg_snr_db",
                    "avg_acc_msg0",
                    "avg_acc_msg1",
                    "num_samples",
                ]
            )
        writer.writerow(
            [
                dataset,  # this is the dataset path from your for dataset in datasets loop
                float(
                    test_avg_snr.item()
                    if torch.is_tensor(test_avg_snr)
                    else test_avg_snr
                ),
                float(test_avg_acc[0]),
                float(test_avg_acc[1]),
                len(dev_audios),
            ]
        )

    from distortions.dl import distortion
    import torch
    from collections import defaultdict

    eps = 1e-12

    def to_B1T(x):
        # Ensure [B, 1, T]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:  # [B, T] -> [B, 1, T]
            x = x.unsqueeze(1)
        return x

    def to_BT(x):
        # Ensure [B, T]
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        return x

    # ---- Instantiate distortion module
    attacker = distortion(process_config).to(device)

    attack_names = {
        0: "none",
        1: "crop(rand small)",
        2: "crop(10%)",
        3: "resample(noop)",
        4: "crop_front",
        5: "crop_middle",
        6: "crop_back",
        7: "resample->22.05k->back",
        8: "resample->8k->back",
        9: "gaussian_noise",
        10: "amp_scale",
        11: "mp3",
        12: "recount(8bit)",
        13: "median_filter",
        14: "low_pass_2k",
        15: "high_pass_500",
        16: "modify_mel",
        19: "crop_mel_wave_front",
        20: "crop_mel_wave_back",
        22: "crop_mel_wave_position(1..10)",
        24: "crop_mel_wave_position_5bins(1..20)",
        26: "crop_mel_wave_position_20bins(1..5)",
        27: "benign_reencode",
        28: "benign_noise_suppression",
        29: "benign_compression",
        30: "benign_phone_disortion",
    }
    # Only include attacks that return WAVEFORMS (skip 17/18/21/23/25 which return spectrograms)
    wave_attacks = [
        27,
        28,
        29,
        30,
    ]

    # Ratios for attacks that use it (tweak as needed)
    ratio_by_attack = {
        4: 10,
        5: 10,
        6: 10,  # 10% cropping
        9: 10,  # noise SNR (dB)
        10: 50,  # amplitude scale (%)
        11: 64,  # mp3 kbps
        13: 3,  # median filter window
        16: 50,  # mel mag scale (%)
        19: 30,
        20: 30,  # mask top % then invert
        22: 5,  # position in 1..10
        24: 5,  # position in 1..20
        26: 2,  # position in 1..5
    }

    print("Number of samples:", len(dev_audios))

    with torch.inference_mode():
        encoder.eval()
        decoder.eval()

        # accumulate per-attack, averaging *per batch*
        sums = defaultdict(
            lambda: {
                "acc_batch_sum": 0.0,  # sum of per-batch accuracies
                "snr_total_sum": 0.0,  # sum of per-batch SNR (vs clean)
                "snr_vs_wm_sum": 0.0,  # sum of per-batch SNR (vs watermarked)
                "batch_count": 0,
            }
        )

        for sample in track(dev_audios_loader):
            wav = sample["matrix"].to(device)  # [B,T] or [B,1,T]
            wav_BT = to_BT(wav)  # [B,T]
            wav_B1T = to_B1T(wav_BT)  # [B,1,T]
            B, T_ref = wav_BT.size(0), wav_BT.size(1)

            # message per sample
            msg = generate_random_msg(
                B, train_config["watermark"]["length"], device
            )  # [B,L]
            tgt = msg >= 0

            # watermark and watermarked audio
            watermark, _ = encoder(wav_BT, msg, 1)  # [B,T]
            y_wm_BT = wav_BT + watermark  # [B,T]
            y_wm_B1T = to_B1T(y_wm_BT)  # [B,1,T]

            # evaluate each attack on y_wm
            for aid in wave_attacks:
                ratio = ratio_by_attack.get(aid, 10)

                # apply attack (may change length)
                y_dist_B1T = attacker(
                    y_wm_B1T, attack_choice=aid, ratio=ratio
                )  # [B,1,T']
                y_dist_BT = to_BT(y_dist_B1T)  # [B,T']

                # ---- batch accuracy (bits): correct_bits / total_bits for this batch
                decoded = decoder(y_dist_BT, 1)  # logits [B,L]
                batch_acc = (decoded[0] >= 0).eq(tgt).float().mean().item()

                # ---- batch SNRs (mean across samples in the batch)
                T_hat = y_dist_BT.size(1)
                T = min(T_ref, T_hat)
                wav_cut = wav_BT[:, :T]
                y_wm_cut = y_wm_BT[:, :T]
                y_dist_cut = y_dist_BT[:, :T]

                sig_pow = (wav_cut**2).mean(dim=1)  # [B]
                noise_tot = (y_dist_cut - wav_cut) ** 2
                noise_pow = noise_tot.mean(dim=1)  # [B]
                snr_total = 10.0 * torch.log10(
                    (sig_pow + eps) / (noise_pow + eps)
                )  # [B]

                wm_pow = (y_wm_cut**2).mean(dim=1)
                noise_wm = (y_dist_cut - y_wm_cut) ** 2
                noise_wm_p = noise_wm.mean(dim=1)
                snr_vs_wm = 10.0 * torch.log10(
                    (wm_pow + eps) / (noise_wm_p + eps)
                )  # [B]

                # accumulate per-attack, per-batch
                sums[aid]["acc_batch_sum"] += batch_acc
                sums[aid]["snr_total_sum"] += snr_total.mean().item()
                sums[aid]["snr_vs_wm_sum"] += snr_vs_wm.mean().item()
                sums[aid]["batch_count"] += 1

        # ---- Report (same format)
        print("\n=== Per-attack results (average over batches) ===")
        header = "ID  Attack".ljust(32) + "Acc     SNR_total(dB)  SNR_vsWM(dB)"
        print(header)
        print("-" * len(header))
        for aid in wave_attacks:
            nb = max(1, sums[aid]["batch_count"])
            avg_acc_batch = sums[aid]["acc_batch_sum"] / nb  # in [0,1]
            avg_snr_tot = sums[aid]["snr_total_sum"] / nb
            avg_snr_wm = sums[aid]["snr_vs_wm_sum"] / nb
            name = f"{aid}: {attack_names.get(aid,'?')}".ljust(32)
            print(
                f"{name}{avg_acc_batch:0.4f}   {avg_snr_tot:10.3f}     {avg_snr_wm:10.3f}"
            )

        robust_csv_path = (
            results_dir
            / "MSE_loudness_split_frequency_adaptive_soft_vad_robust_eval.csv"
        )
        file_exists = robust_csv_path.exists()

        with open(robust_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "dataset",
                        "attack_id",
                        "attack_name",
                        "avg_acc",
                        "avg_snr_total_db",
                        "avg_snr_vs_watermarked_db",
                        "batches_contributed",
                    ]
                )

            for aid in wave_attacks:
                nb = max(1, sums[aid]["batch_count"])
                avg_acc_batch = sums[aid]["acc_batch_sum"] / nb
                avg_snr_tot = sums[aid]["snr_total_sum"] / nb
                avg_snr_wm = sums[aid]["snr_vs_wm_sum"] / nb
                writer.writerow(
                    [
                        dataset,
                        aid,
                        attack_names.get(aid, "?"),
                        avg_acc_batch,
                        avg_snr_tot,
                        avg_snr_wm,
                        nb,
                    ]
                )
