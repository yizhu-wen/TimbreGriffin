from base64 import encode
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from .blocks import (
    FCBlock,
    Conv2Encoder,
    WatermarkEmbedder,
    WatermarkExtracter,
    ReluBlock,
)
from distortions.frequency import TacotronSTFT, fixed_STFT, tacotron_mel
import julius
import math
import torch.nn.functional as F
from silero_vad import load_silero_vad
from torchaudio.functional import resample as tf_resample
from torchaudio.functional import fftconvolve, add_noise

import torchaudio
from typing import Dict, Tuple
import random

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


def save_spectrum(y, flag="linear"):
    import numpy as np
    import os
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    # Directory to save figures
    root = "draw_figure"
    os.makedirs(root, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.specgram(
        y, Fs=16000, NFFT=320, noverlap=160, window=np.hanning(320), cmap="magma"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Amplitude Spectrogram")
    plt.tight_layout()
    plt.savefig(
        os.path.join(root, f"{flag}_amplitude_spectrogram.png"),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()


def save_spectrum_normal(y, flag="linear"):
    import numpy as np
    import os
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y = y / peak

    # Directory to save figures
    root = "draw_figure"
    os.makedirs(root, exist_ok=True)

    plt.figure(figsize=(10, 4))

    # Compute the spectrogram
    Pxx, freqs, bins, im = plt.specgram(
        y, Fs=16000, NFFT=320, noverlap=160, cmap="magma"
    )

    Pxx_dB = librosa.amplitude_to_db(Pxx, ref=np.max)

    # Clear previous plot and redraw with log values
    plt.clf()
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(bins, freqs, Pxx_dB, shading="auto", cmap="magma")

    plt.colorbar(format="%+2.0f dB")
    plt.title("Log Amplitude Spectrogram")
    plt.tight_layout()
    plt.savefig(
        os.path.join(root, f"{flag}_amplitude_spectrogram.png"),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()


def save_feature_map(feature_maps):
    import os
    import matplotlib.pyplot as plt
    import librosa
    import numpy as np
    import librosa.display

    feature_maps = feature_maps.cpu().numpy()
    root = "draw_figure"
    output_folder = os.path.join(root, "feature_map_or")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    n_channels = feature_maps.shape[0]
    for channel_idx in range(n_channels):
        fig, ax = plt.subplots()
        ax.imshow(feature_maps[channel_idx, :, :], cmap="gray")
        ax.axis("off")
        output_file = os.path.join(
            output_folder, f"feature_map_channel_{channel_idx + 1}.png"
        )
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


def save_waveform(a_tensor, flag="original"):
    import os
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile

    root = "draw_figure"
    y = a_tensor.detach().cpu().numpy()
    soundfile.write(os.path.join(root, flag + "_waveform.wav"), y, samplerate=16000)
    # D = librosa.stft(y)
    # spectrogram = np.abs(D)
    # img=librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=22050, x_axis='time', y_axis='log', y_coords=None);
    # plt.axis('off')
    # plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram_from_waveform.png'), bbox_inches='tight', pad_inches=0.0)


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
        self.original_sample_rate = process_config["audio"]["or_sample_rate"]
        self.mel_transform = TacotronSTFT(
            filter_length=process_config["mel"]["n_fft"],
            hop_length=process_config["mel"]["hop_length"],
            win_length=process_config["mel"]["win_length"],
        )
        self.vocoder_step = model_config["structure"]["vocoder_step"]
        self.win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.hop_length = process_config["mel"]["hop_length"]
        self.distortion = train_config["optimize"]["distortion"]
        self.cutoff_freq_low = 500
        self.cutoff_freq_high = 2000
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

    def forward(self, y, global_step=1):
        y_identity = y
        if self.distortion:
            # Load demo assets and resample to sample_rate
            rir, _ = _get_phone_assets(self.original_sample_rate)
            rir = rir.to(y.device)
            noise = torch.randn_like(y)
            # Apply RIR
            rir_applied = fftconvolve(y, rir, mode="same")
            snr_db = torch.randint(20, 26, (1,), device=y.device)
            bg_added = add_noise(rir_applied, noise, snr_db)

            y_d = julius.bandpass_filter(
                bg_added,
                cutoff_low=self.cutoff_freq_low / self.original_sample_rate,
                cutoff_high=self.cutoff_freq_high / self.original_sample_rate,
            )

        else:
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


class Discriminator(nn.Module):
    def __init__(self, process_config):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            ReluBlock(2, 16, 3, 1, 1),
            ReluBlock(16, 32, 3, 1, 1),
            ReluBlock(32, 64, 3, 1, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.linear = nn.Linear(64, 1)
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        )

    def forward(self, x):
        _, _, stft_result = self.stft.transform(x)
        x = self.conv(stft_result)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x
