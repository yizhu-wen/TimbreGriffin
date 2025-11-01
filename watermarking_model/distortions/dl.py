import torch
import random
import torch.nn as nn
import numpy as np
import julius
from audiomentations import Compose, Mp3Compression
import kornia
from distortions.frequency2 import fixed_STFT
from io import BytesIO
from pydub import AudioSegment
from torchaudio.functional import fftconvolve, add_noise
from torchaudio.functional import resample as tf_resample
import torchaudio
import subprocess
from io import BytesIO
import wave
from typing import Dict, Tuple, List


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


class distortion(nn.Module):
    def __init__(self, process_config):
        super(distortion, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sample_rate = process_config["audio"]["or_sample_rate"]
        self.resample_kernel1 = julius.ResampleFrac(self.sample_rate, 22050).to(
            self.device
        )
        self.resample_kernel1_re = julius.ResampleFrac(22050, self.sample_rate).to(
            self.device
        )
        self.resample_kernel2 = julius.ResampleFrac(self.sample_rate, 8000).to(
            self.device
        )
        self.resample_kernel2_re = julius.ResampleFrac(
            8000,
            self.sample_rate,
        ).to(self.device)
        self.augment = Compose([Mp3Compression(p=1.0, min_bitrate=64, max_bitrate=64)])
        self.band_lowpass = julius.LowPassFilter(5000 / self.sample_rate).to(
            self.device
        )
        self.band_highpass = julius.HighPassFilter(500 / self.sample_rate).to(
            self.device
        )
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        ).to(self.device)

    def none(self, x):
        return x

    def crop(self, x):
        length = x.shape[2]
        if length > 18000:
            start = random.randint(0, 1000)
            end = random.randint(1, 1000)
            y = x[:, :, start : 0 - end]
            # print(f"start:{start} and end:{end}")
            # pdb.set_trace()
        else:
            y = x
        return y

    def crop2(self, x):
        length = x.shape[2]
        if length > 18000:
            # import pdb
            # pdb.set_trace()
            cut_len = int(length * 0.1)  # cut 10% off
            start = random.randint(0, cut_len - 1)
            end = cut_len - start
            y = x[:, :, start : 0 - end]
            # print(f"start:{start} and end:{end}")
            # pdb.set_trace()
        else:
            y = x
        return y

    # def resample(self, x):
    #     return x

    def crop_front(self, x, cut_ratio=10):
        cut_len = int(x.shape[-1] * (cut_ratio / 100))
        ret = x[:, :, cut_len:]
        # print(f"{x.shape}:{ret.shape}")
        return ret

    def crop_middle(self, x, cut_ratio=10):
        cut_len = int(x.shape[-1] * (cut_ratio / 100))
        begin = int((x.shape[-1] - cut_len) / 2)
        end = begin + cut_len
        # return torch.cat(x[:,:,:begin], x[:,:,end:],dim=2)
        ret = torch.cat([x[:, :, :begin], x[:, :, end:]], dim=2)
        return ret

    def crop_back(self, x, cut_ratio=10):
        cut_len = int(x.shape[-1] * (cut_ratio / 100))
        begin = int((x.shape[-1] - cut_len))
        # return x[:,:,:begin]
        ret = x[:, :, :begin]
        # print(f"{x.shape}:{ret.shape}")
        return ret

    def resample1(self, y):
        y = self.resample_kernel1_re(self.resample_kernel1(y))
        return y

    def resample2(self, y):
        y = self.resample_kernel2_re(self.resample_kernel2(y))
        return y

    def white_noise(self, y, ratio=10):  # SNR = 10log(ps/pn)
        SNR = ratio
        mean = 0.0
        RMS_s = torch.sqrt(torch.mean(y**2, dim=2))
        RMS_n = torch.sqrt(RMS_s**2 / (pow(10, SNR / 20)))
        for i in range(y.shape[0]):
            noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
            if i == 0:
                batch_noise = noise
            else:
                batch_noise = torch.cat((batch_noise, noise), dim=0)
        batch_noise = batch_noise.unsqueeze(1).to(self.device)
        signal_edit = y + batch_noise
        return signal_edit

    def change_top(self, y, ratio=50):
        y = y * ratio / 100
        return y

    def mp3(self, y, ratio=64):
        self.augment = Compose(
            [Mp3Compression(p=1.0, min_bitrate=ratio, max_bitrate=ratio)]
        )
        f = []
        a = y.cpu().detach().numpy()
        for i in a:
            f.append(torch.Tensor(self.augment(i, sample_rate=self.sample_rate)))
        f = torch.cat(f, dim=0).unsqueeze(1).to(self.device)
        # y = y + (f - y).detach()
        # return y
        return f

    def recount(self, y):
        # y2 = torch.tensor(np.array(y.cpu().squeeze(0).data.numpy()*(2**7)).astype(np.int8)) / (2**7)
        y2 = (
            torch.tensor(
                (
                    np.array(y.cpu().squeeze(0).data.numpy() + 1) / 2 * (2**8 - 1)
                ).astype(np.uint8)
            )
            / (2**8 - 1)
            * 2
            - 1
        )
        y2 = y2.to(self.device)
        y = y + (y2 - y).detach()
        return y

    def medfilt(self, y, ratio=3):
        y = kornia.filters.median_blur(y.unsqueeze(1), (1, ratio)).squeeze(1)
        return y

    def low_band_pass(self, y):
        y = self.band_lowpass(y)
        return y

    def high_band_pass(self, y):
        y = self.band_highpass(y)
        return y

    def modify_mel(self, y, ratio=50):
        num_samples = y.shape[2]
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        spect = spect * ratio / 100
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y

    def crop_mel_front(self, y, ratio=50):
        num_samples = y.shape[2]
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len * (ratio / 100))
        spect = spect * (
            torch.cat(
                [
                    torch.zeros(_, cut_len, time_len),
                    torch.ones(_, fre_len - cut_len, time_len),
                ],
                dim=1,
            ).to(self.device)
        )
        return spect

    def crop_mel_back(self, y, ratio=50):
        num_samples = y.shape[2]
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len * (ratio / 100))
        spect = spect * (
            torch.cat(
                [
                    torch.ones(_, fre_len - cut_len, time_len),
                    torch.zeros(_, cut_len, time_len),
                ],
                dim=1,
            ).to(self.device)
        )
        return spect

    def crop_mel_wave_front(self, y, ratio=50):
        num_samples = y.shape[2]
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len * (ratio / 100))
        spect = spect * (
            torch.cat(
                [
                    torch.zeros(_, cut_len, time_len),
                    torch.ones(_, fre_len - cut_len, time_len),
                ],
                dim=1,
            ).to(self.device)
        )
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y

    def crop_mel_wave_back(self, y, ratio=50):
        num_samples = y.shape[2]
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len * (ratio / 100))
        spect = spect * (
            torch.cat(
                [
                    torch.ones(_, fre_len - cut_len, time_len),
                    torch.zeros(_, cut_len, time_len),
                ],
                dim=1,
            ).to(self.device)
        )
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y

    def crop_mel_position(self, y, ratio=1):
        assert ratio >= 1 and ratio <= 10, "a must be an integer between 1 and 10"
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len * (1 / 10))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        # spect = spect*(torch.cat([torch.zeros(_,cut_len,time_len),torch.ones(_,fre_len-cut_len,time_len)], dim=1).to(self.device))
        return spect

    def crop_mel_wave_position(self, y, ratio=1):
        num_samples = y.shape[2]
        assert ratio >= 1 and ratio <= 10, "a must be an integer between 1 and 10"
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len * (1 / 10))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y

    def crop_mel_position_5(self, y, ratio=1):
        assert ratio >= 1 and ratio <= 20, "a must be an integer between 1 and 20"
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len * (1 / 20))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        # spect = spect*(torch.cat([torch.zeros(_,cut_len,time_len),torch.ones(_,fre_len-cut_len,time_len)], dim=1).to(self.device))
        return spect

    def crop_mel_wave_position_5(self, y, ratio=1):
        num_samples = y.shape[2]
        assert ratio >= 1 and ratio <= 20, "a must be an integer between 1 and 20"
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len * (1 / 20))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y

    def crop_mel_position_20(self, y, ratio=1):
        assert ratio >= 1 and ratio <= 5, "a must be an integer between 1 and 5"
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len * (1 / 5))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        # spect = spect*(torch.cat([torch.zeros(_,cut_len,time_len),torch.ones(_,fre_len-cut_len,time_len)], dim=1).to(self.device))
        return spect

    def crop_mel_wave_position_20(self, y, ratio=1):
        num_samples = y.shape[2]
        assert ratio >= 1 and ratio <= 5, "a must be an integer between 1 and 5"
        yBT = self._as_BT(y)
        spect, phase, _ = self.stft.transform(yBT)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len * (1 / 5))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y

    def benign_reencode(
        self, x: torch.Tensor, ratio, dither: bool = True, passes: int = 3
    ) -> torch.Tensor:
        """
        x: [B, 1, T], float in [-1, 1]
        returns: [B, 1, T]
        Each pass simulates: ffmpeg -i input.wav -acodec pcm_s16le output.wav
        """
        assert x.ndim == 3 and x.size(1) == 1, "expected [B, 1, T]"
        assert passes >= 1, "passes must be >= 1"

        B, _, T = x.shape
        device = x.device
        outs = []

        def _one_pass_f32_to_pcm16_back(y_f32: np.ndarray) -> np.ndarray:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "f32le",
                "-ar",
                str(int(self.sample_rate)),
                "-ac",
                "1",
                "-i",
                "pipe:0",
            ]
            if not dither:
                cmd += ["-af", "aresample=dither_method=none"]
            cmd += [
                "-f",
                "wav",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(int(self.sample_rate)),
                "-ac",
                "1",
                "pipe:1",
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            wav_bytes, _ = proc.communicate(y_f32.tobytes())

            bio = BytesIO(wav_bytes)
            with wave.open(bio, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
            y_int16 = np.frombuffer(frames, dtype=np.int16)
            return y_int16.astype(np.float32) / 32768.0  # back to float32 in [-1, 1]

        for b in range(B):
            y = x[b, 0].detach().cpu().numpy().astype(np.float32)  # [T]
            for _ in range(passes):
                y = _one_pass_f32_to_pcm16_back(y)

            # enforce exact length T, trim or pad if a decoder adds or drops samples
            if y.shape[0] > T:
                y = y[:T]
            elif y.shape[0] < T:
                y = np.pad(y, (0, T - y.shape[0]))

            outs.append(y)

        y_np = np.stack(outs, axis=0)  # [B, T]
        return torch.from_numpy(y_np).unsqueeze(1).to(device)  # [B, 1, T]

    def benign_noise_suppression(
        self,
        x: torch.Tensor,
        ratio,
        energy_threshold=0.01,
        frame_size=400,
        hop_size=160,
    ):
        """
        waveform: [B, 1, T], float, roughly [-1, 1]
        returns:  [B, 1, T]
        """
        assert x.ndim == 3 and x.size(1) == 1, "expected [B, 1, T]"
        B, _, T = x.shape
        y = x.clone()

        # for each item in the batch, apply the same framewise RMS gate
        for b in range(B):
            x = y[b, 0]  # [T]
            for start in range(0, T - frame_size + 1, hop_size):
                frame = x[start : start + frame_size]
                energy = torch.sqrt((frame**2).mean())
                if energy < energy_threshold:
                    x[start : start + frame_size] = 0.0

        return y

    def benign_compression(
        self,
        x: torch.Tensor,
        ratio,
        sample_rate: int = 16000,
        codec: str = "mp3",
        bitrate: str = "128k",
    ):
        """
        waveform: [B, 1, T], float in [-1, 1]
        returns:  [B, 1, T] after lossy compression → decode roundtrip
        Simulates: ffmpeg -i input.wav -b:a 128k temp.mp3; ffmpeg -i temp.mp3 output.wav
        """
        assert x.ndim == 3 and x.size(1) == 1, "expected [B, 1, T]"
        B, _, T = x.shape
        device = x.device
        x_np = x.detach().cpu().numpy()  # [B, 1, T]
        out = []

        for b in range(B):
            # Extract channel = 1 → [T], convert to float32
            wav = x_np[b, 0].astype(np.float32)

            # Convert to int16 PCM for pydub input
            pcm16 = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)

            # Encode to MP3 (or codec provided)
            buf = BytesIO()
            AudioSegment(
                pcm16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # int16
                channels=1,
            ).export(buf, format=codec, bitrate=bitrate)

            # Decode MP3 back to PCM
            buf.seek(0)
            decoded = AudioSegment.from_file(buf)

            # Convert back to float32 [-1, 1]
            y = np.array(decoded.get_array_of_samples()).astype(np.float32) / 32768.0

            # Ensure output length remains T
            if y.shape[0] > T:
                y = y[:T]
            elif y.shape[0] < T:
                y = np.pad(y, (0, T - y.shape[0]))

            out.append(y)

        # Stack back to [B, 1, T]
        y_np = np.stack(out, axis=0)  # [B, T]
        y_t = torch.from_numpy(y_np).unsqueeze(1).to(device)  # [B, 1, T]
        return y_t

    def benign_phone_distortion(self, x: torch.Tensor, ratio, sample_rate: int = 16000):
        # Load demo assets and resample to sample_rate
        x = x.squeeze(1)
        rir, _ = _get_phone_assets(sample_rate)
        rir = rir.to(x.device)
        noise = torch.randn_like(x)
        # Apply RIR
        rir_applied = fftconvolve(x, rir, mode="same")
        snr_db = torch.randint(20, 26, (1,), device=x.device)
        bg_added = add_noise(rir_applied, noise, snr_db)

        y_d = julius.bandpass_filter(
            bg_added,
            cutoff_low=500 / sample_rate,
            cutoff_high=2000 / sample_rate,
        )
        y_d = y_d.unsqueeze(1)
        return y_d

    # ----- helper (optional) -----
    def _as_BT(self, x):  # [B,1,T] -> [B,T]
        return x.squeeze(1) if x.dim() == 3 else x

    def _as_B1T(self, x):  # [B,T] -> [B,1,T]
        return x.unsqueeze(1) if x.dim() == 2 else x

    def _to_numpy(self, x):
        return x.numpy().squeeze() if isinstance(x, torch.Tensor) else x

    def _to_tensor(self, x):
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)

    def forward(self, x, attack_choice=1, ratio=10):
        attack_functions = {
            0: lambda x: self.none(x),
            1: lambda x: self.crop(x),
            2: lambda x: self.crop2(x),
            3: lambda x: self.resample(x),
            4: lambda x: self.crop_front(x, ratio),  # Cropping front
            5: lambda x: self.crop_middle(x, ratio),  # Cropping middle
            6: lambda x: self.crop_back(x, ratio),  # Cropping behind
            7: lambda x: self.resample1(x),  # Resampling 16KHz
            8: lambda x: self.resample2(x),  # Resampling 8KHz
            9: lambda x: self.white_noise(
                x, ratio
            ),  # Gaussian Noise with SNR ratio/2 dB
            10: lambda x: self.change_top(x, ratio),  # Amplitude Scaling ratio%
            11: lambda x: self.mp3(x, ratio),  # MP3 Compression ratio Kbps
            12: lambda x: self.recount(x),  # Recount 8 bps
            13: lambda x: self.medfilt(
                x, ratio
            ),  # Median Filtering with ratio samples as window
            14: lambda x: self.low_band_pass(x),  # Low Pass Filtering 2000 Hz
            15: lambda x: self.high_band_pass(x),  # High Pass Filtering 500 Hz
            16: lambda x: self.modify_mel(x, ratio),  # don't need
            17: lambda x: self.crop_mel_front(x, ratio),  # don't need
            18: lambda x: self.crop_mel_back(x, ratio),  # don't need
            19: lambda x: self.crop_mel_wave_front(x, ratio),  # don't need
            20: lambda x: self.crop_mel_wave_back(
                x, ratio
            ),  # mask from top with ratio "ratio" and transform back to wav
            21: lambda x: self.crop_mel_position(
                x, ratio
            ),  # mask 10% at position "ratio"
            22: lambda x: self.crop_mel_wave_position(
                x, ratio
            ),  # mask 10% at position "ratio" and transform back to wav
            23: lambda x: self.crop_mel_position_5(
                x, ratio
            ),  # mask 5% at position "ratio"
            24: lambda x: self.crop_mel_wave_position_5(
                x, ratio
            ),  # mask 5% at position "ratio" and transform back to wav
            25: lambda x: self.crop_mel_position_20(
                x, ratio
            ),  # mask 20% at position "ratio"
            26: lambda x: self.crop_mel_wave_position_20(
                x, ratio
            ),  # mask 20% at position "ratio" and transform back to wav
            27: lambda x: self.benign_reencode(x, ratio),
            28: lambda x: self.benign_noise_suppression(x, ratio),
            29: lambda x: self.benign_compression(x, ratio),
            30: lambda x: self.benign_phone_distortion(x, ratio),
        }

        x = x.clamp(-1, 1)
        y = attack_functions[attack_choice](x)
        if isinstance(y, list):
            # Apply clamp(-1, 1) to each tensor in the list
            y = [tensor.clamp(-1, 1) for tensor in y]
        else:
            y = y.clamp(-1, 1)
        return y
