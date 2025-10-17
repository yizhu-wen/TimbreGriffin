# decoder_data_generation.py
# Build a reproducible watermark-decoder dataset from VoxCeleb.
# This version:
# - Samples ACROSS FILES (not one-per-video)
# - Per source file, creates out_root/<idXXXX>/<video_id>/<file_stem>/...
# - Writes benign_identity.wav as the pure watermarked audio (no trim/pad/RMS)
# - Applies other benign/malicious ops (POST_RMS_MATCH applies only to those)
# - Adds new benign_phone_recording op (simulated phone chain: RIR + noise + telephone EQ + G.722)
# - Preserves selection order in dataset_index.jsonl (even with parallel processing)
# - Supports sequential or random selection via --selection


import hashlib
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import yaml
from torchaudio.functional import fftconvolve, add_noise
from torchaudio.functional import resample as tf_resample

# Replace with your actual module path that defines Encoder
from model.conv2_mel_modules import Encoder  # type: ignore

# Optional codec support (benign_compression, benign_reencode)
try:
    from pydub import AudioSegment

    _HAVE_PYDUB = True
except Exception:
    _HAVE_PYDUB = False

# Optional AudioEffector support (phone chain)
try:
    from torchaudio.io import AudioEffector

    _HAVE_EFFECTOR = True
except Exception:
    _HAVE_EFFECTOR = False

# -----------------------------
# Config
# -----------------------------
DEFAULT_SEED = 1337
SAMPLE_COUNT = 3800  # set -1 to process ALL eligible files
ACCEPT_EXTS = {".wav", ".flac", ".mp3", ".m4a"}
USE_GPU = True
POST_RMS_MATCH = True  # applied to distorted ops, never to benign_identity

# Phone EQ (used by AudioEffector)
PHONE_EQ_EFFECT = (
    "lowpass=frequency=4000:poles=1,"
    "compand=attacks=0.02:decays=0.05:"
    "points=-60/-60|-30/-10|-20/-8|-5/-8|-2/-8:"
    "gain=-8:volume=-7:delay=0.05"
)


# -----------------------------
# Utils
# -----------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def deterministic_shuffle(items: List, seed: int) -> List:
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def atomic_write_json(obj: dict, out_path: str):
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def atomic_write_bytes(b: bytes, out_path: str):
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, out_path)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ensure_2d_mono(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.view(1, -1)
    if x.dim() == 2:
        return x.mean(dim=0, keepdim=True) if x.size(0) > 1 else x
    if x.dim() == 3:
        b, c, t = x.shape
        x = x.mean(dim=1, keepdim=True) if c > 1 else x
        return x[0]
    raise ValueError(f"Unexpected audio shape {tuple(x.shape)}")


def shape_str(x: torch.Tensor) -> str:
    return "x".join(str(d) for d in x.shape)


def _to_numpy(waveform: torch.Tensor) -> np.ndarray:
    w = ensure_2d_mono(waveform)
    return w[0].detach().cpu().float().numpy()


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).view(1, -1)


def clamp01(wav: torch.Tensor) -> torch.Tensor:
    return torch.clamp(wav, -1.0, 1.0)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x.pow(2).mean().clamp_min(1e-12)))


def _rms_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    rx = _rms(x)
    rr = _rms(ref)
    return x * (rr / rx)


def _frame_params(sr: int, ms_frame: int = 25, ms_hop: int = 10) -> Tuple[int, int]:
    fs = max(160, int(sr * ms_frame / 1000))
    hs = max(80, int(sr * ms_hop / 1000))
    return fs, hs


def _crossfade(a: torch.Tensor, b: torch.Tensor, fade_len: int) -> torch.Tensor:
    a = ensure_2d_mono(a)
    b = ensure_2d_mono(b)
    L = min(fade_len, a.size(-1), b.size(-1))
    if L <= 0:
        return torch.cat([a, b], dim=-1)
    ramp = torch.linspace(0.0, 1.0, L, dtype=a.dtype, device=a.device).view(1, -1)
    a_tail = a[..., -L:] * (1.0 - ramp)
    b_head = b[..., :L] * ramp
    mid = a_tail + b_head
    return torch.cat([a[..., :-L], mid, b[..., L:]], dim=-1)


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    wav = ensure_2d_mono(wav)
    return wav, sr


def save_wav(path: str, wav: torch.Tensor, sr: int, float_pcm: bool = True):
    wav = ensure_2d_mono(wav)
    wav = clamp01(wav)
    ensure_dir(os.path.dirname(path))
    if float_pcm:
        # 32-bit float PCM (no quantization)
        torchaudio.save(path, wav, sr, encoding="PCM_F", bits_per_sample=32)
    else:
        torchaudio.save(path, wav, sr, encoding="PCM_S", bits_per_sample=16)


def min_length_from_sr(sr: int) -> int:
    # int(2*sr + 0.5*sr + 0.5*sr) == 3*sr
    return 3 * sr


def speaker_id_from_path(path: str) -> Optional[str]:
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if p.startswith("id") and p[2:].isdigit():
            return p
    return None


# -----------------------------
# Model integration
# -----------------------------
@dataclass
class WMEncoderCtx:
    device: torch.device
    encoder: Encoder
    msg_len: int


def load_encoder_ctx(
    process_cfg_path: str,
    model_cfg_path: str,
    train_cfg_path: str,
    checkpoint_path: str,
) -> WMEncoderCtx:
    process_config = yaml.load(open(process_cfg_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_cfg_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_cfg_path, "r"), Loader=yaml.FullLoader)

    device = torch.device(
        "cuda:0" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    )
    msg_len = int(train_config["watermark"]["length"])

    enc = Encoder(process_config, model_config, train_config, msg_len).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()

    return WMEncoderCtx(device=device, encoder=enc, msg_len=msg_len)


def _bitstr_to_msg_tensor(
    bitstr: str, total_len: int, device: torch.device, rng: random.Random
) -> torch.Tensor:
    if len(bitstr) > total_len:
        raise ValueError(
            f"Model watermark length {total_len} is less than provided bit length {len(bitstr)}"
        )
    head = [1.0 if c == "1" else -1.0 for c in bitstr]
    tail_bits = [rng.randint(0, 1) for _ in range(total_len - len(bitstr))]
    tail = [1.0 if b == 1 else -1.0 for b in tail_bits]
    vec = np.array(head + tail, dtype=np.float32).reshape(1, 1, total_len)
    return torch.from_numpy(vec).to(device)


def make_encoder_fn(
    ctx: WMEncoderCtx,
) -> Callable[[torch.Tensor, int, str, random.Random], torch.Tensor]:
    def _fn(
        wav_cpu: torch.Tensor, sr: int, bitstr: str, rng: random.Random
    ) -> torch.Tensor:
        x2 = ensure_2d_mono(wav_cpu).to(ctx.device)  # [1, T]
        msg = _bitstr_to_msg_tensor(bitstr, ctx.msg_len, ctx.device, rng)  # [1, 1, L]
        with torch.inference_mode():
            try:
                watermark, _ = ctx.encoder(x2.unsqueeze(0), msg, 1)  # [1, 1, T]
                y = x2.unsqueeze(0) + watermark
                y = y.squeeze(0)  # [1, T]
            except RuntimeError as e:
                if "conv1d" in str(e) and "2D" in str(e):
                    watermark, _ = ctx.encoder(x2, msg, 1)  # [1, T] or [1, 1, T]
                    if watermark.dim() == 2:
                        y = x2 + watermark
                    elif watermark.dim() == 3 and watermark.size(0) == 1:
                        y = x2 + watermark.squeeze(0)
                    else:
                        raise
                else:
                    raise
        return clamp01(y.detach().cpu())

    return _fn


# -----------------------------
# Phone assets (RIR & noise) cache
# -----------------------------
_PHONE_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}  # sr -> (rir, noise)


def _get_phone_assets(target_sr: int = 16000) -> Tuple[torch.Tensor, torch.Tensor]:
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


# -----------------------------
# Worker-scoped effectors (reuse per process)
# -----------------------------
_worker_effectors: Dict[str, object] = {}


def _get_effectors():
    """Create + cache effectors in a worker process."""
    if not _HAVE_EFFECTOR:
        raise RuntimeError("AudioEffector unavailable")
    eq = _worker_effectors.get("eq")
    cdc = _worker_effectors.get("g722")
    if eq is None:
        eq = AudioEffector(effect=PHONE_EQ_EFFECT)
        _worker_effectors["eq"] = eq
    if cdc is None:
        cdc = AudioEffector(format="g722")
        _worker_effectors["g722"] = cdc
    return eq, cdc


# -----------------------------
# AudioProcessor ops
# -----------------------------
class AudioProcessor:
    @staticmethod
    def benign_identity(waveform, sample_rate=None, rng=None):
        # Pure watermarked baseline, no RMS match, no pad or trim
        return ensure_2d_mono(waveform)

    @staticmethod
    def benign_compression(
        waveform, sample_rate, codec="mp3", bitrate="128k", rng=None
    ):
        if not _HAVE_PYDUB:
            raise RuntimeError("pydub or ffmpeg not available for benign_compression")
        x = ensure_2d_mono(waveform)
        wf = _to_numpy(x)
        buffer = BytesIO()
        wf_i16 = (wf * 32767).clip(-32768, 32767).astype(np.int16)
        AudioSegment(
            wf_i16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
        ).export(buffer, format=codec, bitrate=bitrate)
        buffer.seek(0)
        out = (
            np.array(AudioSegment.from_file(buffer).get_array_of_samples()).astype(
                np.float32
            )
            / 32768.0
        )
        T_in = x.size(-1)
        if out.shape[0] > T_in:
            out = out[:T_in]
        elif out.shape[0] < T_in:
            pad = np.zeros((T_in - out.shape[0],), dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        return _to_tensor(out)

    @staticmethod
    def benign_resample(waveform, orig_sr, target_sr=8000, rng=None):
        x = ensure_2d_mono(waveform)
        down = torchaudio.functional.resample(x, orig_sr, target_sr)
        up = torchaudio.functional.resample(down, target_sr, orig_sr)
        T = x.size(1)
        if up.size(1) > T:
            up = up[:, :T]
        elif up.size(1) < T:
            up = torch.nn.functional.pad(up, (0, T - up.size(1)))
        return ensure_2d_mono(up)

    @staticmethod
    def benign_reencode(waveform, sample_rate, passes=3, rng=None):
        if not _HAVE_PYDUB:
            raise RuntimeError("pydub or ffmpeg not available for benign_reencode")
        x = ensure_2d_mono(waveform)
        wf = _to_numpy(x)
        for _ in range(passes):
            buffer = BytesIO()
            wf_i16 = (wf * 32767).clip(-32768, 32767).astype(np.int16)
            AudioSegment(
                wf_i16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
            ).export(buffer, format="wav")
            buffer.seek(0)
            wf = (
                np.array(
                    AudioSegment.from_file(buffer, format="wav").get_array_of_samples()
                ).astype(np.float32)
                / 32768.0
            )
        T_in = x.size(-1)
        if wf.shape[0] > T_in:
            wf = wf[:T_in]
        elif wf.shape[0] < T_in:
            pad = np.zeros((T_in - wf.shape[0],), dtype=np.float32)
            wf = np.concatenate([wf, pad], axis=0)
        return _to_tensor(wf)

    @staticmethod
    def benign_noise_suppression(
        waveform, sr, energy_threshold=0.01, frame_size=None, hop_size=None, rng=None
    ):
        x = ensure_2d_mono(waveform)
        fs, hs = (
            _frame_params(sr)
            if frame_size is None or hop_size is None
            else (frame_size, hop_size)
        )
        x0 = x[0]
        T = x0.size(0)
        for start in range(0, max(T - fs + 1, 1), hs):
            frame = x0[start : start + fs]
            if frame.numel() == 0:
                continue
            energy = torch.sqrt((frame**2).mean())
            if energy < energy_threshold:
                x0[start : start + fs] = 0.0
        return x

    @staticmethod
    def benign_phone_recording(
        waveform,
        sample_rate,
        snr_db: int,
        effect: Optional[str] = None,
        codec: str = "g722",
    ):
        """
        Simulate a phone capture:
        1) Convolve with room impulse response (RIR)
        2) Add background noise at given SNR
        3) Apply lowpass + compand (telephone EQ)
        4) Re-encode with G.722 codec
        Returns a waveform trimmed/padded to match input length.
        """
        if not _HAVE_EFFECTOR:
            raise RuntimeError(
                "torchaudio.io.AudioEffector not available for benign_phone_recording"
            )
        x = ensure_2d_mono(waveform)
        B, T_ref = waveform.shape

        # Load demo assets and resample to sample_rate
        rir, noise = _get_phone_assets(sample_rate)

        # Move to model device and dtype
        rir = rir[:1].to(device=x.device, dtype=x.dtype)  # [1, L_rir]
        noise = noise[:1].to(device=x.device, dtype=x.dtype)  # [1, L_noise]

        rir_applied = fftconvolve(x, rir, mode="same")
        if noise.shape[1] < T_ref:
            reps = (T_ref + noise.shape[1] - 1) // noise.shape[1]
            noise = noise.repeat(1, reps)
        noise = noise[:, : rir_applied.shape[1]]
        # [batch, length]
        bg_added = add_noise(
            rir_applied,
            noise,
            torch.tensor([int(snr_db)]).to(device=x.device, dtype=x.dtype),
        )

        # Effectors (reused per worker)
        eq_eff, cdc_eff = _get_effectors()

        outs: List[torch.Tensor] = []
        # Loop per item: (T,) -> (T,1) CPU -> effect -> codec -> back to (1,T)
        for b in range(B):
            # (T,) -> (T,1) on CPU float32 (effector requirement)
            w_cpu = (
                bg_added[b].detach().to("cpu", torch.float32).unsqueeze(1).contiguous()
            )  # [T,1]
            # Apply EQ/compand
            filtered = eq_eff.apply(w_cpu, sample_rate)  # w_eq: [T2, 1]
            # Apply telephony codec (G.722)
            codec_applied = cdc_eff.apply(filtered, sample_rate)  # w_cdc: [T3, 1]
            y = codec_applied[:, 0].contiguous().view(1, -1)  # [1, T3]
            outs.append(y)  # each [1, T_ref]

        # Stack back to (B, T_ref), then cast back to original dtype/device
        out = torch.cat(outs, dim=0).to(device=x.device, dtype=x.dtype)  # [B, T_ref]
        return out

    # ==== Malicious Operations ====
    @staticmethod
    def malicious_delete(waveform, sample_rate=16000, ratio=0.3, rng=None):
        x = ensure_2d_mono(waveform)
        T = x.size(-1)
        del_len = int(max(8, T * ratio))
        if del_len >= T:
            raise RuntimeError("delete span exceeds length")
        c0 = int(T * 0.1)
        c1 = int(T * 0.9 - del_len)
        start = (T - del_len) // 2 if c1 <= c0 else rng.randint(c0, c1)
        out = torch.cat([x[..., :start], x[..., start + del_len :]], dim=-1)
        return out, {"ratio": ratio, "start": start, "del_len": del_len}

    @staticmethod
    def malicious_silence(
        waveform,
        sample_rate=16000,
        ratio=0.2,
        frame_size=None,
        hop_size=None,
        energy_threshold=0.01,
        rng=None,
    ):
        x = ensure_2d_mono(waveform)
        fs, hs = (
            _frame_params(sample_rate)
            if frame_size is None or hop_size is None
            else (frame_size, hop_size)
        )
        x0 = x[0]
        T = x0.size(0)
        frames = (
            x0.unfold(0, min(fs, max(T, 1)), max(hs, 1)) if T >= fs else x0.view(1, -1)
        )
        energy = torch.sqrt((frames**2).mean(dim=1))
        voiced = (energy > energy_threshold).nonzero(as_tuple=True)[0]
        if len(voiced) == 0:
            return x, {"ratio": ratio, "note": "no voiced frames"}
        start_frame = int(voiced[rng.randint(0, len(voiced) - 1)])
        start = start_frame * hs
        mute_len = int(T * ratio)
        end = min(start + mute_len, T)
        x[..., start:end] = 0
        return x, {"ratio": ratio, "start": start, "end": end}

    @staticmethod
    def malicious_reorder(
        waveform, sample_rate=None, num_segments=None, rng=None, fade_ms=5
    ):
        x = ensure_2d_mono(waveform)
        T = x.shape[-1]
        if num_segments is None:
            num_segments = rng.choice([4, 6, 8])
        if num_segments > max(2, T):
            num_segments = max(2, min(T, 4))
        cut_points = (
            sorted(rng.sample(range(1, T), num_segments - 1))
            if T >= num_segments
            else list(range(1, num_segments))
        )
        seg_bounds = [0] + cut_points + [T]
        segs = [
            x[:, seg_bounds[i] : seg_bounds[i + 1]] for i in range(len(seg_bounds) - 1)
        ]
        rng.shuffle(segs)
        fade_len = int((fade_ms / 1000.0) * (sample_rate if sample_rate else 16000))
        out = segs[0]
        for s in segs[1:]:
            out = _crossfade(out, s, fade_len)
        return out, {"segments": num_segments, "fade_ms": fade_ms}

    @staticmethod
    def malicious_splice(
        waveform, sample_rate, spliced_waveform=None, rng=None, fade_ms=5
    ):
        x = ensure_2d_mono(waveform)
        donor = (
            ensure_2d_mono(spliced_waveform) if spliced_waveform is not None else None
        )
        if donor is None or donor.numel() == 0:
            raise RuntimeError("spliced_waveform unavailable")
        start = rng.randint(0, x.size(-1))
        left = x[..., :start]
        right = x[..., start:]
        fade_len = int((fade_ms / 1000.0) * sample_rate)
        mid = _crossfade(left, donor, fade_len)
        out = torch.cat([mid, right], dim=-1)
        return out, {
            "insert_at": start,
            "donor_len": donor.size(-1),
            "fade_ms": fade_ms,
        }

    @staticmethod
    def malicious_substitute(
        waveform,
        sample_rate,
        replace_waveform=None,
        frame_size=None,
        hop_size=None,
        energy_threshold=0.01,
        rng=None,
        fade_ms=5,
    ):
        x = ensure_2d_mono(waveform)
        donor = (
            ensure_2d_mono(replace_waveform) if replace_waveform is not None else None
        )
        if donor is None or donor.numel() == 0:
            raise RuntimeError("replace_waveform unavailable")
        fs, hs = (
            _frame_params(sample_rate)
            if frame_size is None or hop_size is None
            else (frame_size, hop_size)
        )
        x0 = x[0]
        T = x0.size(0)
        sub_len = donor.size(-1)
        frames = (
            x0.unfold(0, min(fs, max(T, 1)), max(hs, 1)) if T >= fs else x0.view(1, -1)
        )
        energy = torch.sqrt((frames**2).mean(dim=1))
        voiced = (energy > energy_threshold).nonzero(as_tuple=True)[0]
        if len(voiced) == 0:
            start = max(0, (T - sub_len) // 2)
        else:
            valid_starts = voiced * hs
            valid_starts = valid_starts[valid_starts <= max(0, T - sub_len)]
            start = (
                int(valid_starts[rng.randint(0, len(valid_starts) - 1)].item())
                if len(valid_starts) > 0
                else max(0, (T - sub_len) // 2)
            )
        left = x[..., :start]
        right = x[..., start + sub_len :]
        fade_len = int((fade_ms / 1000.0) * sample_rate)
        mid = _crossfade(left, donor, fade_len)
        out = torch.cat([mid, right], dim=-1)
        return out, {"start": start, "sub_len": sub_len, "fade_ms": fade_ms}


# Registry
DISTORTION_REGISTRY = {
    "benign_identity": lambda wav, sr, rng, **kw: (
        AudioProcessor.benign_identity(wav, sr, rng=rng),
        {"note": "pure watermarked baseline"},
    ),
    "benign_compression": lambda wav, sr, rng, **kw: (
        AudioProcessor.benign_compression(wav, sr, rng=rng),
        {"codec": "mp3", "bitrate": "128k"},
    ),
    "benign_resample": lambda wav, sr, rng, **kw: (
        AudioProcessor.benign_resample(wav, sr, rng=rng),
        {"target_sr": 8000},
    ),
    "benign_reencode": lambda wav, sr, rng, **kw: (
        AudioProcessor.benign_reencode(wav, sr, passes=3, rng=rng),
        {"passes": 3},
    ),
    "benign_noise_suppression": lambda wav, sr, rng, **kw: (
        AudioProcessor.benign_noise_suppression(wav, sr, rng=rng),
        {"energy_threshold": 0.01},
    ),
    # "benign_phone_recording_30": lambda wav, sr, rng, **kw: (
    #     AudioProcessor.benign_phone_recording(wav, sr, snr_db=30),
    #     {"snr_db": 20, "codec": "g722"},
    # ),
    # "malicious_delete_0.3": lambda wav, sr, rng, **kw: AudioProcessor.malicious_delete(
    #     wav, sr, ratio=0.3, rng=rng
    # ),
    "malicious_silence_0.2": lambda wav, sr, rng, **kw: AudioProcessor.malicious_silence(
        wav, sr, ratio=0.2, rng=rng
    ),
    # "malicious_reorder": lambda wav, sr, rng, **kw: AudioProcessor.malicious_reorder(
    #     wav, sr, rng=rng
    # ),
    # "malicious_splice": lambda wav, sr, rng, donor=None, **kw: AudioProcessor.malicious_splice(
    #     wav, sr, spliced_waveform=donor, rng=rng
    # ),
    # "malicious_substitute": lambda wav, sr, rng, donor=None, **kw: AudioProcessor.malicious_substitute(
    #     wav, sr, replace_waveform=donor, rng=rng
    # ),
}


# -----------------------------
# Donor selection from same speaker
# -----------------------------
def load_random_segment_from_same_speaker(
    cur_path: str,
    target_len: int,
    rng: random.Random,
    target_sr: int,
    speaker2files: Dict[str, List[str]],
) -> Optional[torch.Tensor]:
    spk = speaker_id_from_path(cur_path)
    if spk is None:
        return None
    pool = [p for p in speaker2files.get(spk, []) if p != cur_path]
    if not pool:
        return None
    donor_path = rng.choice(pool)
    wav_d, sr_d = load_audio(donor_path)
    if sr_d != target_sr:
        wav_d = ensure_2d_mono(tf_resample(wav_d, sr_d, target_sr))
    if wav_d.size(-1) < max(1, target_len):
        return None
    if wav_d.size(-1) == target_len:
        return ensure_2d_mono(wav_d)
    start = rng.randint(0, wav_d.size(-1) - target_len)
    return ensure_2d_mono(wav_d[:, start : start + target_len])


# -----------------------------
# Sampling: collect ELIGIBLE FILES (not one-per-video), then select
# -----------------------------
def find_audio_files(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in ACCEPT_EXTS:
                out.append(os.path.join(dp, fn))
    return out


def is_eligible(path: str) -> bool:
    try:
        info = torchaudio.info(path)
        return info.num_frames > min_length_from_sr(info.sample_rate)
    except Exception:
        return False


def collect_eligible_files(root: str) -> List[str]:
    return [p for p in find_audio_files(root) if is_eligible(p)]


def sample_files(
    files: List[str], k: int, seed: int, mode: str = "sequential"
) -> List[str]:
    # Start from lexicographic path order to keep stable behavior
    sorted_files = sorted(files)
    if k is None or k < 0 or len(sorted_files) <= k:
        return sorted_files
    if mode == "sequential":
        return sorted_files[:k]
    elif mode == "random":
        rng = random.Random(seed)
        idxs = list(range(len(sorted_files)))
        rng.shuffle(idxs)
        sel = [sorted_files[i] for i in idxs[:k]]
        return sorted(sel)  # return in path order
    else:
        raise ValueError(f"unknown selection mode: {mode}")


# -----------------------------
# Per process context
# -----------------------------
_process_ctx = {
    "speaker2files": None,
    "seed": None,
    "voxceleb_root": None,
    "out_root": None,
}


def _worker_init(
    speaker2files: Dict[str, List[str]], seed: int, voxceleb_root: str, out_root: str
):
    _process_ctx["speaker2files"] = speaker2files
    _process_ctx["seed"] = seed
    _process_ctx["voxceleb_root"] = voxceleb_root
    _process_ctx["out_root"] = out_root


def _make_sample_rng(global_seed: int, sample_key: str) -> random.Random:
    h = hashlib.sha256(f"{global_seed}|{sample_key}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


# -----------------------------
# Distortion worker (per FILE folder)
# -----------------------------
@dataclass
class SampleResult:
    ok: bool
    sample_id: str
    dir_rel: str
    meta_rel: str
    error: Optional[str] = None


def _distort_and_write_metadata(
    in_path: str, rel_dir: str, bitstr: str, sr: int
) -> SampleResult:
    try:
        out_root = _process_ctx["out_root"]
        rng = _make_sample_rng(_process_ctx["seed"], rel_dir)
        sdir = os.path.join(out_root, rel_dir)  # idXXXX/VIDEOID/FILESTEM
        base_path = os.path.join(sdir, "benign_identity.wav")

        wm_wav, sr_chk = load_audio(base_path)
        wm_wav = ensure_2d_mono(wm_wav)
        if sr_chk != sr:
            wm_wav = ensure_2d_mono(tf_resample(wm_wav, sr_chk, sr))

        operations_meta = []
        for name, fn in DISTORTION_REGISTRY.items():
            entry = {
                "name": name,
                "distorted_path": None,
                "error": None,
                "validation": None,
            }
            try:
                if name == "benign_identity":
                    entry["distorted_path"] = "benign_identity.wav"
                    entry["validation"] = (
                        f"ok, pure watermarked baseline, in={shape_str(wm_wav)}, sr={sr}"
                    )
                    operations_meta.append(entry)
                    continue

                donor = None
                if name in {"malicious_splice", "malicious_substitute"}:
                    target_len = (
                        int(0.3 * wm_wav.size(-1))
                        if name == "malicious_splice"
                        else int(0.2 * wm_wav.size(-1))
                    )
                    target_len = max(1, min(wm_wav.size(-1) - 1, target_len))
                    donor = load_random_segment_from_same_speaker(
                        in_path, target_len, rng, sr, _process_ctx["speaker2files"]
                    )
                    if donor is None:
                        raise RuntimeError("donor segment unavailable")
                    donor = ensure_2d_mono(donor)

                out_wav, params = fn(wm_wav, sr, rng, donor=donor)
                out_wav = ensure_2d_mono(out_wav)
                if POST_RMS_MATCH:
                    out_wav = _rms_match(out_wav, wm_wav)
                out_wav = clamp01(out_wav)

                out_fname = f"{name}.wav"
                save_wav(os.path.join(sdir, out_fname), out_wav, sr, float_pcm=True)

                entry["distorted_path"] = out_fname
                entry["validation"] = (
                    f"ok, in={shape_str(wm_wav)}, out={shape_str(out_wav)}, sr={sr}, params={params}"
                )
            except Exception as e:
                entry["error"] = str(e)
                entry["validation"] = "skipped, see error"
            operations_meta.append(entry)

        meta = {
            "filepath": "benign_identity.wav",
            "watermark_bits": bitstr,
            "original_sample_rate": sr,
            "min_length_required_samples": min_length_from_sr(sr),
            "operations": operations_meta,
        }
        meta_path = os.path.join(sdir, "metadata.json")
        atomic_write_json(meta, meta_path)

        sample_id = rel_dir.replace(os.sep, "/")
        return SampleResult(
            True, sample_id, rel_dir + "/", os.path.join(rel_dir, "metadata.json")
        )
    except Exception as e:
        return SampleResult(False, rel_dir.replace(os.sep, "/"), "", "", error=str(e))


# -----------------------------
# Coordinator
# -----------------------------
def build_dataset(
    voxceleb_root: str,
    out_root: str,
    process_cfg: str,
    model_cfg: str,
    train_cfg: str,
    checkpoint_path: str,
    sample_count: int = SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    selection: str = "sequential",  # "sequential" or "random"
    max_workers: Optional[int] = None,
) -> None:
    set_global_seed(seed)
    ensure_dir(out_root)

    enc_ctx = load_encoder_ctx(process_cfg, model_cfg, train_cfg, checkpoint_path)
    if enc_ctx.msg_len < 10:
        raise ValueError(
            f"Your model watermark length is {enc_ctx.msg_len}, need at least 10"
        )
    encode_fn = make_encoder_fn(enc_ctx)

    # Build pool of ELIGIBLE FILES (across the whole tree), then select
    all_eligible_files = collect_eligible_files(voxceleb_root)
    chosen_files = sample_files(all_eligible_files, sample_count, seed, mode=selection)

    # speaker -> files map for donor selection
    speaker2files: Dict[str, List[str]] = {}
    for p in all_eligible_files:
        spk = speaker_id_from_path(p)
        if spk:
            speaker2files.setdefault(spk, []).append(p)

    # Bits assignment (deterministic cycle of 10-bit codes)
    def make_bit_pool(bit_len: int, s: int) -> List[str]:
        pool = [format(i, f"0{bit_len}b") for i in range(2**bit_len)]
        rng = random.Random(s)
        rng.shuffle(pool)
        return pool

    pool = make_bit_pool(10, seed)

    if len(chosen_files) == 0:
        raise RuntimeError(
            "No eligible audio files found. Check root path and min-length filter (3*sr)."
        )

    bits_list = [pool[i % len(pool)] for i in range(len(chosen_files))]

    # Provenance
    meta_txt = os.path.join(out_root, "meta.txt")
    sel_rel_files = [
        os.path.relpath(p, voxceleb_root).replace(os.sep, "/") for p in chosen_files
    ]
    meta_lines = [
        f"seed={seed}",
        f"selection={selection}",
        f"voxceleb_root={voxceleb_root}",
        f"out_root={out_root}",
        f"sample_count={len(chosen_files)}",
        f"build_time_unix={int(time.time())}",
        f"encoder_device={enc_ctx.device.type}",
        f"msg_len={enc_ctx.msg_len}",
        "selected_files=" + ",".join(sel_rel_files),
    ]
    atomic_write_bytes(("\n".join(meta_lines) + "\n").encode("utf-8"), meta_txt)

    # To preserve original selection order in the final index
    order_map: Dict[str, int] = {}

    # Stage A: embed watermark and write benign_identity.wav inside per-FILE folder
    prepared: List[Tuple[str, str, str, int]] = []  # (in_path, rel_dir, bitstr, sr)
    for in_path, bitstr in zip(chosen_files, bits_list):
        try:
            # Compute rel_dir consistently: idXXXX/VIDEOID/FILESTEM
            rel_parent = os.path.relpath(
                os.path.dirname(in_path), voxceleb_root
            )  # idXXXX/VIDEOID
            file_stem = os.path.splitext(os.path.basename(in_path))[0]  # e.g., 00001
            rel_dir = os.path.join(rel_parent, file_stem)
            rel_dir_id = rel_dir.replace(os.sep, "/")
            # record order even if this one later fails
            if rel_dir_id not in order_map:
                order_map[rel_dir_id] = len(order_map)

            wav, sr = load_audio(in_path)
            need = min_length_from_sr(sr)
            if wav.size(-1) <= need:
                continue

            sdir = os.path.join(out_root, rel_dir)
            ensure_dir(sdir)

            rng = _make_sample_rng(seed, rel_dir_id)
            wm_wav = encode_fn(
                wav, sr, bitstr, rng
            )  # pure watermarked (no pad/trim/RMS)
            save_wav(
                os.path.join(sdir, "benign_identity.wav"), wm_wav, sr, float_pcm=True
            )

            prepared.append((in_path, rel_dir, bitstr, sr))
        except Exception as e:
            with open(
                os.path.join(out_root, "errors_embed.txt"), "a", encoding="utf-8"
            ) as ef:
                ef.write(f"{os.path.relpath(in_path, voxceleb_root)}\t{str(e)}\n")

    # Stage B: distortions + metadata in parallel
    index_entries = []
    errs = []
    max_workers = (
        max(1, os.cpu_count() or 1) if max_workers in (None, 0, -1) else max_workers
    )
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(speaker2files, seed, voxceleb_root, out_root),
    ) as ex:
        futs = []
        for in_path, rel_dir, bitstr, sr in prepared:
            futs.append(
                ex.submit(_distort_and_write_metadata, in_path, rel_dir, bitstr, sr)
            )
        for fu in as_completed(futs):
            res: SampleResult = fu.result()
            if res.ok:
                index_entries.append(
                    {
                        "id": res.sample_id,
                        "dirpath": res.dir_rel,
                        "metadata_path": res.meta_rel,
                    }
                )
            else:
                errs.append((res.sample_id, res.error))

    # Sort index_entries by original chosen order
    index_entries.sort(key=lambda e: order_map.get(e["id"], 1 << 30))

    # Root index
    idx_path = os.path.join(out_root, "dataset_index.jsonl")
    with open(idx_path, "w", encoding="utf-8") as f:
        for e in index_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    if errs:
        with open(os.path.join(out_root, "errors.txt"), "w", encoding="utf-8") as f:
            for sid, err in errs:
                f.write(f"{sid}\t{err}\n")

    print(f"done, ok={len(index_entries)}, errors={len(errs)}")
    print(f"index at {idx_path}")
    print(f"meta at  {meta_txt}")


# -----------------------------
# Loader (optional, for quick tests)
# -----------------------------
class DecoderDataset(torch.utils.data.Dataset):
    """
    Yields waveform, bits, op. If include_clean is True, yields benign_identity.wav once with op="clean".
    """

    def __init__(self, index_file: str, include_clean: bool = False):
        self.root = os.path.dirname(index_file)
        with open(index_file, "r", encoding="utf-8") as f:
            self.entries = [json.loads(line) for line in f]

        self.items = []
        for e in self.entries:
            meta_path = os.path.join(self.root, e["metadata_path"])
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            base_dir = os.path.join(self.root, e["dirpath"])
            if include_clean:
                self.items.append(
                    {
                        "wav_path": os.path.join(base_dir, "benign_identity.wav"),
                        "bits": meta["watermark_bits"],
                        "op": "clean",
                    }
                )
            for op in meta["operations"]:
                if op["distorted_path"] is not None and op["error"] is None:
                    if include_clean and op["name"] == "benign_identity":
                        continue
                    self.items.append(
                        {
                            "wav_path": os.path.join(base_dir, op["distorted_path"]),
                            "bits": meta["watermark_bits"],
                            "op": op["name"],
                        }
                    )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        wav, sr = load_audio(rec["wav_path"])
        wav = ensure_2d_mono(wav)
        return wav, rec["bits"], rec["op"]


def collate_pad(batch):
    wavs, bits, ops = zip(*batch)
    lengths = torch.tensor([ensure_2d_mono(w).size(-1) for w in wavs], dtype=torch.long)
    maxL = int(lengths.max().item())
    padded = []
    for w in wavs:
        w = ensure_2d_mono(w)
        pad = maxL - w.size(-1)
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))
        padded.append(w)
    x = torch.cat(padded, dim=0)
    return x, lengths, list(bits), list(ops)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--voxceleb_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--process_cfg", type=str, default="./config/process.yaml")
    ap.add_argument("--model_cfg", type=str, default="./config/model.yaml")
    ap.add_argument("--train_cfg", type=str, default="./config/train.yaml")
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to your encoder checkpoint .pth.tar",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--count",
        type=int,
        default=SAMPLE_COUNT,
        help="number of FILES to process; set -1 for ALL eligible",
    )
    ap.add_argument(
        "--selection",
        type=str,
        default="sequential",
        choices=["sequential", "random"],
        help="How to pick --count files from eligible pool",
    )
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    build_dataset(
        voxceleb_root=args.voxceleb_root,
        out_root=args.out_root,
        process_cfg=args.process_cfg,
        model_cfg=args.model_cfg,
        train_cfg=args.train_cfg,
        checkpoint_path=args.ckpt,
        sample_count=args.count,
        seed=args.seed,
        selection=args.selection,
        max_workers=args.workers,
    )
