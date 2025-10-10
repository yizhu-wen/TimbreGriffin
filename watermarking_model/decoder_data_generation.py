# voxceleb_decoder_dataset.py
# Build a reproducible watermark-decoder dataset from VoxCeleb.
# Your requested changes:
# - Mirror the VoxCeleb layout under out_root: idXXXXX/videoid/
# - For each video folder, pick exactly ONE eligible .wav, embed watermark, write benign_identity.wav as the pure watermarked audio
# - Do NOT write original.wav or watermarked.wav
# - Apply remaining benign_* and malicious_* ops in the same folder
# - Store watermark bits in metadata.json and index them at the root
#
# Example output:
# /data/voxceleb_decoder_train/
# ├── id10001/
# │   └── 1zcIwhmdeo4/
# │       ├── benign_identity.wav
# │       ├── benign_compression.wav
# │       ├── benign_resample.wav
# │       ├── malicious_delete_0.3.wav
# │       ├── ...
# │       └── metadata.json
# ├── ...
# ├── dataset_index.jsonl
# └── meta.txt

import os
import json
import time
import random
import hashlib
from io import BytesIO
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample as tf_resample

import yaml
# Replace with your actual module path that defines Encoder
from model.conv2_mel_modules import Encoder, Decoder  # type: ignore

# Optional codec support (benign_compression, benign_reencode)
try:
    from pydub import AudioSegment
    _HAVE_PYDUB = True
except Exception:
    _HAVE_PYDUB = False

# -----------------------------
# Config
# -----------------------------
DEFAULT_SEED = 1337
SAMPLE_COUNT = 3800
ACCEPT_EXTS = {".wav", ".flac", ".mp3", ".m4a"}
USE_GPU = True
POST_RMS_MATCH = True  # applied to distorted ops, never to benign_identity

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

def save_wav(path: str, wav: torch.Tensor, sr: int):
    wav = ensure_2d_mono(wav)
    wav = clamp01(wav)
    ensure_dir(os.path.dirname(path))
    torchaudio.save(path, wav, sr, bits_per_sample=16)

def min_length_from_sr(sr: int) -> int:
    # int(2*sr + 0.5*sr + 0.5*sr) == 3*sr
    return 3 * sr

def speaker_id_from_path(path: str) -> Optional[str]:
    parts = os.path.normpath(path).split(os.sep)
    # VoxCeleb paths contain .../idXXXXX/VIDEOID/XXXX.wav
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

def load_encoder_ctx(process_cfg_path: str, model_cfg_path: str, train_cfg_path: str, checkpoint_path: str) -> WMEncoderCtx:
    process_config = yaml.load(open(process_cfg_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_cfg_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_cfg_path, "r"), Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    msg_len = int(train_config["watermark"]["length"])

    enc = Encoder(process_config, model_config, train_config, msg_len).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()

    return WMEncoderCtx(device=device, encoder=enc, msg_len=msg_len)

def _bitstr_to_msg_tensor(bitstr: str, total_len: int, device: torch.device, rng: random.Random) -> torch.Tensor:
    if len(bitstr) > total_len:
        raise ValueError(f"Model watermark length {total_len} is less than provided bit length {len(bitstr)}")
    head = [1.0 if c == "1" else -1.0 for c in bitstr]
    tail_bits = [rng.randint(0, 1) for _ in range(total_len - len(bitstr))]
    tail = [1.0 if b == 1 else -1.0 for b in tail_bits]
    vec = np.array(head + tail, dtype=np.float32).reshape(1, 1, total_len)
    return torch.from_numpy(vec).to(device)

def make_encoder_fn(ctx: WMEncoderCtx) -> Callable[[torch.Tensor, int, str, random.Random], torch.Tensor]:
    def _fn(wav_cpu: torch.Tensor, sr: int, bitstr: str, rng: random.Random) -> torch.Tensor:
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
# AudioProcessor ops
# -----------------------------
class AudioProcessor:
    @staticmethod
    def benign_identity(waveform, sample_rate=None, rng=None):
        # Pure watermarked baseline, no RMS match, no pad or trim
        return ensure_2d_mono(waveform)

    @staticmethod
    def benign_compression(waveform, sample_rate, codec="mp3", bitrate="128k", rng=None):
        if not _HAVE_PYDUB:
            raise RuntimeError("pydub or ffmpeg not available for benign_compression")
        x = ensure_2d_mono(waveform)
        wf = _to_numpy(x)
        buffer = BytesIO()
        wf_i16 = (wf * 32767).clip(-32768, 32767).astype(np.int16)
        AudioSegment(wf_i16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1).export(
            buffer, format=codec, bitrate=bitrate
        )
        buffer.seek(0)
        out = np.array(AudioSegment.from_file(buffer).get_array_of_samples()).astype(np.float32) / 32768.0
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
    def benign_reencode(waveform, sample_rate, passes=1, rng=None):
        if not _HAVE_PYDUB:
            raise RuntimeError("pydub or ffmpeg not available for benign_reencode")
        x = ensure_2d_mono(waveform)
        wf = _to_numpy(x)
        for _ in range(passes):
            buffer = BytesIO()
            wf_i16 = (wf * 32767).clip(-32768, 32767).astype(np.int16)
            AudioSegment(wf_i16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1).export(buffer, format="wav")
            buffer.seek(0)
            wf = np.array(AudioSegment.from_file(buffer, format="wav").get_array_of_samples()).astype(np.float32) / 32768.0
        T_in = x.size(-1)
        if wf.shape[0] > T_in:
            wf = wf[:T_in]
        elif wf.shape[0] < T_in:
            pad = np.zeros((T_in - wf.shape[0],), dtype=np.float32)
            wf = np.concatenate([wf, pad], axis=0)
        return _to_tensor(wf)

    @staticmethod
    def benign_noise_suppression(waveform, sr, energy_threshold=0.01, frame_size=None, hop_size=None, rng=None):
        x = ensure_2d_mono(waveform)
        fs, hs = _frame_params(sr) if frame_size is None or hop_size is None else (frame_size, hop_size)
        x0 = x[0]
        T = x0.size(0)
        for start in range(0, max(T - fs + 1, 1), hs):
            frame = x0[start:start + fs]
            if frame.numel() == 0:
                continue
            energy = torch.sqrt((frame ** 2).mean())
            if energy < energy_threshold:
                x0[start:start + fs] = 0.0
        return x

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
        out = torch.cat([x[..., :start], x[..., start + del_len:]], dim=-1)
        return out, {"ratio": ratio, "start": start, "del_len": del_len}

    @staticmethod
    def malicious_silence(waveform, sample_rate=16000, ratio=0.2, frame_size=None, hop_size=None, energy_threshold=0.01, rng=None):
        x = ensure_2d_mono(waveform)
        fs, hs = _frame_params(sample_rate) if frame_size is None or hop_size is None else (frame_size, hop_size)
        x0 = x[0]
        T = x0.size(0)
        frames = x0.unfold(0, min(fs, max(T, 1)), max(hs, 1)) if T >= fs else x0.view(1, -1)
        energy = torch.sqrt((frames ** 2).mean(dim=1))
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
    def malicious_reorder(waveform, sample_rate=None, num_segments=None, rng=None, fade_ms=5):
        x = ensure_2d_mono(waveform)
        T = x.shape[-1]
        if num_segments is None:
            num_segments = rng.choice([4, 6, 8])
        if num_segments > max(2, T):
            num_segments = max(2, min(T, 4))
        cut_points = sorted(rng.sample(range(1, T), num_segments - 1)) if T >= num_segments else list(range(1, num_segments))
        seg_bounds = [0] + cut_points + [T]
        segs = [x[:, seg_bounds[i]:seg_bounds[i + 1]] for i in range(len(seg_bounds) - 1)]
        rng.shuffle(segs)
        fade_len = int((fade_ms / 1000.0) * (sample_rate if sample_rate else 16000))
        out = segs[0]
        for s in segs[1:]:
            out = _crossfade(out, s, fade_len)
        return out, {"segments": num_segments, "fade_ms": fade_ms}

    @staticmethod
    def malicious_splice(waveform, sample_rate, spliced_waveform=None, rng=None, fade_ms=5):
        x = ensure_2d_mono(waveform)
        donor = ensure_2d_mono(spliced_waveform) if spliced_waveform is not None else None
        if donor is None or donor.numel() == 0:
            raise RuntimeError("spliced_waveform unavailable")
        start = rng.randint(0, x.size(-1))
        left = x[..., :start]
        right = x[..., start:]
        fade_len = int((fade_ms / 1000.0) * sample_rate)
        mid = _crossfade(left, donor, fade_len)
        out = torch.cat([mid, right], dim=-1)
        return out, {"insert_at": start, "donor_len": donor.size(-1), "fade_ms": fade_ms}

    @staticmethod
    def malicious_substitute(waveform, sample_rate, replace_waveform=None, frame_size=None, hop_size=None, energy_threshold=0.01, rng=None, fade_ms=5):
        x = ensure_2d_mono(waveform)
        donor = ensure_2d_mono(replace_waveform) if replace_waveform is not None else None
        if donor is None or donor.numel() == 0:
            raise RuntimeError("replace_waveform unavailable")
        fs, hs = _frame_params(sample_rate) if frame_size is None or hop_size is None else (frame_size, hop_size)
        x0 = x[0]
        T = x0.size(0)
        sub_len = donor.size(-1)
        frames = x0.unfold(0, min(fs, max(T, 1)), max(hs, 1)) if T >= fs else x0.view(1, -1)
        energy = torch.sqrt((frames ** 2).mean(dim=1))
        voiced = (energy > energy_threshold).nonzero(as_tuple=True)[0]
        if len(voiced) == 0:
            start = max(0, (T - sub_len) // 2)
        else:
            valid_starts = (voiced * hs)
            valid_starts = valid_starts[valid_starts <= max(0, T - sub_len)]
            start = int(valid_starts[rng.randint(0, len(valid_starts) - 1)].item()) if len(valid_starts) > 0 else max(0, (T - sub_len) // 2)
        left = x[..., :start]
        right = x[..., start + sub_len:]
        fade_len = int((fade_ms / 1000.0) * sample_rate)
        mid = _crossfade(left, donor, fade_len)
        out = torch.cat([mid, right], dim=-1)
        return out, {"start": start, "sub_len": sub_len, "fade_ms": fade_ms}

# Registry
DISTORTION_REGISTRY = {
    "benign_identity":      lambda wav, sr, rng, **kw: (AudioProcessor.benign_identity(wav, sr, rng=rng), {"note": "pure watermarked baseline"}),
    "benign_compression":   lambda wav, sr, rng, **kw: (AudioProcessor.benign_compression(wav, sr, rng=rng), {"codec": "mp3", "bitrate": "128k"}),
    "benign_resample":      lambda wav, sr, rng, **kw: (AudioProcessor.benign_resample(wav, sr, rng=rng), {"target_sr": 8000}),
    "benign_reencode":      lambda wav, sr, rng, **kw: (AudioProcessor.benign_reencode(wav, sr, passes=3, rng=rng), {"passes": 3}),
    "benign_noise_suppression": lambda wav, sr, rng, **kw: (AudioProcessor.benign_noise_suppression(wav, sr, rng=rng), {"energy_threshold": 0.01}),
    "malicious_delete_0.3": lambda wav, sr, rng, **kw: AudioProcessor.malicious_delete(wav, sr, ratio=0.3, rng=rng),
    "malicious_silence_0.2":lambda wav, sr, rng, **kw: AudioProcessor.malicious_silence(wav, sr, ratio=0.2, rng=rng),
    "malicious_reorder":    lambda wav, sr, rng, **kw: AudioProcessor.malicious_reorder(wav, sr, rng=rng),
    "malicious_splice":     lambda wav, sr, rng, donor=None, **kw: AudioProcessor.malicious_splice(wav, sr, spliced_waveform=donor, rng=rng),
    "malicious_substitute": lambda wav, sr, rng, donor=None, **kw: AudioProcessor.malicious_substitute(wav, sr, replace_waveform=donor, rng=rng),
}

# -----------------------------
# Donor selection from same speaker
# -----------------------------
def load_random_segment_from_same_speaker(cur_path: str, target_len: int, rng: random.Random, target_sr: int, speaker2files: Dict[str, List[str]]) -> Optional[torch.Tensor]:
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
    return ensure_2d_mono(wav_d[:, start:start + target_len])

# -----------------------------
# Sampling: pick exactly one eligible .wav per video folder
# -----------------------------
def find_audio_files(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in ACCEPT_EXTS:
                out.append(os.path.join(dp, fn))
    return out

def filter_by_min_length(path: str) -> bool:
    try:
        info = torchaudio.info(path)
        sr = info.sample_rate
        num = info.num_frames
        return num > min_length_from_sr(sr)
    except Exception:
        return False

def choose_one_per_parent(paths: List[str], seed: int) -> List[str]:
    # Group by parent dir (video folder), choose one eligible file per group deterministically
    groups: Dict[str, List[str]] = {}
    for p in paths:
        parent = os.path.dirname(p)
        groups.setdefault(parent, []).append(p)

    chosen = []
    for parent, files in groups.items():
        eligible = [p for p in files if filter_by_min_length(p)]
        if not eligible:
            continue
        rng = random.Random(int(hashlib.sha256((parent + str(seed)).encode()).hexdigest()[:16], 16))
        eligible.sort()  # stable
        chosen.append(rng.choice(eligible))
    return chosen  # one file per video folder

def sample_limit(files_one_per_parent: List[str], seed: int, k: int) -> List[str]:
    if len(files_one_per_parent) <= k:
        return files_one_per_parent
    return deterministic_shuffle(files_one_per_parent, seed)[:k]

# -----------------------------
# Per process context
# -----------------------------
_process_ctx = {"speaker2files": None, "seed": None, "voxceleb_root": None, "out_root": None}

def _worker_init(speaker2files: Dict[str, List[str]], seed: int, voxceleb_root: str, out_root: str):
    _process_ctx["speaker2files"] = speaker2files
    _process_ctx["seed"] = seed
    _process_ctx["voxceleb_root"] = voxceleb_root
    _process_ctx["out_root"] = out_root

def _make_sample_rng(global_seed: int, sample_key: str) -> random.Random:
    h = hashlib.sha256(f"{global_seed}|{sample_key}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))

# -----------------------------
# Distortion worker for a single selected file
# -----------------------------
@dataclass
class SampleResult:
    ok: bool
    sample_id: str
    dir_rel: str
    meta_rel: str
    error: Optional[str] = None

def _distort_and_write_metadata(in_path: str, rel_parent: str, bitstr: str, sr: int) -> SampleResult:
    try:
        out_root = _process_ctx["out_root"]
        rng = _make_sample_rng(_process_ctx["seed"], rel_parent)
        sdir = os.path.join(out_root, rel_parent)  # idXXXXX/VIDEOID
        base_path = os.path.join(sdir, "benign_identity.wav")

        wm_wav, sr_chk = load_audio(base_path)
        wm_wav = ensure_2d_mono(wm_wav)
        if sr_chk != sr:
            wm_wav = ensure_2d_mono(tf_resample(wm_wav, sr_chk, sr))

        operations_meta = []
        for name, fn in DISTORTION_REGISTRY.items():
            entry = {"name": name, "distorted_path": None, "error": None, "validation": None}
            try:
                # Identity already written
                if name == "benign_identity":
                    entry["distorted_path"] = "benign_identity.wav"
                    entry["validation"] = f"ok, pure watermarked baseline, in={shape_str(wm_wav)}, sr={sr}"
                    operations_meta.append(entry)
                    continue

                donor = None
                params = {}
                if name in {"malicious_splice", "malicious_substitute"}:
                    target_len = int(0.3 * wm_wav.size(-1)) if name == "malicious_splice" else int(0.2 * wm_wav.size(-1))
                    target_len = max(1, min(wm_wav.size(-1) - 1, target_len))
                    donor = load_random_segment_from_same_speaker(in_path, target_len, rng, sr, _process_ctx["speaker2files"])
                    if donor is None:
                        raise RuntimeError("donor segment unavailable")
                    donor = ensure_2d_mono(donor)

                out_wav, params = fn(wm_wav, sr, rng, donor=donor)
                out_wav = ensure_2d_mono(out_wav)
                out_wav = clamp01(out_wav)
                if POST_RMS_MATCH:
                    out_wav = _rms_match(out_wav, wm_wav)

                out_fname = f"{name}.wav"
                save_wav(os.path.join(sdir, out_fname), out_wav, sr)

                entry["distorted_path"] = out_fname
                entry["validation"] = f"ok, in={shape_str(wm_wav)}, out={shape_str(out_wav)}, sr={sr}, params={params}"
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

        sample_id = rel_parent.replace(os.sep, "/")
        return SampleResult(True, sample_id, rel_parent + "/", os.path.join(rel_parent, "metadata.json"))
    except Exception as e:
        return SampleResult(False, rel_parent.replace(os.sep, "/"), "", "", error=str(e))

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
    max_workers: Optional[int] = None,
) -> None:
    set_global_seed(seed)
    ensure_dir(out_root)

    enc_ctx = load_encoder_ctx(process_cfg, model_cfg, train_cfg, checkpoint_path)
    if enc_ctx.msg_len < 10:
        raise ValueError(f"Your model watermark length is {enc_ctx.msg_len}, need at least 10")
    encode_fn = make_encoder_fn(enc_ctx)

    # Discover all audio, pick one eligible per video folder, then limit to sample_count
    all_files = find_audio_files(voxceleb_root)
    one_per_parent = choose_one_per_parent(all_files, seed)
    chosen = sample_limit(one_per_parent, seed, sample_count)

    # Build a speaker->files map for donor selection
    speaker2files: Dict[str, List[str]] = {}
    for p in all_files:
        spk = speaker_id_from_path(p)
        if spk:
            speaker2files.setdefault(spk, []).append(p)

    # Bits assignment
    def make_bit_pool(s: int) -> List[str]:
        pool = [format(i, "010b") for i in range(1024)]
        return deterministic_shuffle(pool, s)
    pool = make_bit_pool(seed)
    bits_list = [pool[i % len(pool)] for i in range(len(chosen))]

    # Provenance
    meta_txt = os.path.join(out_root, "meta.txt")
    rel_parents = [os.path.relpath(os.path.dirname(p), voxceleb_root).replace(os.sep, "/") for p in chosen]
    meta_lines = [
        f"seed={seed}",
        f"voxceleb_root={voxceleb_root}",
        f"out_root={out_root}",
        f"sample_count={len(chosen)}",
        f"build_time_unix={int(time.time())}",
        f"encoder_device={enc_ctx.device.type}",
        f"msg_len={enc_ctx.msg_len}",
        "selected_parents=" + ",".join(rel_parents),
    ]
    atomic_write_bytes(("\n".join(meta_lines) + "\n").encode("utf-8"), meta_txt)

    # Stage A, embed watermark and write benign_identity.wav inside mirrored folder
    prepared: List[Tuple[str, str, str, int]] = []  # (in_path, rel_parent, bitstr, sr)
    for in_path, bitstr in zip(chosen, bits_list):
        try:
            wav, sr = load_audio(in_path)
            need = min_length_from_sr(sr)
            if wav.size(-1) <= need:
                continue
            rel_parent = os.path.relpath(os.path.dirname(in_path), voxceleb_root)  # idXXXXX/VIDEOID
            sdir = os.path.join(out_root, rel_parent)
            ensure_dir(sdir)

            # per folder RNG to pad message tail deterministically
            rng = _make_sample_rng(seed, rel_parent)
            wm_wav = encode_fn(wav, sr, bitstr, rng)  # pure watermarked
            save_wav(os.path.join(sdir, "benign_identity.wav"), wm_wav, sr)

            prepared.append((in_path, rel_parent, bitstr, sr))
        except Exception as e:
            with open(os.path.join(out_root, "errors_embed.txt"), "a", encoding="utf-8") as ef:
                ef.write(f"{os.path.relpath(in_path, voxceleb_root)}\t{str(e)}\n")

    # Stage B, distortions and metadata, in parallel
    index_entries = []
    errs = []
    max_workers = (max(1, os.cpu_count() or 1)
                   if max_workers in (None, 0, -1)
                   else max_workers)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(speaker2files, seed, voxceleb_root, out_root),
    ) as ex:
        futs = []
        for in_path, rel_parent, bitstr, sr in prepared:
            futs.append(ex.submit(_distort_and_write_metadata, in_path, rel_parent, bitstr, sr))
        for fu in as_completed(futs):
            res: SampleResult = fu.result()
            if res.ok:
                index_entries.append({"id": res.sample_id, "dirpath": res.dir_rel, "metadata_path": res.meta_rel})
            else:
                errs.append((res.sample_id, res.error))

    # Root index
    idx_path = os.path.join(out_root, "dataset_index.jsonl")
    index_entries = sorted(index_entries, key=lambda x: x["id"])
    with open(idx_path, "w", encoding="utf-8") as f:
        for e in index_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Optional error report
    if errs:
        with open(os.path.join(out_root, "errors.txt"), "w", encoding="utf-8") as f:
            for sid, err in errs:
                f.write(f"{sid}\t{err}\n")

    print(f"done, ok={len(index_entries)}, errors={len(errs)}")
    print(f"index at {idx_path}")
    print(f"meta at  {meta_txt}")

# -----------------------------
# Loader
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
                self.items.append({
                    "wav_path": os.path.join(base_dir, "benign_identity.wav"),
                    "bits": meta["watermark_bits"],
                    "op": "clean",
                })
            for op in meta["operations"]:
                if op["distorted_path"] is not None and op["error"] is None:
                    if include_clean and op["name"] == "benign_identity":
                        continue
                    self.items.append({
                        "wav_path": os.path.join(base_dir, op["distorted_path"]),
                        "bits": meta["watermark_bits"],
                        "op": op["name"],
                    })

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
    ap.add_argument("--ckpt", type=str, required=True, help="Path to your encoder checkpoint .pth.tar")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--count", type=int, default=SAMPLE_COUNT, help="number of video folders to process")
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
        max_workers=args.workers,
    )
