# eval_decoder_unified.py
# Pure inference: evaluate per-distortion accuracy (including benign_identity) on a built dataset.

import hashlib
import json
import os
import random
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torchaudio
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Replace with your actual module path
from model.conv2_mel_modules import Decoder  # type: ignore


# ---------------------------
# Repro and helpers
# ---------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_2d_mono(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.view(1, -1)
    if x.dim() == 2:
        return x.mean(dim=0, keepdim=True) if x.size(0) > 1 else x
    if x.dim() == 3:
        b, c, t = x.shape
        x = x.mean(dim=1, keepdim=True) if c > 1 else x
        return x[0]
    raise ValueError(f"Unexpected shape {tuple(x.shape)}")


def load_wav(path: str) -> Tuple[torch.Tensor, int]:
    w, sr = torchaudio.load(path)
    return ensure_2d_mono(w), sr


def bits_to_vec(bits: str, device: torch.device) -> torch.Tensor:
    arr = np.array([1.0 if c == "1" else -1.0 for c in bits], dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def to_batch_1T(x_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([w.size(-1) for w in x_list], dtype=torch.long)
    T = int(lengths.max().item())
    padded = []
    for w in x_list:
        pad = T - w.size(-1)
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))
        padded.append(w)
    x = torch.cat(padded, dim=0)  # [B, T]
    return x, lengths


def try_decode(decoder, x_batch: torch.Tensor) -> torch.Tensor:
    # Always feed [B, T] to match your decoder forward
    decoder.eval()
    with torch.no_grad():
        out = decoder(x_batch)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.dim() == 3 and out.size(1) == 1:
        out = out[:, 0, :]
    return out  # [B, L]


def bit_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ((pred >= 0).eq(target >= 0).sum().float() / target.numel()).item()


def crop_or_pad_fixed(
    w: torch.Tensor, max_samples: int, key: str, center: bool = True
) -> torch.Tensor:
    """Deterministic crop/pad using key; center crop by default for eval."""
    if max_samples <= 0:
        return w
    T = w.size(-1)
    if T == max_samples:
        return w
    if T > max_samples:
        if center:
            start = (T - max_samples) // 2
        else:
            h = hashlib.sha256(f"{key}".encode("utf-8")).hexdigest()
            rng = random.Random(int(h[:16], 16))
            start = rng.randint(0, T - max_samples)
        return w[:, start : start + max_samples]
    return torch.nn.functional.pad(w, (0, max_samples - T))


# ---------------------------
# Inference dataset
# ---------------------------
class InferenceDecoderDataset(Dataset):
    """
    Reads dataset_index.jsonl and yields (wav_path, bits, op, is_benign).
    Keeps benign_identity (identity) and all other valid distortions.
    Filters by a minimum length rule (default: > 3 * sr samples).
    """

    def __init__(self, index_file: str, minlen_rule: str = "3x"):
        self.root = os.path.dirname(index_file)

        def _min_length_from_sr(sr: int) -> int:
            if minlen_rule == "3x":
                return 3 * sr
            if minlen_rule.endswith("ms"):
                ms = float(minlen_rule[:-2])
                return int(sr * (ms / 1000.0))
            if minlen_rule.endswith("s"):
                sec = float(minlen_rule[:-1])
                return int(sr * sec)
            if minlen_rule.endswith("x"):
                mult = float(minlen_rule[:-1])
                return int(mult * sr)
            raise ValueError(f"Unknown minlen_rule: {minlen_rule}")

        with open(index_file, "r", encoding="utf-8") as f:
            entries = [json.loads(l) for l in f]

        items = []
        for e in entries:
            meta_path = os.path.join(self.root, e["metadata_path"])
            base_dir = os.path.join(self.root, e["dirpath"])
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            bits = meta["watermark_bits"]

            for op in meta["operations"]:
                if op["distorted_path"] is None or op["error"] is not None:
                    continue
                path = os.path.join(base_dir, op["distorted_path"])
                try:
                    info = torchaudio.info(path)
                    need = _min_length_from_sr(info.sample_rate)
                    if info.num_frames <= need:
                        continue
                except Exception:
                    continue

                name = op["name"]
                is_benign = name.startswith("benign_")
                items.append(
                    {"wav_path": path, "bits": bits, "op": name, "is_benign": is_benign}
                )

        if len(items) == 0:
            raise RuntimeError("No audio items found in index after filtering")

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        rec = self.items[i]
        return rec["wav_path"], rec["bits"], rec["op"], rec["is_benign"]


def collate(batch):
    # load audio here to parallelize with num_workers
    waves, bits, ops, keys, is_benigns = [], [], [], [], []
    for wav_path, bitstr, op, is_b in batch:
        w, sr = load_wav(wav_path)
        waves.append(w)
        bits.append(bitstr)
        ops.append(op)
        keys.append(wav_path)  # use path as deterministic key
        is_benigns.append(is_b)
    return waves, bits, ops, keys, is_benigns


# ---------------------------
# Evaluation
# ---------------------------
@torch.no_grad()
def evaluate(
    index_file: str,
    decoder_ckpt: str,
    process_cfg: str,
    model_cfg: str,
    train_cfg: str,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: int = 176000,
    minlen_rule: str = "3x",
    save_json: str = "",
) -> Dict[str, object]:
    # Load configs for message length
    process_config = yaml.load(open(process_cfg, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_cfg, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_cfg, "r"), Loader=yaml.FullLoader)
    msg_len = int(train_config["watermark"]["length"])

    # Model
    decoder = Decoder(process_config, model_config, train_config, msg_len).to(device)
    if decoder_ckpt and os.path.isfile(decoder_ckpt):
        ckpt = torch.load(decoder_ckpt, map_location=device)
        sd = ckpt["decoder"] if "decoder" in ckpt else ckpt
        decoder.load_state_dict(sd, strict=False)
        print(f"loaded decoder from {decoder_ckpt}")
    decoder.eval()

    # Dataset / Loader
    ds = InferenceDecoderDataset(index_file, minlen_rule=minlen_rule)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    # Accumulators
    benign_accs, malicious_accs = [], []
    op_sum: Dict[str, float] = defaultdict(float)
    op_count: Dict[str, int] = defaultdict(int)

    pbar = tqdm(dl, total=len(dl), desc="inference", leave=False)
    for waves, bits_list, ops, keys, is_benigns in pbar:
        # center crop/pad for eval
        waves_c = [
            crop_or_pad_fixed(w, max_samples, key=k, center=True)
            for w, k in zip(waves, keys)
        ]
        x, _ = to_batch_1T(waves_c)
        x = x.to(device)

        # targets
        K = len(bits_list[0])
        tgt = torch.stack([bits_to_vec(b, device) for b in bits_list], dim=0)

        # predict
        out = try_decode(decoder, x)
        K_eff = min(K, out.size(1))

        ba, ma = [], []
        for i, is_b in enumerate(is_benigns):
            pred_i = out[i, :K_eff].unsqueeze(0)
            tgt_i = tgt[i, :K_eff].unsqueeze(0)  # compare directly to ground truth
            acc_i = bit_accuracy(pred_i, tgt_i)
            if is_b:
                ba.append(acc_i)
            else:
                ma.append(acc_i)
            op_name = ops[i]
            op_sum[op_name] += acc_i
            op_count[op_name] += 1

        if ba:
            benign_accs.extend(ba)
        if ma:
            malicious_accs.extend(ma)

        cur_b = float(np.mean(ba)) if ba else 0.0
        cur_m = float(np.mean(ma)) if ma else 0.0
        pbar.set_postfix(benign=f"{cur_b:.3f}", malicious=f"{cur_m:.3f}")

    # Final metrics
    benign_acc = float(np.mean(benign_accs)) if benign_accs else 0.0
    malicious_acc = float(np.mean(malicious_accs)) if malicious_accs else 0.0
    overall_acc = (
        float(np.mean(benign_accs + malicious_accs))
        if (benign_accs or malicious_accs)
        else 0.0
    )
    per_op_acc = {
        op: (op_sum[op] / max(1, op_count[op])) for op in sorted(op_sum.keys())
    }
    per_op_count = dict(op_count)

    results = {
        "overall_acc": overall_acc,
        "benign_acc": benign_acc,
        "malicious_acc": malicious_acc,
        "per_op_acc": per_op_acc,
        "per_op_count": per_op_count,
        "num_items": len(ds),
        "batch_size": batch_size,
        "max_samples": max_samples,
        "minlen_rule": minlen_rule,
    }

    # Pretty print summary
    print("\n=== Inference Summary ===")
    print(
        f"items: {len(ds)} | overall_acc: {overall_acc:.4f} | benign: {benign_acc:.4f} | malicious: {malicious_acc:.4f}"
    )
    print("per-op accuracy (including benign_identity):")
    for name, acc in sorted(per_op_acc.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:28s} acc={acc:.4f}  n={per_op_count[name]}")

    if save_json:
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"saved results to {save_json}")

    return results


# ---------------------------
# CLI
# ---------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index_file",
        type=str,
        required=True,
        help="dataset_index.jsonl of the build to evaluate",
    )
    ap.add_argument("--process_cfg", type=str, default="./config/process.yaml")
    ap.add_argument("--model_cfg", type=str, default="./config/model.yaml")
    ap.add_argument("--train_cfg", type=str, default="./config/train.yaml")
    ap.add_argument(
        "--decoder_ckpt",
        type=str,
        required=True,
        help="decoder checkpoint to load (.pth.tar)",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--max_samples", type=int, default=176000, help="cap per waveform (samples)"
    )
    ap.add_argument(
        "--minlen_rule",
        type=str,
        default="3x",
        help='min length rule: e.g., "3x", "2s", "250ms"',
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    evaluate(
        index_file=args.index_file,
        decoder_ckpt=args.decoder_ckpt,
        process_cfg=args.process_cfg,
        model_cfg=args.model_cfg,
        train_cfg=args.train_cfg,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        minlen_rule=args.minlen_rule,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
