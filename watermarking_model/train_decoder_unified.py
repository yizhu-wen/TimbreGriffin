# train_decoder_unified.py
# Unified training with balanced batches, safe input shapes, deterministic cropping,
# tqdm progress bars, per epoch accuracy, per operation accuracy, and Weights & Biases logging.

import hashlib
import itertools
import json
import math
import os
import random
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm

import wandb

# Replace with your actual module path
from model.conv2_mel_modules import Encoder, Decoder  # type: ignore


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


def flip_mask_for_key(n: int, key: str, flip_ratio: float = 0.5) -> np.ndarray:
    k = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(k)
    idxs = list(range(n))
    rng.shuffle(idxs)
    m = int(round(n * flip_ratio))
    mask = np.zeros(n, dtype=bool)
    mask[idxs[:m]] = True
    return mask


def apply_flip(vec: torch.Tensor, key: str) -> torch.Tensor:
    if vec.dim() == 1:
        mask = torch.from_numpy(flip_mask_for_key(vec.numel(), key)).to(vec.device)
        out = vec.clone()
        out[mask] = -out[mask]
        return out
    outs = []
    for i in range(vec.size(0)):
        m = torch.from_numpy(flip_mask_for_key(vec.size(1), f"{key}:{i}")).to(
            vec.device
        )
        x = vec[i].clone()
        x[m] = -x[m]
        outs.append(x)
    return torch.stack(outs, dim=0)


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
    with torch.set_grad_enabled(decoder.training):
        out = decoder(x_batch)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.dim() == 3 and out.size(1) == 1:
        out = out[:, 0, :]
    return out  # [B, L]


def bit_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ((pred >= 0).eq(target >= 0).sum().float() / target.numel()).item()


# ---------------------------
# Deterministic cropping
# ---------------------------
def crop_or_pad_fixed(
    w: torch.Tensor, max_samples: int, key: str, epoch: int, center: bool = False
) -> torch.Tensor:
    if max_samples <= 0:
        return w
    T = w.size(-1)
    if T == max_samples:
        return w
    if T > max_samples:
        if center:
            start = (T - max_samples) // 2
        else:
            h = hashlib.sha256(f"{key}|{epoch}".encode("utf-8")).hexdigest()
            rng = random.Random(int(h[:16], 16))
            start = rng.randint(0, T - max_samples)
        return w[:, start : start + max_samples]
    return torch.nn.functional.pad(w, (0, max_samples - T))


# ---------------------------
# Unified dataset
# ---------------------------
class UnifiedDecoderDataset(Dataset):
    def __init__(self, index_file: str):
        print("index_file", index_file)
        self.root = os.path.dirname(index_file)
        entries = [json.loads(l) for l in open(index_file, "r", encoding="utf-8")]
        items = []
        for e in entries:
            meta_path = os.path.join(self.root, e["metadata_path"])

            print("root", self.root)
            print("e", e["metadata_path"])

            base_dir = os.path.join(self.root, e["dirpath"])
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            bits = meta["watermark_bits"]
            for op in meta["operations"]:
                if op["distorted_path"] is None or op["error"] is not None:
                    continue
                path = os.path.join(base_dir, op["distorted_path"])
                try:
                    info = torchaudio.info(path)
                    if info.num_frames < int(
                        2 * info.sample_rate
                        + 0.5 * info.sample_rate
                        + 0.5 * info.sample_rate
                    ):
                        continue  # drop too-short item
                except Exception:
                    continue
                name = op["name"]
                is_benign = name.startswith("benign_")
                items.append(
                    {"wav_path": path, "bits": bits, "op": name, "is_benign": is_benign}
                )
        if len(items) == 0:
            raise RuntimeError("No audio items found in index")
        self.items = items
        self.benign_idxs = [i for i, it in enumerate(items) if it["is_benign"]]
        self.malicious_idxs = [i for i, it in enumerate(items) if not it["is_benign"]]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        rec = self.items[i]
        wav, sr = load_wav(rec["wav_path"])
        return wav, rec["bits"], rec["op"], rec["wav_path"], rec["is_benign"]


# ---------------------------
# Balanced batch sampler
# ---------------------------
class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        benign_idxs: List[int],
        malicious_idxs: List[int],
        batch_size: int,
        benign_ratio: float = 0.5,
    ):
        assert 0.0 < benign_ratio < 1.0
        self.benign_idxs = list(benign_idxs)
        self.malicious_idxs = list(malicious_idxs)
        if len(self.benign_idxs) == 0 or len(self.malicious_idxs) == 0:
            raise ValueError("Both benign and malicious pools must be non empty")
        self.batch_size = batch_size
        self.b_count = max(1, int(round(batch_size * benign_ratio)))
        self.m_count = batch_size - self.b_count
        if self.m_count < 1:
            self.m_count = 1
            self.b_count = batch_size - 1
        self.num_batches = math.ceil(
            max(
                len(self.benign_idxs) / self.b_count,
                len(self.malicious_idxs) / self.m_count,
            )
        )

    def __iter__(self):
        rng = random.Random()
        b = self.benign_idxs[:]
        m = self.malicious_idxs[:]
        rng.shuffle(b)
        rng.shuffle(m)
        b_cycle = itertools.cycle(b)
        m_cycle = itertools.cycle(m)
        for _ in range(self.num_batches):
            batch = [next(b_cycle) for _ in range(self.b_count)] + [
                next(m_cycle) for _ in range(self.m_count)
            ]
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# ---------------------------
# Train and validate
# ---------------------------
def train_one_epoch(
    decoder, optimizer, device, loader, msg_len, max_samples: int, epoch: int
) -> Dict[str, float]:
    loss_fn = nn.MSELoss()
    decoder.train()

    running_loss = 0.0
    running_n = 0

    pbar = tqdm(loader, total=len(loader), desc=f"train [epoch {epoch}]", leave=False)
    for waves, bits_list, ops, keys, is_benigns in pbar:
        waves_c = [
            crop_or_pad_fixed(w, max_samples, key=k, epoch=epoch, center=False)
            for w, k in zip(waves, keys)
        ]
        x, _ = to_batch_1T(waves_c)
        x = x.to(device)

        K = len(bits_list[0])
        tgt = torch.stack([bits_to_vec(b, device) for b in bits_list], dim=0)
        for i, is_b in enumerate(is_benigns):
            if not is_b:
                tgt[i] = apply_flip(tgt[i], key=f"{ops[i]}|{keys[i]}")

        optimizer.zero_grad()
        out = try_decode(decoder, x)
        K_eff = min(K, out.size(1))
        pred = out[:, :K_eff]
        gold = tgt[:, :K_eff]

        loss = loss_fn(pred, gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()

        bs = x.size(0)
        running_loss += float(loss.item()) * bs
        running_n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": running_loss / max(1, running_n)}


@torch.no_grad()
def evaluate(decoder, device, loader, msg_len, max_samples: int) -> Dict[str, object]:
    decoder.eval()
    benign_accs = []
    malicious_accs = []

    # per operation accumulators
    op_sum = defaultdict(float)
    op_count = defaultdict(int)

    pbar = tqdm(loader, total=len(loader), desc="eval", leave=False)
    for waves, bits_list, ops, keys, is_benigns in pbar:
        waves_c = [
            crop_or_pad_fixed(w, max_samples, key=k, epoch=0, center=True)
            for w, k in zip(waves, keys)
        ]
        x, _ = to_batch_1T(waves_c)
        x = x.to(device)

        K = len(bits_list[0])
        tgt = torch.stack([bits_to_vec(b, device) for b in bits_list], dim=0)

        out = try_decode(decoder, x)
        K_eff = min(K, out.size(1))

        ba, ma = [], []
        for i, is_b in enumerate(is_benigns):
            pred_i = out[i, :K_eff].unsqueeze(0)
            tgt_i = tgt[i, :K_eff].unsqueeze(0)
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

    b_acc = float(np.mean(benign_accs)) if benign_accs else 0.0
    m_acc = float(np.mean(malicious_accs)) if malicious_accs else 0.0

    per_op_acc = {op: (op_sum[op] / max(1, op_count[op])) for op in op_sum.keys()}
    per_op_count = dict(op_count)

    return {
        "benign_acc": b_acc,
        "malicious_acc": m_acc,
        "per_op_acc": per_op_acc,
        "per_op_count": per_op_count,
    }


# ---------------------------
# WandB helpers
# ---------------------------
def _sanitize_key(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(".", "_").replace(":", "_")


def log_epoch_to_wandb(
    ep: int,
    train_stats: Dict[str, float],
    eval_stats: Dict[str, object],
    optimizer: torch.optim.Optimizer,
):
    # Scalars
    lr = None
    for pg in optimizer.param_groups:
        if "lr" in pg:
            lr = pg["lr"]
            break

    log_dict = {
        "epoch": ep,
        "train/loss": float(train_stats["loss"]),
        "eval/benign_acc": float(eval_stats["benign_acc"]),
        "eval/malicious_acc": float(eval_stats["malicious_acc"]),
    }
    if lr is not None:
        log_dict["opt/lr"] = float(lr)

    # Per-op scalars
    for op, acc in eval_stats["per_op_acc"].items():
        log_dict[f"eval/per_op/{_sanitize_key(op)}"] = float(acc)

    wandb.log(log_dict, step=ep)

    # Per-op table (for nice UI)
    rows = []
    for op, acc in eval_stats["per_op_acc"].items():
        rows.append([op, float(acc), int(eval_stats["per_op_count"].get(op, 0))])
    if rows:
        table = wandb.Table(data=rows, columns=["operation", "accuracy", "count"])
        wandb.log({"eval/per_op_table": table}, step=ep)


# ---------------------------
# Main
# ---------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--index_file", type=str, required=True)
    ap.add_argument("--eval_index_file", type=str, required=True)
    ap.add_argument("--process_cfg", type=str, default="./config/process.yaml")
    ap.add_argument("--model_cfg", type=str, default="./config/model.yaml")
    ap.add_argument("--train_cfg", type=str, default="./config/train.yaml")
    ap.add_argument("--decoder_ckpt_in", type=str, default="")
    ap.add_argument("--decoder_ckpt_out", type=str, default="decoder_unified.pth.tar")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--benign_ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--max_samples", type=int, default=176000, help="cap per waveform, samples"
    )

    # New: wandb config
    ap.add_argument("--wandb_project", type=str, default="wm-decoder")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_run", type=str, default=None)
    ap.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    ap.add_argument("--wandb_tags", type=str, default="", help="comma-separated tags")

    args = ap.parse_args()

    set_seed(args.seed)

    process_config = yaml.load(open(args.process_cfg, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_cfg, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_cfg, "r"), Loader=yaml.FullLoader)
    msg_len = int(train_config["watermark"]["length"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # WandB init
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb.login(key="9a11e5364efe3bb8fedb3741188ee0d714e942e2")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run,
        mode=args.wandb_mode,  # "disabled" by default so it doesn't require login unless you opt in
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "benign_ratio": args.benign_ratio,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "msg_len": msg_len,
            "index_file": args.index_file,
            "eval_index_file": args.eval_index_file,
            "model_cfg": args.model_cfg,
            "process_cfg": args.process_cfg,
            "train_cfg": args.train_cfg,
        },
        tags=tags or None,
    )

    decoder = Decoder(process_config, model_config, train_config, msg_len).to(device)
    if args.decoder_ckpt_in and os.path.isfile(args.decoder_ckpt_in):
        ckpt = torch.load(args.decoder_ckpt_in, map_location=device)
        sd = ckpt["decoder"] if "decoder" in ckpt else ckpt
        decoder.load_state_dict(sd, strict=False)
        print(f"loaded decoder from {args.decoder_ckpt_in}")

    # Optional: gradients/params watching
    if args.wandb_mode != "disabled":
        wandb.watch(decoder, log="gradients", log_freq=100, log_graph=False)

    train_dataset = UnifiedDecoderDataset(args.index_file)
    test_dataset = UnifiedDecoderDataset(args.eval_index_file)
    train_sampler = BalancedBatchSampler(
        train_dataset.benign_idxs,
        train_dataset.malicious_idxs,
        args.batch_size,
        benign_ratio=args.benign_ratio,
    )
    test_sampler = BalancedBatchSampler(
        test_dataset.benign_idxs,
        test_dataset.malicious_idxs,
        args.batch_size,
        benign_ratio=args.benign_ratio,
    )

    def collate(batch):
        waves, bits, ops, keys, is_benigns = zip(*batch)
        return list(waves), list(bits), list(ops), list(keys), list(is_benigns)

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=4, collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset, batch_sampler=test_sampler, num_workers=4, collate_fn=collate
    )

    # Quick dataset stats to W&B
    wandb.summary["train_size"] = len(train_dataset)
    wandb.summary["test_size"] = len(test_dataset)
    wandb.summary["train_benign"] = len(train_dataset.benign_idxs)
    wandb.summary["train_malicious"] = len(train_dataset.malicious_idxs)
    wandb.summary["test_benign"] = len(test_dataset.benign_idxs)
    wandb.summary["test_malicious"] = len(test_dataset.malicious_idxs)

    optimizer = torch.optim.Adam(
        decoder.parameters(), betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0, lr=args.lr
    )

    for ep in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            decoder,
            optimizer,
            device,
            train_loader,
            msg_len,
            args.max_samples,
            epoch=ep,
        )
        eval_stats = evaluate(decoder, device, test_loader, msg_len, args.max_samples)

        # epoch summary to stdout
        print(
            f"[epoch {ep}] train_loss {train_stats['loss']:.4f}, "
            f"benign_acc {eval_stats['benign_acc']:.4f}, malicious_acc {eval_stats['malicious_acc']:.4f}"
        )

        # per operation overview to stdout
        ops_sorted = sorted(
            eval_stats["per_op_acc"].items(), key=lambda kv: kv[1], reverse=True
        )
        head = ", ".join(
            [
                f"{name}: {acc:.3f} [n={eval_stats['per_op_count'][name]}]"
                for name, acc in ops_sorted
            ]
        )
        print(f"[epoch {ep}] per_op_acc [{head}]")

        # log to W&B
        log_epoch_to_wandb(ep, train_stats, eval_stats, optimizer)

        # Optionally save checkpoints each epoch
        # torch.save({"decoder": decoder.state_dict()}, args.decoder_ckpt_out)

    print("done")
    wandb.finish()


if __name__ == "__main__":
    main()
