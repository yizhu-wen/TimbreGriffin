import os
import torch
import yaml
import logging
import argparse
import wandb
from io import BytesIO
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from rich.progress import track
from torch.utils.data import DataLoader
from model.loss import Loss_identity
from utils.tools import save, log, save_op
from utils.optimizer import ScheduledOptimMain, ScheduledOptimDisc, my_step
from itertools import chain
import librosa.display
from torch.nn.functional import mse_loss
from dataset.data import collate_fn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from model.conv2_mel_modules import (
    Encoder,
    Decoder,
    Discriminator,
    save_waveform,
    save_spectrum,
    save_spectrum_normal,
)
from dataset.data import WavDataset as MyDataset
import tempfile
import warnings
import random
import shutil
from PIL import Image
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def save_spectrogram_to_buffer(signal, sample_rate=16000):
    buf = BytesIO()
    plt.figure(figsize=(10, 4))
    plt.specgram(
        np.maximum(signal.cpu().detach().numpy(), 1e-10),
        Fs=sample_rate,
        NFFT=320,
        noverlap=160,
        window=np.hanning(320),
        cmap="magma",
        vmin=-100,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.0)
    plt.close()
    buf.seek(0)  # reset buffer pointer to beginning
    return buf


def buffer_to_wandb_image(buffer, caption=""):
    # Convert the buffer to a PIL Image, then to a numpy array.
    img = Image.open(buffer)
    np_img = np.array(img)
    return wandb.Image(np_img, caption=caption)


def normalize_audio(y: torch.Tensor) -> torch.Tensor:
    """Normalize an audio tensor so its maximum absolute value is 1."""
    peak = torch.max(torch.abs(y))
    if peak.item() > 1e-8:
        y = y / peak
    return y


# Set random seed for reproducibility
def set_random_seed(seed: int):
    random.seed(seed)  # For Python random module
    np.random.seed(seed)  # For NumPy
    torch.manual_seed(seed)  # For PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # For PyTorch (GPU)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables optimization for reproducibility


def generate_random_msg(batch_size, msg_length, device):
    # random [0, 1], mapped to [-1, 1]
    return (
        torch.randint(0, 2, (batch_size, 1, msg_length), device=device).float() * 2
    ) - 1


def cpu_metrics_worker(y_np: np.ndarray, x_np: np.ndarray, sr: int):
    """
    Runs SI-SNR, STOI, PESQ on CPU for a single utterance [T] each.
    Returns floats (sisnr, stoi, pesq). Uses torchmetrics on CPU.

    Notes:
    - STOI needs pystoi installed.
    - PESQ can be slow and may fail on some utterances, we catch exceptions and return NaN.
    """
    import numpy as np
    import torch

    from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
    from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

    y = torch.from_numpy(y_np).float().unsqueeze(0)  # [1, T]
    x = torch.from_numpy(x_np).float().unsqueeze(0)  # [1, T]

    with torch.no_grad():
        sisnr = ScaleInvariantSignalNoiseRatio()
        sisnr_v = float(sisnr(y, x).item())

        # STOI
        try:
            stoi = ShortTimeObjectiveIntelligibility(fs=sr, extended=True)
            stoi_v = float(stoi(y, x).item())
        except Exception:
            stoi_v = float("nan")

        # PESQ
        try:
            # "wb" requires 16000 Hz, "nb" requires 8000 Hz in classic PESQ constraints
            pesq = PerceptualEvaluationSpeechQuality(sr, "wb")
            pesq_v = float(pesq(y, x).item())
        except Exception:
            pesq_v = float("nan")

    return sisnr_v, stoi_v, pesq_v


# Set random seed for reproducibility
SEED = 2022
set_random_seed(SEED)

logging_mark = "#" * 20
# warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(configs):
    logging.info("main function")
    process_config, model_config, train_config = configs

    pre_step = 0
    # ---------------- get train dataset
    train_audios = MyDataset(
        process_config=process_config, train_config=train_config, flag="train"
    )
    val_audios = MyDataset(
        process_config=process_config, train_config=train_config, flag="val"
    )
    dev_audios = MyDataset(
        process_config=process_config, train_config=train_config, flag="test"
    )

    batch_size = train_config["optimize"]["batch_size"]
    assert batch_size <= len(train_audios)
    train_audios_loader = DataLoader(
        train_audios,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,  # Enable pin memory for faster data transfer to GPU
        num_workers=20,  # Enable multiple workers for data loading
        persistent_workers=True,  # Keep workers alive between epochs
    )
    val_audios_loader = DataLoader(
        val_audios,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=20,
        persistent_workers=True,
    )
    dev_audios_loader = DataLoader(
        dev_audios,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=20,
        persistent_workers=True,
    )

    # Initialize wandb only if enabled in config
    if train_config["wandb"]["enabled"]:
        wandb.login(key=train_config["wandb"]["key"])
        wandb.init(
            project=train_config["wandb"]["project"],
            name=train_config["wandb"]["name"],
            config={
                "dataset": "LibriSpeech",
                "data_size": "whole size divided by {}".format(
                    train_config["iter"]["data_divider"]
                ),
                "batch_size": train_config["optimize"]["batch_size"],
                "learning_rate": train_config["optimize"]["lr"],
                "delay_amt_second": train_config["watermark"]["delay_amt_second"],
                "future_amt_second": train_config["watermark"]["future_amt_second"],
                "nbits": train_config["watermark"]["length"],
                "epochs": train_config["iter"]["epoch"],
                "save_circle": train_config["iter"]["save_circle"],
                "show_circle": train_config["iter"]["show_circle"],
                "val_circle": train_config["iter"]["val_circle"],
            },
        )

    # ---------------- build model
    msg_length = train_config["watermark"]["length"]

    encoder = Encoder(process_config, model_config, train_config, msg_length).to(device)
    decoder = Decoder(process_config, model_config, train_config, msg_length).to(device)
    # adv
    if train_config["adv"]:
        discriminator = Discriminator(process_config).to(device)
        d_op = Adam(
            params=discriminator.parameters(),  # No need for chain() with single model
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr=train_config["optimize"]["lr"],
        )
        lr_sched_d = StepLR(
            d_op,
            step_size=train_config["optimize"]["step_size"],
            gamma=train_config["optimize"]["gamma"],
        )
    # shared parameters
    if model_config["structure"]["share"]:
        decoder.wav_encoder = encoder.wav_encoder
    # ---------------- optimizer
    en_de_op = Adam(
        params=chain(decoder.parameters(), encoder.parameters()),
        betas=train_config["optimize"]["betas"],
        eps=train_config["optimize"]["eps"],
        weight_decay=train_config["optimize"]["weight_decay"],
        lr=train_config["optimize"]["lr"],
    )
    lr_sched = StepLR(
        en_de_op,
        step_size=train_config["optimize"]["step_size"],
        gamma=train_config["optimize"]["gamma"],
    )

    # ---------------- Loss
    loss = Loss_identity()

    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    # ---------------- train
    logging.info(logging_mark + "\t" + "Begin Training" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    lambda_b = train_config["optimize"]["lambda_b"]
    sample_rate = process_config["audio"]["or_sample_rate"]
    hop_length = process_config["mel"]["hop_length"]
    global_step = 0
    train_len = len(train_audios_loader)

    for ep in range(1, epoch_num + 1):
        encoder.train()
        decoder.train()
        if train_config["adv"]:
            discriminator.train()
        step = 0
        logging.info("Epoch {}/{}".format(ep, epoch_num))
        train_avg_acc = [0, 0]
        train_avg_snr = 0
        train_avg_wav_loss = 0
        train_avg_msg_loss = 0
        train_avg_loudness_loss = 0
        train_avg_d_loss_on_encoded = 0
        train_avg_d_loss_on_cover = 0
        for sample in track(train_audios_loader):
            # inside the loop, e.g. every 500 steps
            global_step += 1
            step += 1
            b = sample["matrix"].shape[0]
            # ---------------- build watermark
            wav_matrix = sample["matrix"].to(device)  # [B, T]
            msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
            watermark, _ = encoder(wav_matrix, msg, global_step)
            y_wm = wav_matrix + watermark
            decoded = decoder(y_wm, global_step)
            losses = loss.en_de_loss(
                wav_matrix,
                y_wm,
                msg,
                decoded,
            )
            # lamda_e = 1.
            # lamda_m = 10
            # lamda_b = 1
            if global_step < pre_step:
                sum_loss = lambda_m * losses[1]
            else:
                sum_loss = (
                    lambda_e * losses[0] + lambda_m * losses[1] + lambda_b * losses[2]
                )

            # adv
            if train_config["adv"]:
                lambda_a = train_config["optimize"][
                    "lambda_a"
                ]

                g_target_label_encoded = torch.ones((b, 1), device=device)
                d_on_encoded_for_enc = discriminator(y_wm)
                # target label for encoded images should be 'cover', because we want to fool the discriminator
                g_loss_adv = F.binary_cross_entropy_with_logits(
                    d_on_encoded_for_enc, g_target_label_encoded
                )
                sum_loss += lambda_a * g_loss_adv

            # ----- Generator backward + step
            en_de_op.zero_grad(set_to_none=True)
            sum_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                chain(encoder.parameters(), decoder.parameters()), max_norm=1.0
            )

            my_step(en_de_op, lr_sched, global_step, train_len)

            # ----- Discriminator step
            d_loss_on_cover = None
            d_loss_on_encoded = None

            if train_config["adv"]:
                d_op.zero_grad(set_to_none=True)

                d_target_label_cover = torch.ones((b, 1), device=device)
                d_target_label_encoded = torch.zeros((b, 1), device=device)

                d_on_cover = discriminator(wav_matrix)
                d_on_encoded = discriminator(y_wm.detach())

                d_loss_on_cover = F.binary_cross_entropy_with_logits(
                    d_on_cover, d_target_label_cover
                )

                # target label for encoded images should be 'encoded', because we want discriminator fight with encoder
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(
                    d_on_encoded, d_target_label_encoded
                )

                d_loss = d_loss_on_cover + d_loss_on_encoded

                d_loss.backward()

                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                my_step(d_op, lr_sched_d, global_step, train_len)

            decoder_acc = [
                ((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
            ]
            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            snr = 10 * torch.log10(
                mse_loss(wav_matrix.detach(), zero_tensor)
                / mse_loss(wav_matrix.detach(), y_wm.detach())
            )
            norm2 = mse_loss(wav_matrix.detach(), zero_tensor)

            # Update averages
            train_avg_acc[0] += decoder_acc[0]
            train_avg_acc[1] += decoder_acc[1]
            train_avg_snr += snr.item()
            train_avg_wav_loss += losses[0].item()
            train_avg_msg_loss += losses[1].item()
            train_avg_loudness_loss += losses[2].item()
            if train_config["adv"]:
                train_avg_d_loss_on_cover += d_loss_on_cover.item()
                train_avg_d_loss_on_encoded += d_loss_on_encoded.item()

            if step % show_circle == 0:
                logging.info("-" * 100)
                logging.info(
                    "step:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} - acc:[{:.8f},{:.8f}] - snr:{:.8f} - "
                    "norm:{:.8f} - patch_num:{} - pad_num:{} - wav_len:{} ".format(
                        step,
                        losses[0],
                        losses[1],
                        losses[2],
                        decoder_acc[0],
                        decoder_acc[1],
                        snr,
                        norm2,
                        sample["patch_num"].tolist(),
                        sample["pad_num"].tolist(),
                        wav_matrix.shape[-1],
                    )
                )

        train_avg_acc[0] /= step
        train_avg_acc[1] /= step
        train_avg_snr /= step
        train_avg_wav_loss /= step
        train_avg_msg_loss /= step
        train_avg_loudness_loss /= step
        train_avg_d_loss_on_encoded /= step
        train_avg_d_loss_on_cover /= step

        train_metrics = {
            "train/wav_loss": train_avg_wav_loss,
            "train/msg_loss": train_avg_msg_loss,
            "train/loudness_loss": train_avg_loudness_loss,
            "train/acc_1": train_avg_acc[0],
            "train/acc_2": train_avg_acc[1],
            "train/snr": train_avg_snr,
            "train/d_loss_on_encoded": train_avg_d_loss_on_encoded,
            "train/d_loss_on_cover": train_avg_d_loss_on_cover,
        }

        # if ep % save_circle == 0 or ep == 1 or ep == 2:
        if ep % save_circle == 0:
            if not model_config["structure"]["ab"]:
                path = os.path.join(train_config["path"]["ckpt"], "pth")
            else:
                path = os.path.join(train_config["path"]["ckpt"], "pth_ab")
            save_op(path, ep, encoder, decoder, en_de_op)
            # shutil.copyfile(os.path.realpath(__file__), os.path.join(path, os.path.basename(os.path.realpath(__file__)))) # save training scripts

        # ---------------- validation
        with torch.inference_mode():
            encoder.eval()
            decoder.eval()
            if train_config["adv"]:
                discriminator.eval()
            val_avg_acc = [0.0, 0.0]
            val_avg_snr = 0.0
            val_avg_wav_loss = 0.0
            val_avg_msg_loss = 0.0
            val_avg_loudness_loss = 0.0
            val_avg_d_loss_on_encoded = 0.0
            val_avg_d_loss_on_cover = 0.0
            val_avg_d_acc = 0.0
            count = 0
            for sample in track(val_audios_loader):
                count += 1
                wav_matrix = sample["matrix"].to(device)
                b = sample["matrix"].shape[0]

                # ---------------- build watermark
                msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
                watermark, zeros_right = encoder(wav_matrix, msg, global_step)
                y_wm = wav_matrix + watermark
                decoded = decoder(y_wm, global_step)
                losses = loss.en_de_loss(
                    wav_matrix,
                    y_wm,
                    msg,
                    decoded,
                )
                # adv
                if train_config["adv"]:
                    g_target_label_encoded = torch.ones((b, 1), device=device)
                    d_target_label_encoded = torch.zeros((b, 1), device=device)
                    logits_cover = discriminator(wav_matrix)
                    logits_encoded = discriminator(y_wm)

                    d_loss_on_cover = F.binary_cross_entropy_with_logits(
                        logits_cover, g_target_label_encoded
                    )
                    d_loss_on_encoded = F.binary_cross_entropy_with_logits(
                        logits_encoded, d_target_label_encoded
                    )

                    val_avg_d_loss_on_cover += d_loss_on_cover.item()
                    val_avg_d_loss_on_encoded += d_loss_on_encoded.item()

                    # optional discriminator accuracy
                    acc_cover = (torch.sigmoid(logits_cover) > 0.5).float().mean()
                    acc_encoded = (torch.sigmoid(logits_encoded) < 0.5).float().mean()
                    val_avg_d_acc += (0.5 * (acc_cover + acc_encoded)).item()

                decoder_acc = [
                    ((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                    ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                ]
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)
                snr = 10 * torch.log10(
                    mse_loss(wav_matrix.detach(), zero_tensor)
                    / mse_loss(wav_matrix.detach(), y_wm.detach())
                )
                val_avg_acc[0] += decoder_acc[0]
                val_avg_acc[1] += decoder_acc[1]
                val_avg_snr += snr.item()
                val_avg_wav_loss += losses[0].item()
                val_avg_msg_loss += losses[1].item()
                val_avg_loudness_loss += losses[2].item()

            val_avg_acc[0] /= count
            val_avg_acc[1] /= count
            val_avg_snr /= count
            val_avg_wav_loss /= count
            val_avg_msg_loss /= count
            val_avg_loudness_loss /= count
            if train_config["adv"]:
                val_avg_d_loss_on_encoded /= count
                val_avg_d_loss_on_cover /= count
                val_avg_d_acc /= count
            logging.info("#e" * 60)
            if train_config["adv"]:
                logging.info(
                    "epoch:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} "
                    "- acc:[{:.8f},{:.8f}] - snr:{:.8f} - d_acc:{:.4f} "
                    "- d_loss_on_encoded:{:.6f} - d_loss_on_cover:{:.6f}".format(
                        ep,
                        val_avg_wav_loss,
                        val_avg_msg_loss,
                        val_avg_loudness_loss,
                        val_avg_acc[0],
                        val_avg_acc[1],
                        val_avg_snr,
                        val_avg_d_acc,
                        val_avg_d_loss_on_encoded,
                        val_avg_d_loss_on_cover,
                    )
                )
            else:
                logging.info(
                    "epoch:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} "
                    "- acc:[{:.8f},{:.8f}] - snr:{:.8f}".format(
                        ep,
                        val_avg_wav_loss,
                        val_avg_msg_loss,
                        val_avg_loudness_loss,
                        val_avg_acc[0],
                        val_avg_acc[1],
                        val_avg_snr,
                    )
                )

        val_metrics = {
            "val/wav_loss": val_avg_wav_loss,
            "val/msg_loss": val_avg_msg_loss,
            "val/tfloudness_loss": val_avg_loudness_loss,
            "val/acc_1": val_avg_acc[0],
            "val/acc_2": val_avg_acc[1],
            "val/snr": val_avg_snr,
            "val/d_loss_on_encoded": val_avg_d_loss_on_encoded,
            "val/d_loss_on_cover": val_avg_d_loss_on_cover,
        }
        if train_config["adv"]:
            val_metrics["val/d_acc"] = val_avg_d_acc

        if train_config["wandb"]["enabled"]:
            wandb.log(
                {
                    **train_metrics,
                    **val_metrics,
                    "epoch": ep,
                },
                step=ep,
            )

    # ---------------- test (dev)
    with torch.inference_mode():
        encoder.eval()
        decoder.eval()
        if train_config["adv"]:
            discriminator.eval()
        test_avg_acc = [0.0, 0.0]
        test_avg_snr = 0.0
        test_avg_si_snr = 0.0
        test_avg_pesq = 0.0
        test_avg_stoi = 0.0
        test_avg_wav_loss = 0.0
        test_avg_msg_loss = 0.0
        test_avg_loudness_loss = 0.0
        test_avg_d_loss_on_encoded = 0.0
        test_avg_d_loss_on_cover = 0.0
        test_avg_d_acc = 0.0  # optional

        # CPU pool setup
        ctx = multiprocessing.get_context("spawn")
        max_workers = train_config.get("test", {}).get("cpu_workers", 4)
        pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        futures = []
        n_utts = 0

        count = 0
        for sample in track(dev_audios_loader):
            count += 1
            wav_matrix = sample["matrix"].to(device)
            b = sample["matrix"].shape[0]
            n_utts += b
            # ---------------- build watermark
            msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
            watermark, zeros_right = encoder(wav_matrix, msg, global_step)

            y_wm = wav_matrix + watermark
            decoded = decoder(y_wm, global_step)
            losses = loss.en_de_loss(
                wav_matrix,
                y_wm,
                msg,
                decoded,
            )
            # adv
            if train_config["adv"]:
                cover_target = torch.ones((b, 1), device=device)
                encoded_target = torch.zeros((b, 1), device=device)

                logits_cover = discriminator(wav_matrix)
                logits_encoded = discriminator(y_wm)

                d_loss_on_cover = F.binary_cross_entropy_with_logits(
                    logits_cover, cover_target
                )
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(
                    logits_encoded, encoded_target
                )

                test_avg_d_loss_on_cover += d_loss_on_cover.item()
                test_avg_d_loss_on_encoded += d_loss_on_encoded.item()

                acc_cover = (torch.sigmoid(logits_cover) > 0.5).float().mean()
                acc_encoded = (torch.sigmoid(logits_encoded) < 0.5).float().mean()
                test_avg_d_acc += (0.5 * (acc_cover + acc_encoded)).item()

            decoder_acc = [
                ((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
            ]
            zero_tensor = torch.zeros_like(wav_matrix)
            snr = 10 * torch.log10(
                mse_loss(wav_matrix, zero_tensor) / mse_loss(wav_matrix, y_wm)
            )

            test_avg_acc[0] += decoder_acc[0]
            test_avg_acc[1] += decoder_acc[1]
            test_avg_snr += snr.item()
            test_avg_wav_loss += losses[0].item()
            test_avg_msg_loss += losses[1].item()
            test_avg_loudness_loss += losses[2].item()

            # Submit CPU metrics per utterance
            x_cpu = wav_matrix.detach().cpu().numpy()
            y_cpu = y_wm.detach().cpu().numpy()
            for i in range(b):
                futures.append(
                    pool.submit(cpu_metrics_worker, y_cpu[i], x_cpu[i], sample_rate)
                )

                # Initialize wandb only if enabled in config
                # if train_config["wandb"]["enabled"]:
                #     if count <= num_save_img:
                #         with tempfile.TemporaryDirectory() as tmpdir:
                #             # Normalize each audio signal:
                #             y_norm = normalize_audio(wav_matrix[0].cpu().detach())
                #             y_wm_norm = normalize_audio(y_wm[0].cpu().detach())
                #             wm_norm = normalize_audio(watermark[0].cpu().detach())
                #
                #             original_buf = save_spectrogram_to_buffer(y_norm)
                #             watermarked_buf = save_spectrogram_to_buffer(y_wm_norm)
                #             watermark_buf = save_spectrogram_to_buffer(wm_norm)
                #
                #             test_audio_table.add_data(
                #                 wandb.Audio(wav_matrix[0].cpu().detach().numpy(), sample_rate=sample_rate),
                #                 wandb.Audio(y_wm[0].cpu().detach().numpy(), sample_rate=sample_rate),
                #                 wandb.Audio(watermark[0].cpu().detach().numpy(), sample_rate=sample_rate),
                #                 buffer_to_wandb_image(original_buf),
                #                 buffer_to_wandb_image(watermarked_buf),
                #                 buffer_to_wandb_image(watermark_buf))

        # Collect CPU metric results
        n_valid_pesq = 0
        n_valid_stoi = 0
        for fut in as_completed(futures):
            sisnr_v, stoi_v, pesq_v = fut.result()
            test_avg_si_snr += sisnr_v
            if not np.isnan(stoi_v):
                test_avg_stoi += stoi_v
                n_valid_stoi += 1
            if not np.isnan(pesq_v):
                test_avg_pesq += pesq_v
                n_valid_pesq += 1

        pool.shutdown(wait=True)

        test_avg_acc[0] /= count
        test_avg_acc[1] /= count
        test_avg_snr /= count
        test_avg_wav_loss /= count
        test_avg_msg_loss /= count
        test_avg_loudness_loss /= count

        # Per-utterance averages for CPU metrics
        test_avg_si_snr /= max(n_utts, 1)
        test_avg_stoi = test_avg_stoi / max(n_valid_stoi, 1)
        test_avg_pesq = test_avg_pesq / max(n_valid_pesq, 1)

        if train_config["adv"]:
            test_avg_d_loss_on_encoded /= count
            test_avg_d_loss_on_cover /= count
            test_avg_d_acc /= count

        test_metrics = {
            "test/wav_loss": test_avg_wav_loss,
            "test/msg_loss": test_avg_msg_loss,
            "test/tfloudness_loss": test_avg_loudness_loss,
            "test/acc_1": test_avg_acc[0],
            "test/acc_2": test_avg_acc[1],
            "test/snr": test_avg_snr,
            "test/si_snr_cpu": test_avg_si_snr,
            "test/stoi_cpu": test_avg_stoi,
            "test/pesq_cpu": test_avg_pesq,
        }
        if train_config["adv"]:
            test_metrics["test/d_loss_on_encoded"] = test_avg_d_loss_on_encoded
            test_metrics["test/d_loss_on_cover"] = test_avg_d_loss_on_cover
            test_metrics["test/d_acc"] = test_avg_d_acc

        logging.info("#t" * 60)
        logging.info(
            "test - wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} - "
            "acc:[{:.8f},{:.8f}] - snr:{:.8f} - si_snr_cpu:{:.6f} - stoi_cpu:{:.6f} - pesq_cpu:{:.6f}".format(
                test_avg_wav_loss,
                test_avg_msg_loss,
                test_avg_loudness_loss,
                test_avg_acc[0],
                test_avg_acc[1],
                test_avg_snr,
                test_avg_si_snr,
                test_avg_stoi,
                test_avg_pesq,
            )
        )

        if train_config["wandb"]["enabled"] and wandb.run is not None:
            test_table = wandb.Table(columns=["metric", "value"])
            for k, v in test_metrics.items():
                test_table.add_data(k, float(v))
            wandb.log({"test/results": test_table})


if __name__ == "__main__":
    if torch.cuda.is_available():
        pc = torch.backends.cuda.cufft_plan_cache
        pc.max_size = 256  # start here

        # optional warmup logic
        # run a few warmup batches first, then:
        used = pc.size
        headroom = int(pc.max_size * 1.0)  # 20 percent headroom
        pc.max_size = max(pc.max_size, headroom)
        print({"plans_in_use": used, "max_allowed": pc.max_size})
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--process_config",
        type=str,
        help="path to process.yaml",
        default=r"./config/process.yaml",
    )
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        help="path to model.yaml",
        default=r"./config/model.yaml",
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        help="path to train.yaml",
        default=r"./config/train.yaml",
    )
    args = parser.parse_args()

    # Read Config
    process_config = yaml.load(open(args.process_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(configs)
