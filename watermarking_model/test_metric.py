import torch
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

sr = 16000
B = 8
T = sr * 1  # 1 second

preds = torch.randn(B, T)
target = torch.randn(B, T)

# SI-SNR: returns one value per item if you set reduction, otherwise usually returns scalar average.
si_snr = ScaleInvariantSignalNoiseRatio()
si_snr_val = si_snr(preds, target)
print("SI-SNR:", si_snr_val)

# STOI: returns per-item scores (B,) internally and Metric averages them over updates
stoi = ShortTimeObjectiveIntelligibility(
    sr, extended=True
)  # extended=True often for 16k
stoi_val = stoi(preds, target)
print("STOI:", stoi_val)

# PESQ: expects speech shaped [B, T] or [B, 1, T] depending on version, [B, T] usually works
pesq = PerceptualEvaluationSpeechQuality(sr, mode="wb")
pesq_val = pesq(preds, target)
print("PESQ:", pesq_val)
