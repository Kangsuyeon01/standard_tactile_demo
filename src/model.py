"""
Model definition (LiteSeq2SeqCNNGRU_AttnPool) and composite loss.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .config import (
    AMP_WEIGHT_ALPHA, LAMBDA_POINT, LAMBDA_DIFF, LAMBDA_SPEC, LAMBDA_ENV,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class LiteSeq2SeqCNNGRU_AttnPool(nn.Module):
    """Lightweight 1-D CNN + GRU with attention pooling + FiLM roughness conditioning.

    ch0~2 (acc, force, vel) 는 Conv1D → GRU 로 처리하고,
    ch3 (roughness 0~1) 는 FiLM 네트워크를 통해 conv 출력을
    scale/shift 한다. 상수 채널을 conv 에 그냥 넣으면 bias 효과만
    생기지만, FiLM 을 쓰면 roughness 가 직접 feature map 을 조절한다.
    """

    def __init__(self, in_ch: int = 3, output_steps: int = 40):
        # in_ch: dynamic 채널 수 (acc, force, vel). roughness 는 별도 처리.
        super().__init__()

        # roughness scalar → conv feature 의 gamma / beta
        self.film_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 32 * 2),   # gamma 32 + beta 32
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, 24, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=32, hidden_size=32,
            num_layers=1, batch_first=True, bidirectional=False,
        )
        self.attn = nn.Sequential(
            nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(), nn.Linear(64, output_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x[:, 3, 0:1]               # [B, 1]  roughness 0~1 (상수 채널)
        h = self.conv1(x[:, :3, :])    # [B, 24, T]  dynamic 3채널만
        h = self.conv2(h)              # [B, 32, T]

        # FiLM: roughness 로 feature map 을 scale & shift
        film  = self.film_net(r)                 # [B, 64]
        gamma = film[:, :32].unsqueeze(2)        # [B, 32, 1]
        beta  = film[:, 32:].unsqueeze(2)        # [B, 32, 1]
        h     = gamma * h + beta                 # [B, 32, T]

        h = h.transpose(1, 2)                    # [B, T, 32]
        h, _ = self.gru(h)                       # [B, T, 32]
        w   = torch.softmax(self.attn(h), dim=1) # [B, T, 1]
        ctx = (h * w).sum(dim=1)                 # [B, 32]
        return self.head(ctx)                    # [B, output_steps]


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def weighted_point_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = 1.0 + AMP_WEIGHT_ALPHA * torch.abs(target)
    return ((pred - target) ** 2 * weights).mean()


def diff_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(
        torch.abs((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1]))
    )


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft(pred,   dim=-1)
    true_fft = torch.fft.rfft(target, dim=-1)
    return torch.mean(torch.abs(torch.abs(pred_fft) - torch.abs(true_fft)))


def envelope_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_rms = torch.sqrt(torch.mean(pred   ** 2, dim=-1) + 1e-8)
    true_rms = torch.sqrt(torch.mean(target ** 2, dim=-1) + 1e-8)
    return torch.mean(torch.abs(pred_rms - true_rms))


def total_loss(pred: torch.Tensor, target: torch.Tensor):
    lp = weighted_point_loss(pred, target)
    ld = diff_loss(pred, target)
    ls = spectral_loss(pred, target)
    le = envelope_loss(pred, target)
    loss = LAMBDA_POINT * lp + LAMBDA_DIFF * ld + LAMBDA_SPEC * ls + LAMBDA_ENV * le
    return loss, {
        "point": lp.item(), "diff": ld.item(),
        "spec": ls.item(),  "env":  le.item(),
        "total": loss.item(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.Y = torch.from_numpy(np.asarray(Y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
