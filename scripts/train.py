"""
scripts/train.py
================
Step 2 - Train LiteSeq2SeqCNNGRU_AttnPool on the preprocessed NPZ cache.

Loss 구성:
  - roughness_rms_calibration_loss : roughness별 출력 진폭을 데이터 통계에 맞게 유도
  - roughness_contrastive_loss     : 같은 배치 내 roughness 차이가 클수록 출력 RMS 차이도 크게
  - spectral_centroid_loss         : roughness 높을수록 스펙트럼 무게중심이 높아지도록
  - hf_energy_ratio_loss           : roughness 높을수록 HF 에너지 비율이 높아지도록
  - force_contrastive_loss         : force 높은 쪽의 RMS가 더 크도록 (directionality)

각 loss 가중치는 argparse로 조절 가능. 0으로 설정하면 비활성화.

Usage:
    python -m scripts.train
    python -m scripts.train --npz pt_files/inference_cache_participant.npz --lambda-force 1.0 --note "force contrastive"
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    OUT_DIR, DEVICE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    INPUT_STEPS, OUTPUT_STEPS,
)
from src.model import LiteSeq2SeqCNNGRU_AttnPool, total_loss


# roughness별 목표 RMS / FFT 프로파일 (main()에서 데이터 로드 후 채워짐)
ROUGHNESS_TARGET_RMS:    dict = {}
ROUGHNESS_FFT_PROFILES:  dict = {}   # r -> normalized FFT array [output_steps//2+1]
# Y normalization stats (set in main() — used to normalize RMS targets to model output space)
Y_STD:  float = 1.0
Y_MEAN: float = 0.0

# roughness classification classes (must match training data)
ROUGHNESS_CLASSES = [5, 12, 23, 45, 58, 66, 100]


def roughness_to_class_idx(r: float) -> int:
    return int(np.argmin([abs(r - c) for c in ROUGHNESS_CLASSES]))


def roughness_to_target_rms_hard(roughness_val: float,
                                  rms_min: float = 0.18,
                                  rms_max: float = 0.90,
                                  gamma:   float = 0.60) -> float:
    """
    realtime.py의 roughness_to_target_rms와 동일한 hard-coded 함수 (velocity 독립).
    R=0 → 0.18,  R=100 → 0.90  (5× 차이)
    이 값을 training target으로 사용하면 모델이 roughness별 amplitude를 직접 학습.
    """
    r = np.clip(float(roughness_val) / 100.0, 0.0, 1.0)
    return float(rms_min + (rms_max - rms_min) * (r ** gamma))


def rescale_y_to_target_rms(X: np.ndarray, Y: np.ndarray,
                              scale_min: float = 0.5,
                              scale_max: float = 3.0) -> np.ndarray:
    """
    roughness 그룹 평균 RMS를 target으로 이동시키되,
    그룹 내 상대적 variation(force·velocity 영향)은 그대로 유지.

    기존 per-sample 방식은 R=5+F=6N과 R=5+F=0.3N을 같은 amplitude로
    만들어버려 force 응답을 지웠음. 이 방식은 그 문제를 해결:
      scale_R = target_rms(R) / mean_rms_of_group(R)
      Y_i *= scale_R   ← 그룹 전체에 동일한 scale 적용
    """
    roughness_vals = X[:, 3, 0] * 100.0
    Y_out = Y.copy()

    for r_val in ROUGHNESS_CLASSES:
        mask = np.abs(roughness_vals - r_val) < 5
        if mask.sum() == 0:
            continue
        tgt      = roughness_to_target_rms_hard(float(r_val))
        mean_rms = float(np.sqrt(np.mean(Y[mask] ** 2))) + 1e-8
        scale    = np.clip(tgt / mean_rms, scale_min, scale_max)
        Y_out[mask] = Y[mask] * scale

    print(f"[RESCALE] Y rescaled by group-mean RMS per roughness "
          f"(scale clamp [{scale_min}, {scale_max}])")
    for r_val in [5, 23, 58, 100]:
        mask = np.abs(roughness_vals - r_val) < 5
        if mask.sum() > 0:
            rms_before = float(np.sqrt(np.mean(Y[mask] ** 2)))
            rms_after  = float(np.sqrt(np.mean(Y_out[mask] ** 2)))
            scale_used = rms_after / (rms_before + 1e-8)
            print(f"  R={r_val:3d}: target={roughness_to_target_rms_hard(r_val):.3f}  "
                  f"before={rms_before:.3f}  after={rms_after:.3f}  scale={scale_used:.3f}")
    return Y_out


def compute_roughness_target_rms(X_train: np.ndarray, Y_train: np.ndarray) -> dict:
    roughness_raw = X_train[:, 3, 0] * 100.0
    keys, vals = [], []
    for r in sorted(np.unique(np.round(roughness_raw)).astype(int)):
        mask = np.abs(roughness_raw - r) < 1
        rms  = float(np.sqrt(np.mean(Y_train[mask] ** 2)))
        keys.append(int(r))
        vals.append(rms)
    # F/V 혼합 평균이라 단조증가가 아닐 수 있음 → 누적 최댓값으로 단조증가 강제
    mono_vals = np.maximum.accumulate(vals).tolist()
    result = {k: v for k, v in zip(keys, mono_vals)}
    print("[RMS] roughness별 target RMS (단조증가 보정):")
    for k, raw, mono in zip(keys, vals, mono_vals):
        flag = " *" if abs(raw - mono) > 1e-6 else ""
        print(f"  roughness={k:3d} -> raw={raw:.4f}  mono={mono:.4f}{flag}")
    return result


def interpolate_target_rms(roughness_val: float) -> float:
    keys = sorted(ROUGHNESS_TARGET_RMS.keys())
    vals = [ROUGHNESS_TARGET_RMS[k] for k in keys]
    return float(np.interp(roughness_val, keys, vals))


def compute_roughness_fft_profiles(X_train: np.ndarray, Y_train: np.ndarray) -> dict:
    """
    Training Y windows(length=OUTPUT_STEPS)에서 roughness별 평균 FFT 프로파일 계산.
    Returns dict: r(int) -> normalized FFT array (sums to 1).
    """
    roughness_raw = X_train[:, 3, 0] * 100.0
    result = {}
    for r in sorted(np.unique(np.round(roughness_raw)).astype(int)):
        mask = np.abs(roughness_raw - r) < 1
        Y_r  = Y_train[mask]
        ffts = np.abs(np.fft.rfft(Y_r, axis=1))  # [N, F]
        mean_fft = ffts.mean(0)                    # [F]
        norm_fft = mean_fft / (mean_fft.sum() + 1e-8)
        result[int(r)] = norm_fft.astype(np.float32)
    print("[FFT] roughness FFT profiles computed:")
    for k, v in result.items():
        peak_bin = int(v.argmax())
        print(f"  r={k:3d}  peak_bin={peak_bin}  shape={v.shape}")
    return result


def interpolate_fft_profile(roughness_val: float) -> np.ndarray:
    """Linearly interpolate FFT profile for arbitrary roughness value."""
    keys = sorted(ROUGHNESS_FFT_PROFILES.keys())
    idx  = np.searchsorted(keys, roughness_val)
    if idx == 0:
        return ROUGHNESS_FFT_PROFILES[keys[0]]
    if idx >= len(keys):
        return ROUGHNESS_FFT_PROFILES[keys[-1]]
    r_lo, r_hi = keys[idx - 1], keys[idx]
    alpha = (roughness_val - r_lo) / (r_hi - r_lo + 1e-8)
    p = (1 - alpha) * ROUGHNESS_FFT_PROFILES[r_lo] + alpha * ROUGHNESS_FFT_PROFILES[r_hi]
    return (p / (p.sum() + 1e-8)).astype(np.float32)


# --- Run ID / 디렉토리 관리 ---

def generate_run_id(out_dir: Path) -> str:
    today = datetime.now().strftime("%Y%m%d")
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(runs_dir.glob(f"{today}-*"))
    if existing:
        last_num = int(existing[-1].name.split("-")[-1])
        num = last_num + 1
    else:
        num = 1
    return f"{today}-{num:03d}"


def make_run_dir(out_dir: Path, run_id: str) -> Path:
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# --- Dataset: X와 함께 roughness, force, vel 값도 반환 ---

class SeqDatasetWithRoughness(Dataset):
    """X의 물리 채널에서 roughness/force/vel 값을 읽어 함께 반환."""
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.Y = torch.from_numpy(np.asarray(Y, dtype=np.float32))
        n = len(X)
        # placeholders: overwritten in main() with un-normalized values
        self.roughness = torch.zeros(n, dtype=torch.float32)
        self.force     = torch.zeros(n, dtype=torch.float32)
        self.vel       = torch.zeros(n, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx],
                self.roughness[idx], self.force[idx], self.vel[idx])


# --- Loss 함수들 ---

def roughness_rms_calibration_loss(pred: torch.Tensor,
                                    roughness_values: torch.Tensor,
                                    force_values: torch.Tensor | None = None,
                                    vel_values: torch.Tensor | None = None,
                                    force_thresh: float = 0.1,
                                    vel_thresh: float = 0.005) -> torch.Tensor:
    # Exclude zero-contact samples (force=0 or vel=0): rms_calib contradicts Y=0 target.
    # vel_zero_aug has F>0 but V=0 → must also be excluded.
    if force_values is not None or vel_values is not None:
        contact_mask = torch.ones(len(pred), dtype=torch.bool, device=pred.device)
        if force_values is not None:
            contact_mask = contact_mask & (force_values.detach().abs() >= force_thresh)
        if vel_values is not None:
            contact_mask = contact_mask & (vel_values.detach().abs() >= vel_thresh)
        if contact_mask.sum() < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred             = pred[contact_mask]
        roughness_values = roughness_values[contact_mask]

    r_np = roughness_values.detach().cpu().numpy()
    # Normalize raw targets into the model's output space (pred is z-score normalized).
    # hard-coded target (realtime.py와 동일): R=0→0.18, R=100→0.90 (5× range)
    targets = torch.tensor(
        [roughness_to_target_rms_hard(float(r)) / Y_STD for r in r_np],
        dtype=torch.float32, device=pred.device,
    )
    pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=1) + 1e-8)
    return torch.mean((pred_rms - targets) ** 2)


def roughness_contrastive_loss(pred: torch.Tensor,
                                roughness_values: torch.Tensor,
                                margin: float = 0.08,
                                min_r_diff: float = 20.0) -> torch.Tensor:
    pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=1) + 1e-8)
    r = roughness_values.float()
    r_diff   = torch.abs(r.unsqueeze(0) - r.unsqueeze(1))
    rms_diff = torch.abs(pred_rms.unsqueeze(0) - pred_rms.unsqueeze(1))
    target_diff = r_diff / 100.0 * margin
    loss_mat    = torch.relu(target_diff - rms_diff)
    mask  = (r_diff >= min_r_diff).float()
    count = mask.sum().clamp(min=1)
    return (loss_mat * mask).sum() / count


def spectral_centroid_loss(pred: torch.Tensor,
                            roughness_values: torch.Tensor,
                            target_low: float = 3.0,
                            target_high: float = 12.0) -> torch.Tensor:
    fft  = torch.fft.rfft(pred, dim=-1)
    mag  = torch.abs(fft)
    F    = mag.shape[-1]
    freqs = torch.arange(F, dtype=torch.float32, device=pred.device)
    centroid = (freqs * mag).sum(dim=-1) / (mag.sum(dim=-1) + 1e-8)
    r_norm   = roughness_values.float() / 100.0
    targets  = target_low + (target_high - target_low) * r_norm
    return torch.mean((centroid - targets) ** 2)


def hf_energy_ratio_loss(pred: torch.Tensor,
                          roughness_values: torch.Tensor,
                          cutoff_ratio: float = 0.4,
                          min_r_diff: float = 20.0) -> torch.Tensor:
    fft_mag = torch.abs(torch.fft.rfft(pred, dim=-1))
    F = fft_mag.shape[-1]
    cutoff_bin = max(1, int(F * cutoff_ratio))
    total_energy = torch.sum(fft_mag ** 2, dim=-1) + 1e-8
    hf_energy    = torch.sum(fft_mag[:, cutoff_bin:] ** 2, dim=-1)
    hf_ratio     = hf_energy / total_energy
    r = roughness_values.float()
    r_diff  = r.unsqueeze(0) - r.unsqueeze(1)
    hr_diff = hf_ratio.unsqueeze(0) - hf_ratio.unsqueeze(1)
    pair_mask = (torch.abs(r_diff) >= min_r_diff).float()
    violation = torch.relu(-hr_diff * torch.sign(r_diff))
    count = pair_mask.sum().clamp(min=1)
    return (violation * pair_mask).sum() / count


def spectral_profile_loss(pred: torch.Tensor,
                          roughness_values: torch.Tensor) -> torch.Tensor:
    """
    pred FFT shape 을 roughness별 실측 FFT 프로파일에 맞추도록 학습.
    - 정규화된 FFT (shape only) 비교 -> RMS calibration loss 와 역할 분리
    - 단조 가정 없음, 데이터 실측 기반
    pred: [B, output_steps]   roughness_values: [B] (0~100)
    """
    pred_fft  = torch.abs(torch.fft.rfft(pred, dim=-1))         # [B, F]
    pred_norm = pred_fft / (pred_fft.sum(dim=-1, keepdim=True) + 1e-8)  # [B, F]

    r_np = roughness_values.detach().cpu().numpy()
    targets = torch.tensor(
        np.stack([interpolate_fft_profile(float(r)) for r in r_np]),
        dtype=torch.float32, device=pred.device,
    )  # [B, F]
    return torch.mean((pred_norm - targets) ** 2)


def force_contrastive_loss(pred: torch.Tensor,
                            force_values: torch.Tensor,
                            vel_values: torch.Tensor | None = None,
                            min_f_diff: float = 0.3,
                            vel_thresh: float = 0.005) -> torch.Tensor:
    """
    Force 높은 쪽의 RMS >= Force 낮은 쪽의 RMS 를 강제.
    velocity 낮은 샘플은 제외 — transition aug 샘플(low vel → Y≈0)과 충돌 방지.
    """
    pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=1) + 1e-8)  # [B]
    f = force_values.float()

    # velocity 충분한 샘플만 비교 (양쪽 모두 vel >= thresh)
    if vel_values is not None:
        v = vel_values.float()
        active = (v >= vel_thresh)
        active_pair = active.unsqueeze(0) & active.unsqueeze(1)  # [B, B]
    else:
        active_pair = torch.ones(len(f), len(f), dtype=torch.bool, device=f.device)

    f_diff   = f.unsqueeze(1) - f.unsqueeze(0)
    rms_diff = pred_rms.unsqueeze(1) - pred_rms.unsqueeze(0)

    mask      = (f_diff > min_f_diff).float() * active_pair.float()
    violation = torch.relu(-rms_diff)
    count     = mask.sum().clamp(min=1)
    return (violation * mask).sum() / count


def vel_gate_loss(pred: torch.Tensor,
                  vel_values: torch.Tensor,
                  force_values: torch.Tensor,
                  vel_thresh: float = 0.005,
                  force_thresh: float = 0.3) -> torch.Tensor:
    """
    Force는 충분한데 velocity가 낮으면 출력이 0에 가까워야 함.
    transition aug만으로 부족한 velocity gating을 loss로 직접 강제.
    """
    pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=1) + 1e-8)
    f = force_values.float()
    v = vel_values.float()

    mask = (f >= force_thresh) & (v < vel_thresh)
    if mask.sum() < 1:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return torch.mean(pred_rms[mask] ** 2)  # 해당 샘플 RMS를 0으로


def gate_calibration_loss(gate: torch.Tensor,
                          force_values: torch.Tensor,
                          vel_values: torch.Tensor,
                          force_thresh: float = 0.3, force_full: float = 2.0,
                          vel_thresh: float = 0.005, vel_full: float = 0.03) -> torch.Tensor:
    """
    gate_net 출력을 force_velocity_gate 공식의 target gate 값으로 직접 학습.
    f=0/v=0 → target=0,  f≥force_full & v≥vel_full → target=1.
    gate: [B, 1] (model에서 return_gate=True로 받은 값)
    """
    f = force_values.float()
    v = vel_values.float()
    f_gain = ((f - force_thresh) / (force_full - force_thresh)).clamp(0.0, 1.0)
    v_gain = ((v - vel_thresh)   / (vel_full   - vel_thresh)).clamp(0.0, 1.0)
    target = (f_gain * v_gain).unsqueeze(1)  # [B, 1]
    return torch.mean((gate - target) ** 2)


def roughness_cls_loss(ctx: torch.Tensor,
                       roughness_values: torch.Tensor,
                       roughness_head: nn.Module) -> torch.Tensor:
    """
    ctx [B, 32] → roughness class 예측.
    모델이 hidden state에 roughness 정보를 인코딩하도록 강제.
    학습 중에만 사용; ONNX export / inference 에서는 roughness_head 제외.
    """
    r_np = roughness_values.detach().cpu().numpy()
    labels = torch.tensor(
        [roughness_to_class_idx(float(r)) for r in r_np],
        dtype=torch.long, device=ctx.device,
    )
    logits = roughness_head(ctx)  # [B, n_classes]
    return nn.functional.cross_entropy(logits, labels)


def combined_loss(pred, ctx, gate, target, roughness_values, force_values, vel_values, roughness_head,
                  lambda_rms=2.0, lambda_contrast=0.6,
                  lambda_centroid=0.0, lambda_hf=0.0,
                  lambda_force=1.0, lambda_profile=2.0, lambda_rough_cls=1.0,
                  lambda_vel_gate=2.0, lambda_gate_calib=5.0):
    base_loss, loss_dict = total_loss(pred, target)

    l_rms        = roughness_rms_calibration_loss(pred, roughness_values, force_values, vel_values)
    l_contrast   = roughness_contrastive_loss(pred, roughness_values)
    l_centroid   = spectral_centroid_loss(pred, roughness_values)
    l_hf         = hf_energy_ratio_loss(pred, roughness_values)
    l_force      = force_contrastive_loss(pred, force_values, vel_values)
    l_profile    = spectral_profile_loss(pred, roughness_values)
    l_rough_cls  = roughness_cls_loss(ctx, roughness_values, roughness_head)
    l_vel_gate   = vel_gate_loss(pred, vel_values, force_values)
    l_gate_calib = gate_calibration_loss(gate, force_values, vel_values)

    total = (base_loss
             + lambda_rms        * l_rms
             + lambda_contrast   * l_contrast
             + lambda_centroid   * l_centroid
             + lambda_hf         * l_hf
             + lambda_force      * l_force
             + lambda_profile    * l_profile
             + lambda_rough_cls  * l_rough_cls
             + lambda_vel_gate   * l_vel_gate
             + lambda_gate_calib * l_gate_calib)

    loss_dict.update({
        "rms_calib":   l_rms.item(),
        "contrast":    l_contrast.item(),
        "centroid":    l_centroid.item(),
        "hf_ratio":    l_hf.item(),
        "force_ctr":   l_force.item(),
        "profile":     l_profile.item(),
        "rough_cls":   l_rough_cls.item(),
        "vel_gate":    l_vel_gate.item(),
        "gate_calib":  l_gate_calib.item(),
        "total":       total.item(),
    })
    return total, loss_dict


# --- Training / eval loop ---

def run_epoch(loader, model, optimizer, device, args, train=True):
    model.train() if train else model.eval()

    keys = ("point", "diff", "spec", "env",
            "rms_calib", "contrast", "centroid", "hf_ratio",
            "force_ctr", "profile", "rough_cls", "vel_gate", "gate_calib", "total")
    loss_log  = {k: 0.0 for k in keys}
    total_count = 0
    preds_all, trues_all = [], []

    for xb, yb, rb, fb, vb in loader:
        xb, yb, rb, fb, vb = xb.to(device), yb.to(device), rb.to(device), fb.to(device), vb.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred, ctx, gate = model(xb, return_ctx=True, return_gate=True)
            loss, loss_dict = combined_loss(
                pred, ctx, gate, yb, rb, fb, vb, model.roughness_head,
                lambda_rms=args.lambda_rms,
                lambda_contrast=args.lambda_contrast,
                lambda_centroid=args.lambda_centroid,
                lambda_hf=args.lambda_hf,
                lambda_force=args.lambda_force,
                lambda_profile=args.lambda_profile,
                lambda_rough_cls=args.lambda_rough_cls,
                lambda_vel_gate=args.lambda_vel_gate,
                lambda_gate_calib=args.lambda_gate_calib,
            )
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        bs = xb.size(0)
        total_count += bs
        for k in keys:
            loss_log[k] += loss_dict.get(k, 0.0) * bs

        preds_all.append(pred.detach().cpu().numpy())
        trues_all.append(yb.detach().cpu().numpy())

    for k in keys:
        loss_log[k] /= total_count

    return loss_log, np.concatenate(preds_all), np.concatenate(trues_all)


# --- Main ---

def main(args):
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    NPZ_PATH = Path(args.npz) if args.npz else out_dir / "inference_cache_allinone.npz"
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"NPZ 파일 없음: {NPZ_PATH}\n"
            f"  전처리: python -m scripts.preprocess\n"
            f"  재분할: python -m scripts.resplit_npz"
        )

    run_id  = args.run_id if args.run_id else generate_run_id(out_dir)
    run_dir = make_run_dir(out_dir, run_id)
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"  Run ID : {run_id}")
    print(f"  Dir    : {run_dir}")
    print(f"{'='*60}\n")

    # --- Load data ---
    npz = np.load(NPZ_PATH, allow_pickle=True)
    X_train, Y_train = npz["X_train"], npz["Y_train"]
    X_val,   Y_val   = npz["X_val"],   npz["Y_val"]
    X_test,  Y_test  = npz["X_test"],  npz["Y_test"]

    n_real = len(X_train)  # augmentation 전 실제 샘플 수

    # --- Zero-contact augmentation ---
    if args.zero_augment_ratio > 0:
        n_zero = int(n_real * args.zero_augment_ratio)
        roughness_vals = np.unique(np.round(X_train[:, 3, 0] * 100)).astype(np.float32)
        rng = np.random.default_rng(42)
        r_samples = rng.choice(roughness_vals, size=n_zero) / 100.0

        X_zero = np.zeros((n_zero, X_train.shape[1], X_train.shape[2]), dtype=np.float32)
        X_zero[:, 3, :] = r_samples[:, None]
        Y_zero = np.zeros((n_zero, Y_train.shape[1]), dtype=np.float32)

        X_train = np.concatenate([X_train, X_zero], axis=0)
        Y_train = np.concatenate([Y_train, Y_zero], axis=0)
        print(f"[ZERO AUG] {n_zero} zero-contact samples added "
              f"({args.zero_augment_ratio*100:.0f}% of train) -> total {len(X_train)}")

    # --- Y rescaling (roughness별 target amplitude 강제) ---
    if args.rescale_y_to_target_rms:
        print("\n[RESCALE] Applying per-sample Y amplitude rescaling ...")
        Y_train = rescale_y_to_target_rms(X_train, Y_train)
        Y_val   = rescale_y_to_target_rms(X_val,   Y_val)
        # test는 원본 유지 (평가 공정성)
        print("[RESCALE] Note: Y_test kept as original for fair evaluation.")

    # --- Transition-zone augmentation ---
    # force/velocity가 0~최대값 사이인 전환 구간을 학습에 포함.
    # realtime.py의 force_velocity_gate와 동일한 파라미터로 Y를 스케일링.
    if args.transition_augment_ratio > 0:
        _FORCE_THRESH, _FORCE_FULL = 0.3, 2.0
        _VEL_THRESH,   _VEL_FULL   = 0.005, 0.03

        n_trans = int(n_real * args.transition_augment_ratio)
        rng_trans = np.random.default_rng(43)

        # 실제 샘플에서만 복사 (zero-augment 샘플 제외)
        idx = rng_trans.integers(0, n_real, size=n_trans)
        X_trans = X_train[idx].copy()
        Y_trans = Y_train[idx].copy()

        # force/velocity를 전환 구간 내 랜덤값으로 교체
        # vel 샘플링은 gate 기준(vel_full=0.03)보다 넓게 0.20까지 커버:
        # vel > vel_full이면 v_gain=1.0으로 클램프되므로 force 게이팅만 학습
        f_vals = rng_trans.uniform(0.0, _FORCE_FULL, size=n_trans).astype(np.float32)
        v_vals = rng_trans.uniform(0.0, 0.20,        size=n_trans).astype(np.float32)

        # gate 계산 (force_velocity_gate와 동일한 공식)
        f_gain = np.clip((f_vals - _FORCE_THRESH) / (_FORCE_FULL - _FORCE_THRESH), 0.0, 1.0)
        v_gain = np.clip((v_vals - _VEL_THRESH)   / (_VEL_FULL   - _VEL_THRESH),   0.0, 1.0)
        gate   = (f_gain * v_gain).astype(np.float32)

        X_trans[:, 1, :] = f_vals[:, None]  # force 채널 교체
        X_trans[:, 2, :] = v_vals[:, None]  # velocity 채널 교체
        Y_trans = Y_trans * gate[:, None]   # gate 비율로 Y 감쇄

        X_train = np.concatenate([X_train, X_trans], axis=0)
        Y_train = np.concatenate([Y_train, Y_trans], axis=0)
        print(f"[TRANSITION AUG] {n_trans} transition samples added "
              f"({args.transition_augment_ratio*100:.0f}% of real train) -> total {len(X_train)}")

    # --- Hard-gate augmentation ---
    # real acc 히스토리 + force=0, vel=0 → Y=0
    # transition aug만으로는 force=0 샘플 비중이 낮아서 별도 추가.
    if args.hard_gate_augment_ratio > 0:
        n_hard = int(n_real * args.hard_gate_augment_ratio)
        rng_hard = np.random.default_rng(44)

        idx = rng_hard.integers(0, n_real, size=n_hard)
        X_hard = X_train[idx].copy()
        Y_hard = np.zeros((n_hard, Y_train.shape[1]), dtype=np.float32)

        X_hard[:, 1, :] = 0.0  # force=0
        X_hard[:, 2, :] = 0.0  # velocity=0
        # acc 채널(ch0)은 real 신호 그대로 유지 → "acc 있어도 force=0이면 출력=0" 학습

        X_train = np.concatenate([X_train, X_hard], axis=0)
        Y_train = np.concatenate([Y_train, Y_hard], axis=0)
        print(f"[HARD GATE AUG] {n_hard} hard-gate samples added "
              f"({args.hard_gate_augment_ratio*100:.0f}% of real train) -> total {len(X_train)})")

    # --- Contact-end augmentation ---
    # real acc 히스토리가 있는 상태에서 force/vel이 윈도우 중간에 0으로 떨어지는 샘플.
    # gate_net이 보는 x[:, 1, -1] / x[:, 2, -1] = 0이 되므로 Y=0을 직접 학습.
    # "과거 acc가 있어도 현재 force/vel=0이면 즉시 출력=0" 케이스를 커버.
    if args.contact_end_augment_ratio > 0:
        n_end = int(n_real * args.contact_end_augment_ratio)
        rng_end = np.random.default_rng(45)

        idx = rng_end.integers(0, n_real, size=n_end)
        X_end = X_train[idx].copy()
        Y_end = np.zeros((n_end, Y_train.shape[1]), dtype=np.float32)

        # 윈도우 후반부(50~100% 지점)에서 force/vel을 0으로 드롭 → 마지막 샘플은 항상 0
        drop_t = rng_end.integers(INPUT_STEPS // 2, INPUT_STEPS, size=n_end)
        for i, t in enumerate(drop_t):
            X_end[i, 1, t:] = 0.0  # force drop
            X_end[i, 2, t:] = 0.0  # velocity drop

        X_train = np.concatenate([X_train, X_end], axis=0)
        Y_train = np.concatenate([Y_train, Y_end], axis=0)
        print(f"[CONTACT-END AUG] {n_end} contact-end samples added "
              f"({args.contact_end_augment_ratio*100:.0f}% of real train) -> total {len(X_train)}")

    # --- Force-zero augmentation ---
    # real acc + real vel + force=0 → Y=0
    # "force 없으면 velocity가 뭐든 출력=0" 을 직접 학습.
    # hard_gate_aug(F=0,V=0)만으로는 V≠0일 때 F=0 케이스를 커버 못함.
    if args.force_zero_augment_ratio > 0:
        n_fz = int(n_real * args.force_zero_augment_ratio)
        rng_fz = np.random.default_rng(46)

        idx = rng_fz.integers(0, n_real, size=n_fz)
        X_fz = X_train[idx].copy()
        Y_fz = np.zeros((n_fz, Y_train.shape[1]), dtype=np.float32)
        X_fz[:, 1, :] = 0.0   # force=0, velocity는 real 값 유지

        X_train = np.concatenate([X_train, X_fz], axis=0)
        Y_train = np.concatenate([Y_train, Y_fz], axis=0)
        print(f"[FORCE-ZERO AUG] {n_fz} samples added "
              f"({args.force_zero_augment_ratio*100:.0f}% of real train) -> total {len(X_train)}")

    # --- Velocity-zero augmentation ---
    # real acc + real force + vel=0 → Y=0
    # "velocity 없으면 force가 얼마든 출력=0" 을 직접 학습.
    if args.vel_zero_augment_ratio > 0:
        n_vz = int(n_real * args.vel_zero_augment_ratio)
        rng_vz = np.random.default_rng(47)

        idx = rng_vz.integers(0, n_real, size=n_vz)
        X_vz = X_train[idx].copy()
        Y_vz = np.zeros((n_vz, Y_train.shape[1]), dtype=np.float32)
        X_vz[:, 2, :] = 0.0   # vel=0, force는 real 값 유지

        X_train = np.concatenate([X_train, X_vz], axis=0)
        Y_train = np.concatenate([Y_train, Y_vz], axis=0)
        print(f"[VEL-ZERO AUG] {n_vz} samples added "
              f"({args.vel_zero_augment_ratio*100:.0f}% of real train) -> total {len(X_train)}")

    print("[DATA SHAPES]")
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        print(f"  {name}: {arr.shape}")

    report: dict = {
        "run_id":    run_id,
        "note":      args.note,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": {
            "lambda_rms":      args.lambda_rms,
            "lambda_contrast": args.lambda_contrast,
            "lambda_centroid": args.lambda_centroid,
            "lambda_hf":       args.lambda_hf,
            "lambda_force":    args.lambda_force,
            "out_dir":         str(out_dir),
        },
        "config": {
            "batch_size":    BATCH_SIZE,
            "epochs":        args.epochs if args.epochs is not None else EPOCHS,
            "lr":            LR,
            "weight_decay":  WEIGHT_DECAY,
            "input_steps":   INPUT_STEPS,
            "output_steps":  OUTPUT_STEPS,
        },
        "data": {
            "npz_path":  str(NPZ_PATH),
            "n_train":   int(X_train.shape[0]),
            "n_val":     int(X_val.shape[0]),
            "n_test":    int(X_test.shape[0]),
        },
        "training":  {},
        "test":      {},
        "roughness_rms": {},
        "model_path": str(run_dir / "best_model.pt"),
    }

    # --- Normalise (train statistics) ---
    x_mean = X_train[:, :3, :].mean(axis=(0, 2), keepdims=True).astype(np.float32)
    x_std  = (X_train[:, :3, :].std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32)
    y_mean = float(Y_train.mean())
    y_std  = float(Y_train.std() + 1e-8)

    def norm_x(X):
        Xn = X.copy()
        Xn[:, :3, :] = (X[:, :3, :] - x_mean) / x_std
        return Xn

    def norm_y(Y): return (Y - y_mean) / y_std

    global ROUGHNESS_TARGET_RMS, ROUGHNESS_FFT_PROFILES, Y_STD, Y_MEAN
    Y_STD = y_std
    Y_MEAN = y_mean
    ROUGHNESS_TARGET_RMS   = compute_roughness_target_rms(X_train, Y_train)
    ROUGHNESS_FFT_PROFILES = compute_roughness_fft_profiles(X_train, Y_train)

    train_ds = SeqDatasetWithRoughness(norm_x(X_train), norm_y(Y_train))
    val_ds   = SeqDatasetWithRoughness(norm_x(X_val),   norm_y(Y_val))
    test_ds  = SeqDatasetWithRoughness(norm_x(X_test),  norm_y(Y_test))

    # Set un-normalized roughness / force / vel (from original X, before norm_x)
    train_ds.roughness = torch.from_numpy((X_train[:, 3, 0] * 100.0).astype(np.float32))
    val_ds.roughness   = torch.from_numpy((X_val[:,   3, 0] * 100.0).astype(np.float32))
    test_ds.roughness  = torch.from_numpy((X_test[:,  3, 0] * 100.0).astype(np.float32))

    # Use LAST sample to match gate_net which reads x[:, 1, -1] / x[:, 2, -1]
    train_ds.force = torch.from_numpy(X_train[:, 1, -1].astype(np.float32))
    val_ds.force   = torch.from_numpy(X_val[:,   1, -1].astype(np.float32))
    test_ds.force  = torch.from_numpy(X_test[:,  1, -1].astype(np.float32))

    train_ds.vel = torch.from_numpy(X_train[:, 2, -1].astype(np.float32))
    val_ds.vel   = torch.from_numpy(X_val[:,   2, -1].astype(np.float32))
    test_ds.vel  = torch.from_numpy(X_test[:,  2, -1].astype(np.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n[LOSS WEIGHTS]  rms={args.lambda_rms}  contrast={args.lambda_contrast}"
          f"  hf={args.lambda_hf}  centroid={args.lambda_centroid}"
          f"  force={args.lambda_force}  rough_cls={args.lambda_rough_cls}")

    # --- Model & optimiser ---
    model = LiteSeq2SeqCNNGRU_AttnPool(in_ch=3, output_steps=OUTPUT_STEPS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    history = {"train_total": [], "val_total": [],
               "train_rms": [], "train_contrast": [],
               "train_centroid": [], "train_hf": [], "train_force": [],
               "train_profile": [], "train_rough_cls": []}
    best_val, best_state, best_epoch = np.inf, None, -1
    patience_cnt, wait = 12, 0
    n_epochs = args.epochs if args.epochs is not None else EPOCHS

    print("\n[TRAINING]")
    for epoch in range(1, n_epochs + 1):
        train_log, _, _ = run_epoch(train_loader, model, optimizer, DEVICE, args, train=True)
        val_log,   _, _ = run_epoch(val_loader,   model, optimizer, DEVICE, args, train=False)

        scheduler.step(val_log["total"])

        history["train_total"].append(train_log["total"])
        history["val_total"].append(val_log["total"])
        history["train_rms"].append(train_log["rms_calib"])
        history["train_contrast"].append(train_log["contrast"])
        history["train_centroid"].append(train_log["centroid"])
        history["train_hf"].append(train_log["hf_ratio"])
        history["train_force"].append(train_log["force_ctr"])
        history["train_profile"].append(train_log["profile"])
        history["train_rough_cls"].append(train_log["rough_cls"])

        print(f"  [{epoch:03d}/{n_epochs}] "
              f"train={train_log['total']:.4f}  val={val_log['total']:.4f}  "
              f"rms={train_log['rms_calib']:.4f}  "
              f"force={train_log['force_ctr']:.4f}  "
              f"profile={train_log['profile']:.4f}  "
              f"rough_cls={train_log['rough_cls']:.4f}")

        if val_log["total"] < best_val:
            best_val   = val_log["total"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience_cnt:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest epoch: {best_epoch}  best_val: {best_val:.4f}")

    report["training"] = {
        "best_epoch":    best_epoch,
        "best_val_loss": round(float(best_val), 6),
        "final_epoch":   epoch,
        "history": {
            "train_total": [round(v, 6) for v in history["train_total"]],
            "val_total":   [round(v, 6) for v in history["val_total"]],
        },
    }

    # --- Test evaluation ---
    test_log, pred_test_n, true_test_n = run_epoch(
        test_loader, model, optimizer, DEVICE, args, train=False,
    )
    pred_test = pred_test_n * y_std + y_mean
    true_test = true_test_n * y_std + y_mean

    rmse = float(np.sqrt(np.mean((pred_test - true_test) ** 2)))
    mae  = float(np.mean(np.abs(pred_test - true_test)))
    pf, tf = pred_test.reshape(-1), true_test.reshape(-1)
    corr = float(np.corrcoef(pf, tf)[0, 1]) if np.std(pf) > 0 and np.std(tf) > 0 else np.nan
    print(f"\n[TEST]  RMSE={rmse:.4f}  MAE={mae:.4f}  Corr={corr:.4f}")

    report["test"] = {
        "rmse": round(rmse, 6),
        "mae":  round(mae,  6),
        "corr": round(corr, 6) if not np.isnan(corr) else None,
    }

    test_roughness = test_ds.roughness.numpy()
    print("\n[TEST roughness별 출력 RMS]")
    rms_rows = {}
    for r in sorted(np.unique(np.round(test_roughness)).astype(int)):
        mask = np.abs(test_roughness - r) < 1
        if mask.sum() == 0:
            continue
        rms_vals   = np.sqrt(np.mean(pred_test[mask] ** 2, axis=1))
        pred_rms   = float(np.mean(rms_vals))
        target_rms = interpolate_target_rms(r)
        print(f"  r={r:3d} | pred_rms={pred_rms:.4f} (target={target_rms:.4f})")
        rms_rows[str(r)] = {"pred_rms": round(pred_rms, 6), "target_rms": round(target_rms, 6)}
    report["roughness_rms"] = rms_rows

    # --- Save model ---
    PT_PATH = run_dir / "best_model.pt"
    torch.save({
        "run_id": run_id,
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "config": {
            "seg_target_len":  4000,
            "input_steps":     INPUT_STEPS,
            "output_steps":    OUTPUT_STEPS,
            "lambda_rms":      args.lambda_rms,
            "lambda_contrast": args.lambda_contrast,
            "lambda_centroid": args.lambda_centroid,
            "lambda_hf":       args.lambda_hf,
            "lambda_force":    args.lambda_force,
        },
    }, PT_PATH)
    print(f"[SAVE] model -> {PT_PATH}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Run: {run_id}  |  RMSE={rmse:.4f}  Corr={corr:.4f}", fontsize=11)

    axes[0].plot(history["train_total"], label="train total")
    axes[0].plot(history["val_total"],   label="val total")
    axes[0].axvline(best_epoch - 1, color="red", linestyle="--",
                    alpha=0.5, label=f"best epoch {best_epoch}")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    # Only plot losses that are actually active (lambda > 0) to avoid scale distortion
    loss_items = [
        (history["train_rms"],       f"rms_calib (x{args.lambda_rms})",           args.lambda_rms),
        (history["train_contrast"],  f"contrast (x{args.lambda_contrast})",        args.lambda_contrast),
        (history["train_force"],     f"force_ctr (x{args.lambda_force})",          args.lambda_force),
        (history["train_profile"],   f"spectral_profile (x{args.lambda_profile})", args.lambda_profile),
        (history["train_rough_cls"], f"rough_cls (x{args.lambda_rough_cls})",      args.lambda_rough_cls),
        (history["train_centroid"],  f"centroid (x{args.lambda_centroid})",         args.lambda_centroid),
    ]
    for data, label, lam in loss_items:
        if lam > 0:
            axes[1].plot(data, label=label)
    axes[1].set_title("Loss Components (active only)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plot_path = run_dir / "training_history.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[SAVE] plot -> {plot_path}")

    # --- Report ---
    elapsed = time.time() - start_time
    report["finished_at"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report["elapsed_sec"]   = round(elapsed, 1)
    report["elapsed_human"] = (
        f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"
    )
    report["model_path"] = str(PT_PATH)

    report_path = run_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] report -> {report_path}")

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Run ID   : {run_id}")
    if args.note:
        print(f"  Note     : {args.note}")
    print(f"  Model    : {PT_PATH}")
    print(f"  Epochs   : {best_epoch} / {report['training']['final_epoch']}")
    print(f"  Val loss : {best_val:.4f}")
    print(f"  RMSE     : {rmse:.4f}  MAE: {mae:.4f}  Corr: {corr:.4f}")
    print(f"  Elapsed  : {report['elapsed_human']}")
    print(f"{sep}")

    # --- Auto eval ---
    if not getattr(args, "no_eval", False):
        import subprocess, sys as _sys
        cwd = Path(__file__).resolve().parent.parent

        print("\n[AUTO EVAL 1/3] eval_roughness_signal.py ...")
        r1 = subprocess.run(
            [_sys.executable, "-m", "scripts.eval_roughness_signal",
             "--run-id", run_id, "--out-dir", str(out_dir)],
            cwd=cwd,
        )
        if r1.returncode != 0:
            print("[AUTO EVAL 1/3] failed (exit code", r1.returncode, ")")

        print("\n[AUTO EVAL 2/3] eval_comprehensive.py ...")
        r2 = subprocess.run(
            [_sys.executable, "-m", "scripts.eval_comprehensive",
             "--run-id", run_id, "--out-dir", str(out_dir)]
            + (["--npz", str(NPZ_PATH)] if NPZ_PATH else []),
            cwd=cwd,
        )
        if r2.returncode != 0:
            print("[AUTO EVAL 2/3] failed (exit code", r2.returncode, ")")

        print("\n[AUTO EVAL 3/3] eval_test_samples.py ...")
        # Use allinone NPZ for test-sample eval (participant NPZ has mismatched counts)
        _allinone_npz = Path(out_dir) / "inference_cache_allinone.npz"
        _ts_npz = str(_allinone_npz) if _allinone_npz.exists() else (str(NPZ_PATH) if NPZ_PATH else None)
        r3 = subprocess.run(
            [_sys.executable, "-m", "scripts.eval_test_samples",
             "--run-id", run_id, "--out-dir", str(out_dir)]
            + (["--npz", _ts_npz] if _ts_npz else []),
            cwd=cwd,
        )
        if r3.returncode != 0:
            print("[AUTO EVAL 3/3] failed (exit code", r3.returncode, ")")

        print("\n[AUTO EVAL 4/4] gen_ramp_plots.py ...")
        # ONNX export (없으면 자동 생성)
        _onnx = run_dir / "best_model.onnx"
        if not _onnx.exists():
            print("[AUTO EVAL 4/4] exporting ONNX ...")
            subprocess.run(
                [_sys.executable, "-m", "scripts.realtime",
                 "--pt-path", str(run_dir / "best_model.pt"),
                 "--cache-path", str(_allinone_npz),
                 "--onnx-path", str(_onnx)],
                cwd=cwd,
            )
        if _onnx.exists() and _allinone_npz.exists():
            r4 = subprocess.run(
                [_sys.executable, "-m", "scripts.gen_ramp_plots",
                 "--pt-path", str(run_dir / "best_model.pt"),
                 "--cache-path", str(_allinone_npz),
                 "--out-dir", str(run_dir)],
                cwd=cwd,
            )
            if r4.returncode != 0:
                print("[AUTO EVAL 4/4] failed (exit code", r4.returncode, ")")
        else:
            print("[AUTO EVAL 4/4] skipped (ONNX or cache not found)")


# --- Argparse ---

def build_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out-dir",  type=str, default=str(OUT_DIR))
    p.add_argument("--npz",      type=str, default=None)
    p.add_argument("--run-id",   type=str, default=None)
    p.add_argument("--note",     type=str, default="")
    p.add_argument("--lambda-rms",      type=float, default=10.0)
    p.add_argument("--lambda-contrast", type=float, default=0.6)
    p.add_argument("--lambda-centroid", type=float, default=0.0)
    p.add_argument("--lambda-hf",       type=float, default=0.0,
                   help="HF ratio loss (disabled by default: data shows HF decreases with roughness)")
    p.add_argument("--lambda-force",    type=float, default=1.0,
                   help="Force contrastive loss weight (higher F -> higher RMS)")
    p.add_argument("--lambda-profile",  type=float, default=2.0,
                   help="Spectral profile loss: match roughness-specific FFT shape from training data")
    p.add_argument("--lambda-rough-cls", type=float, default=1.0,
                   help="Roughness cls loss on hidden ctx (train-only, zero inference overhead)")
    p.add_argument("--hard-gate-augment-ratio", type=float, default=0.10,
                   help="real acc + force=0, vel=0 → Y=0 샘플 추가 비율")
    p.add_argument("--contact-end-augment-ratio", type=float, default=0.15,
                   help="acc 있다가 force/vel이 윈도우 중간에 0으로 떨어지는 샘플 추가 비율")
    p.add_argument("--force-zero-augment-ratio", type=float, default=0.15,
                   help="real acc + real vel + force=0 → Y=0 샘플 비율 (F=0이면 vel 무관 출력=0 학습)")
    p.add_argument("--vel-zero-augment-ratio", type=float, default=0.15,
                   help="real acc + real force + vel=0 → Y=0 샘플 비율 (V=0이면 force 무관 출력=0 학습)")
    p.add_argument("--lambda-vel-gate", type=float, default=5.0,
                   help="Velocity gate loss: force 있는데 vel 낮으면 출력 0으로 강제")
    p.add_argument("--lambda-gate-calib", type=float, default=10.0,
                   help="Gate calibration loss: gate_net 출력을 force_velocity_gate 공식 target으로 직접 학습")
    p.add_argument("--epochs", type=int, default=None,
                   help="학습 epoch 수 (미지정 시 src.config.EPOCHS 사용)")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--zero-augment-ratio", type=float, default=0.05)
    p.add_argument("--transition-augment-ratio", type=float, default=0.20,
                   help="force/velocity 전환 구간 augmentation 비율. "
                        "실제 샘플의 force/vel을 0~최대값으로 교체하고 Y를 gate 비율로 감쇄.")
    p.add_argument("--rescale-y-to-target-rms", action="store_true",
                   help="훈련 Y를 roughness별 target RMS로 per-sample rescaling. "
                        "모델이 roughness→amplitude 관계를 직접 학습.")
    return p.parse_args()


if __name__ == "__main__":
    main(build_args())
