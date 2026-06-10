"""
scripts/train.py
================
Step 2 – Train LiteSeq2SeqCNNGRU_AttnPool on the preprocessed NPZ cache.

추가된 Loss:
  - roughness_rms_calibration_loss : roughness별 출력 진폭을 데이터 통계에 맞게 유도
  - roughness_contrastive_loss     : 같은 배치 내 roughness 차이가 클수록 출력 RMS 차이도 크게
  - spectral_centroid_loss         : roughness 높을수록 스펙트럼 무게중심이 높아지도록

각 loss 가중치는 argparse로 조절 가능. 0으로 설정하면 비활성화.

Usage:
    python -m scripts.train
    python -m scripts.train --lambda-rms 0.3 --lambda-contrast 0.1 --lambda-centroid 0.05
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


# ──────────────────────────────────────────────────────────────────────────────
# roughness별 목표 RMS
# - 하드코딩 대신 학습 데이터에서 실측 계산 (compute_roughness_target_rms 참고)
# - 중간값(val/test roughness)은 선형 보간으로 자동 계산
# ──────────────────────────────────────────────────────────────────────────────
ROUGHNESS_TARGET_RMS: dict = {}   # main() 에서 데이터 로드 후 채워짐


def compute_roughness_target_rms(X_train: np.ndarray, Y_train: np.ndarray) -> dict:
    """학습 데이터에서 roughness별 실측 RMS 를 계산해 반환."""
    roughness_raw = X_train[:, 3, 0] * 100.0   # ch3: 0~1 → 0~100
    result = {}
    for r in sorted(np.unique(np.round(roughness_raw)).astype(int)):
        mask = np.abs(roughness_raw - r) < 1
        rms  = float(np.sqrt(np.mean(Y_train[mask] ** 2)))
        result[int(r)] = rms
    print("[RMS] roughness별 실측 target RMS:")
    for k, v in result.items():
        print(f"  roughness={k:3d} → {v:.4f}")
    return result


def interpolate_target_rms(roughness_val: float) -> float:
    """학습 데이터 roughness 범위 밖이면 외삽, 안이면 선형 보간."""
    keys = sorted(ROUGHNESS_TARGET_RMS.keys())
    vals = [ROUGHNESS_TARGET_RMS[k] for k in keys]
    return float(np.interp(roughness_val, keys, vals))


# ──────────────────────────────────────────────────────────────────────────────
# Run ID / 디렉토리 관리
# ──────────────────────────────────────────────────────────────────────────────

def generate_run_id(out_dir: Path) -> str:
    """YYYYMMDD-NNN 형식 run ID 생성. 같은 날 실행 시 번호 자동 증가."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Dataset: X와 함께 roughness 값도 반환
# ──────────────────────────────────────────────────────────────────────────────
class SeqDatasetWithRoughness(Dataset):
    """X의 roughness 채널(ch3)에서 roughness 0~1 값을 읽어 함께 반환."""
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.Y = torch.from_numpy(np.asarray(Y, dtype=np.float32))
        # roughness 채널(ch3)은 0~1 정규화된 값 → 0~100으로 복원
        self.roughness = torch.from_numpy(
            np.asarray(X[:, 3, 0] * 100.0, dtype=np.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.roughness[idx]


# ──────────────────────────────────────────────────────────────────────────────
# 추가 Loss 함수들
# ──────────────────────────────────────────────────────────────────────────────

def roughness_rms_calibration_loss(pred: torch.Tensor,
                                    roughness_values: torch.Tensor) -> torch.Tensor:
    """
    roughness별 기대 RMS에 가깝도록 유도. [완전 벡터화]
    pred: [B, output_steps]  roughness_values: [B] (0~100)
    """
    # 목표 RMS를 numpy 보간 → tensor 변환
    r_np = roughness_values.detach().cpu().numpy()
    targets = torch.tensor(
        [interpolate_target_rms(float(r)) for r in r_np],
        dtype=torch.float32, device=pred.device,
    )  # [B]
    pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=1) + 1e-8)  # [B]
    return torch.mean((pred_rms - targets) ** 2)


def roughness_contrastive_loss(pred: torch.Tensor,
                                roughness_values: torch.Tensor,
                                margin: float = 0.08,
                                min_r_diff: float = 20.0) -> torch.Tensor:
    """
    roughness 차이가 min_r_diff 이상인 쌍에 대해
    출력 RMS 차이를 강제. [완전 벡터화 - for loop 없음]
    pred: [B, output_steps]  roughness_values: [B] (0~100)
    """
    pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=1) + 1e-8)  # [B]

    # 모든 쌍의 roughness 차이, RMS 차이를 한 번에 계산
    r = roughness_values.float()                          # [B]
    r_diff = torch.abs(r.unsqueeze(0) - r.unsqueeze(1))  # [B, B]
    rms_diff = torch.abs(pred_rms.unsqueeze(0) - pred_rms.unsqueeze(1))  # [B, B]

    target_diff = r_diff / 100.0 * margin                # [B, B]
    loss_mat = torch.relu(target_diff - rms_diff)         # [B, B]

    # min_r_diff 미만 쌍 마스킹
    mask = (r_diff >= min_r_diff).float()
    count = mask.sum().clamp(min=1)
    return (loss_mat * mask).sum() / count


def spectral_centroid_loss(pred: torch.Tensor,
                            roughness_values: torch.Tensor,
                            target_low: float = 3.0,
                            target_high: float = 12.0) -> torch.Tensor:
    """
    roughness 높을수록 스펙트럼 무게중심이 높아지도록 유도. [완전 벡터화]
    pred: [B, output_steps]  roughness_values: [B] (0~100)
    """
    fft  = torch.fft.rfft(pred, dim=-1)                            # [B, F]
    mag  = torch.abs(fft)                                          # [B, F]
    F    = mag.shape[-1]
    freqs = torch.arange(F, dtype=torch.float32, device=pred.device)  # [F]

    centroid = (freqs * mag).sum(dim=-1) / (mag.sum(dim=-1) + 1e-8)  # [B]

    r_norm  = roughness_values.float() / 100.0                    # [B]
    targets = target_low + (target_high - target_low) * r_norm    # [B]

    return torch.mean((centroid - targets) ** 2)


def hf_energy_ratio_loss(pred: torch.Tensor,
                          roughness_values: torch.Tensor,
                          cutoff_ratio: float = 0.4,
                          min_r_diff: float = 20.0) -> torch.Tensor:
    """
    roughness 차이가 min_r_diff 이상인 쌍에서
    roughness 높은 쪽의 HF 에너지 비율이 더 크도록 강제.

    centroid loss와 달리 절대 목표값 없이 '방향'만 강제해
    재구성 loss와 간섭이 적다.

    pred: [B, output_steps]   roughness_values: [B] (0~100)
    cutoff_ratio: 전체 주파수 중 이 비율 이상을 HF로 정의 (기본 40% = 200Hz/500Hz)
    """
    fft_mag = torch.abs(torch.fft.rfft(pred, dim=-1))   # [B, F]
    F = fft_mag.shape[-1]
    cutoff_bin = max(1, int(F * cutoff_ratio))

    total_energy = torch.sum(fft_mag ** 2, dim=-1) + 1e-8  # [B]
    hf_energy    = torch.sum(fft_mag[:, cutoff_bin:] ** 2, dim=-1)
    hf_ratio     = hf_energy / total_energy                 # [B]  0~1

    r = roughness_values.float()                            # [B]
    r_diff  = r.unsqueeze(0) - r.unsqueeze(1)              # [B, B]  signed
    hr_diff = hf_ratio.unsqueeze(0) - hf_ratio.unsqueeze(1)  # [B, B]  signed

    # roughness 높은 쪽(r_diff > 0) → hf_ratio도 높아야(hr_diff > 0) → 위반하면 패널티
    pair_mask = (torch.abs(r_diff) >= min_r_diff).float()
    violation = torch.relu(-hr_diff * torch.sign(r_diff))  # [B, B]  위반량

    count = pair_mask.sum().clamp(min=1)
    return (violation * pair_mask).sum() / count


def combined_loss(pred, target, roughness_values,
                  lambda_rms=2.0, lambda_contrast=0.6,
                  lambda_centroid=0.0, lambda_hf=0.5):
    """기존 total_loss + roughness-aware loss."""
    base_loss, loss_dict = total_loss(pred, target)

    l_rms      = roughness_rms_calibration_loss(pred, roughness_values)
    l_contrast = roughness_contrastive_loss(pred, roughness_values)
    l_centroid = spectral_centroid_loss(pred, roughness_values)
    l_hf       = hf_energy_ratio_loss(pred, roughness_values)

    total = (base_loss
             + lambda_rms      * l_rms
             + lambda_contrast * l_contrast
             + lambda_centroid * l_centroid
             + lambda_hf       * l_hf)

    loss_dict.update({
        "rms_calib":  l_rms.item(),
        "contrast":   l_contrast.item(),
        "centroid":   l_centroid.item(),
        "hf_ratio":   l_hf.item(),
        "total":      total.item(),
    })
    return total, loss_dict


# ──────────────────────────────────────────────────────────────────────────────
# Training / eval loop
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(loader, model, optimizer, device, args, train=True):
    model.train() if train else model.eval()

    keys = ("point", "diff", "spec", "env", "rms_calib", "contrast", "centroid", "hf_ratio", "total")
    loss_log = {k: 0.0 for k in keys}
    total_count = 0
    preds_all, trues_all = [], []

    for xb, yb, rb in loader:
        xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss, loss_dict = combined_loss(
                pred, yb, rb,
                lambda_rms=args.lambda_rms,
                lambda_contrast=args.lambda_contrast,
                lambda_centroid=args.lambda_centroid,
                lambda_hf=args.lambda_hf,
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


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # --npz 지정 시 그 경로 사용, 아니면 out_dir 기본 경로
    if args.npz:
        NPZ_PATH = Path(args.npz)
    else:
        NPZ_PATH = out_dir / "inference_cache_allinone.npz"
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"NPZ 파일 없음: {NPZ_PATH}\n"
            f"  전처리: python -m scripts.preprocess\n"
            f"  재분할: python -m scripts.resplit_npz"
        )

    # ── Run ID / 디렉토리 ────────────────────────────────────────────────────
    run_id  = args.run_id if args.run_id else generate_run_id(out_dir)
    run_dir = make_run_dir(out_dir, run_id)
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"  Run ID : {run_id}")
    print(f"  Dir    : {run_dir}")
    print(f"{'='*60}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    npz = np.load(NPZ_PATH, allow_pickle=True)
    X_train, Y_train = npz["X_train"], npz["Y_train"]
    X_val,   Y_val   = npz["X_val"],   npz["Y_val"]
    X_test,  Y_test  = npz["X_test"],  npz["Y_test"]

    print("[DATA SHAPES]")
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        print(f"  {name}: {arr.shape}")

    # ── Report 초기화 ────────────────────────────────────────────────────────
    report: dict = {
        "run_id":    run_id,
        "note":      args.note,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": {
            "lambda_rms":      args.lambda_rms,
            "lambda_contrast": args.lambda_contrast,
            "lambda_centroid": args.lambda_centroid,
            "lambda_hf":       args.lambda_hf,
            "out_dir":         str(out_dir),
        },
        "config": {
            "batch_size":    BATCH_SIZE,
            "epochs":        EPOCHS,
            "lr":            LR,
            "weight_decay":  WEIGHT_DECAY,
            "input_steps":   INPUT_STEPS,
            "output_steps":  OUTPUT_STEPS,
        },
        "data": {
            "npz_path":      str(NPZ_PATH),
            "n_train":       int(X_train.shape[0]),
            "n_val":         int(X_val.shape[0]),
            "n_test":        int(X_test.shape[0]),
        },
        "training":  {},
        "test":      {},
        "roughness_rms": {},
        "model_path": str(run_dir / "best_model.pt"),
    }

    # ── Normalise (train statistics) ─────────────────────────────────────────
    # ch0~2 (acc, force, vel) 만 정규화. ch3 (roughness 0~1) 는 그대로 유지해
    # FiLM 네트워크가 원래 스케일의 roughness 값을 받도록 한다.
    x_mean = X_train[:, :3, :].mean(axis=(0, 2), keepdims=True).astype(np.float32)  # [1,3,1]
    x_std  = (X_train[:, :3, :].std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32)
    y_mean = float(Y_train.mean())
    y_std  = float(Y_train.std() + 1e-8)

    def norm_x(X):
        Xn = X.copy()
        Xn[:, :3, :] = (X[:, :3, :] - x_mean) / x_std
        # ch3 (roughness) 은 0~1 원본 그대로
        return Xn

    def norm_y(Y): return (Y - y_mean) / y_std

    # roughness별 실측 RMS 계산 (정규화 전 Y 사용)
    global ROUGHNESS_TARGET_RMS
    ROUGHNESS_TARGET_RMS = compute_roughness_target_rms(X_train, Y_train)

    train_ds = SeqDatasetWithRoughness(norm_x(X_train), norm_y(Y_train))
    val_ds   = SeqDatasetWithRoughness(norm_x(X_val),   norm_y(Y_val))
    test_ds  = SeqDatasetWithRoughness(norm_x(X_test),  norm_y(Y_test))

    # roughness 는 정규화 전 X 에서 뽑아야 0~100 복원이 정확함
    train_ds.roughness = torch.from_numpy((X_train[:, 3, 0] * 100.0).astype(np.float32))
    val_ds.roughness   = torch.from_numpy((X_val[:,   3, 0] * 100.0).astype(np.float32))
    test_ds.roughness  = torch.from_numpy((X_test[:,  3, 0] * 100.0).astype(np.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n[LOSS WEIGHTS]  rms={args.lambda_rms}  contrast={args.lambda_contrast}  hf={args.lambda_hf}  centroid={args.lambda_centroid}")

    # ── Model & optimiser ────────────────────────────────────────────────────
    model = LiteSeq2SeqCNNGRU_AttnPool(in_ch=3, output_steps=OUTPUT_STEPS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    history = {"train_total": [], "val_total": [],
               "train_rms": [], "train_contrast": [], "train_centroid": [], "train_hf": []}
    best_val, best_state, best_epoch = np.inf, None, -1
    patience_cnt, wait = 12, 0

    print("\n[TRAINING]")
    for epoch in range(1, EPOCHS + 1):
        train_log, _, _ = run_epoch(train_loader, model, optimizer, DEVICE, args, train=True)
        val_log,   _, _ = run_epoch(val_loader,   model, optimizer, DEVICE, args, train=False)

        scheduler.step(val_log["total"])

        history["train_total"].append(train_log["total"])
        history["val_total"].append(val_log["total"])
        history["train_rms"].append(train_log["rms_calib"])
        history["train_contrast"].append(train_log["contrast"])
        history["train_centroid"].append(train_log["centroid"])
        history["train_hf"].append(train_log["hf_ratio"])

        print(f"  [{epoch:03d}/{EPOCHS}] "
              f"train={train_log['total']:.4f}  val={val_log['total']:.4f}  "
              f"rms={train_log['rms_calib']:.4f}  "
              f"contrast={train_log['contrast']:.4f}  "
              f"hf={train_log['hf_ratio']:.4f}")

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

    # ── Test evaluation ───
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

    # roughness별 출력 RMS 확인
    test_roughness = test_ds.roughness.numpy()
    print("\n[TEST roughness별 출력 RMS]")
    rms_rows = {}
    for r in sorted(np.unique(np.round(test_roughness)).astype(int)):
        mask = np.abs(test_roughness - r) < 1
        if mask.sum() == 0:
            continue
        rms_vals = np.sqrt(np.mean(pred_test[mask] ** 2, axis=1))
        pred_rms = float(np.mean(rms_vals))
        target_rms = interpolate_target_rms(r)
        print(f"  r={r:3d} | pred_rms={pred_rms:.4f} (target={target_rms:.4f})")
        rms_rows[str(r)] = {"pred_rms": round(pred_rms, 6), "target_rms": round(target_rms, 6)}
    report["roughness_rms"] = rms_rows

    # ── Save model ───────────────────────────────────────────────────────────
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
        },
    }, PT_PATH)
    print(f"[SAVE] model -> {PT_PATH}")

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Run: {run_id}  |  RMSE={rmse:.4f}  Corr={corr:.4f}", fontsize=11)

    axes[0].plot(history["train_total"], label="train total")
    axes[0].plot(history["val_total"],   label="val total")
    axes[0].axvline(best_epoch - 1, color="red", linestyle="--", alpha=0.5, label=f"best epoch {best_epoch}")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_rms"],      label=f"rms_calib (x{args.lambda_rms})")
    axes[1].plot(history["train_contrast"], label=f"contrast (x{args.lambda_contrast})")
    axes[1].plot(history["train_hf"],       label=f"hf_ratio (x{args.lambda_hf})")
    axes[1].plot(history["train_centroid"], label=f"centroid (x{args.lambda_centroid})")
    axes[1].set_title("Roughness Loss Components")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = run_dir / "training_history.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[SAVE] plot   -> {plot_path}")

    # ── Report 저장 ──────────────────────────────────────────────────────────
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

    # 콘솔 요약 출력
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

    # ── Auto eval ────────────────────────────────────────────────────────────
    if not getattr(args, "no_eval", False):
        print("\n[AUTO EVAL] Running eval_roughness_signal.py ...")
        import subprocess, sys as _sys
        result = subprocess.run(
            [_sys.executable, "-m", "scripts.eval_roughness_signal",
             "--run-id", run_id,
             "--out-dir", str(out_dir)],
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode != 0:
            print("[AUTO EVAL] eval_roughness_signal.py failed (exit code",
                  result.returncode, ")")


# ── Argparse ──────────────────────────────────────────────────────────────────

def build_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                   help="Output root directory for model/cache/report")
    p.add_argument("--npz", type=str, default=None,
                   help="NPZ path (default: out-dir/inference_cache_allinone.npz)")
    p.add_argument("--run-id", type=str, default=None,
                   help="Run ID (auto-generated if omitted, e.g. 20260610-001)")
    p.add_argument("--note", type=str, default="",
                   help="Experiment note saved in report.json")
    p.add_argument("--lambda-rms",      type=float, default=2.0,
                   help="RMS calibration loss weight (0 to disable)")
    p.add_argument("--lambda-contrast", type=float, default=0.6,
                   help="Roughness contrastive loss weight")
    p.add_argument("--lambda-centroid", type=float, default=0.0,
                   help="Spectral centroid loss weight (0 = disabled)")
    p.add_argument("--lambda-hf",       type=float, default=0.5,
                   help="HF energy ratio contrastive loss weight (0 to disable)")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip automatic eval_roughness_signal.py after training")
    return p.parse_args()


if __name__ == "__main__":
    main(build_args())
