"""
scripts/eval_roughness_signal.py
=================================
Roughness별 생성 신호 비교 평가 도구.

고정된 force/velocity 조건에서 roughness만 바꿔가며 모델 출력 신호를 생성하고
파형, FFT 스펙트럼, 지표(RMS, HF 에너지 비율, ZCR, Crest Factor)를 시각화한다.

Usage:
    # 가장 최근 run 자동 사용
    python -m scripts.eval_roughness_signal

    # 특정 run 지정
    python -m scripts.eval_roughness_signal --run-id 20260610-001

    # NPZ 데이터에서 실제 신호와 비교
    python -m scripts.eval_roughness_signal --run-id 20260610-001 --compare-data
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUT_DIR, DEVICE, INPUT_STEPS, OUTPUT_STEPS
from src.model import LiteSeq2SeqCNNGRU_AttnPool
from src.inference import load_model_from_pt


# ── 분석 대상 roughness 레벨 ──────────────────────────────────────────────────
ROUGHNESS_LEVELS = [5, 12, 23, 45, 58, 66, 100]
SAMPLE_RATE = 1000  # Hz (OUTPUT_STEPS=40 samples → 40ms)

# FiLM 모델은 in_ch=3
MODEL_IN_CH = 3


def get_latest_run_dir(out_dir: Path) -> Path:
    runs_dir = out_dir / "runs"
    runs = sorted(runs_dir.glob("*-*"))
    if not runs:
        raise FileNotFoundError(f"runs/ 폴더에 학습된 모델이 없습니다: {runs_dir}")
    return runs[-1]


def load_model(run_dir: Path, device: str):
    pt_path = run_dir / "best_model.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {pt_path}")
    model, x_mean, x_std, y_mean, y_std, _ = load_model_from_pt(
        pt_path, device=device, in_ch=MODEL_IN_CH, output_steps=OUTPUT_STEPS,
    )
    return model, x_mean, x_std, y_mean, y_std


def build_input(roughness_level: int,
                force_val: float, vel_val: float,
                x_mean: np.ndarray, x_std: np.ndarray) -> torch.Tensor:
    """
    고정 force/velocity + 지정 roughness로 모델 입력 윈도우를 만든다.
    ch0=acc(0으로 시작), ch1=force, ch2=vel, ch3=roughness(0~1)

    x_mean / x_std 는 (1, 3, 1) 형태 (ch0~2 만).
    """
    T = INPUT_STEPS
    acc   = np.zeros(T, dtype=np.float32)
    force = np.full(T, force_val, dtype=np.float32)
    vel   = np.full(T, vel_val,   dtype=np.float32)
    r_norm = roughness_level / 100.0

    X = np.stack([acc, force, vel, np.full(T, r_norm)], axis=0)  # (4, T)

    # ch0~2 정규화, ch3 그대로
    Xn = X.copy()
    Xn[:3, :] = (X[:3, :] - x_mean[0, :3, 0:1]) / (x_std[0, :3, 0:1] + 1e-8)

    return torch.tensor(Xn[np.newaxis], dtype=torch.float32)  # (1, 4, T)


def generate_signal(model, x_tensor: torch.Tensor, device: str,
                    y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    """모델로 신호를 생성하고 역정규화해 반환."""
    model.eval()
    with torch.no_grad():
        out = model(x_tensor.to(device)).cpu().numpy()  # (1, output_steps)
    return out[0] * y_std + y_mean  # (output_steps,)


# ── 지표 계산 ─────────────────────────────────────────────────────────────────

def compute_metrics(sig: np.ndarray, fs: int = SAMPLE_RATE) -> dict:
    rms = float(np.sqrt(np.mean(sig ** 2)))
    peak = float(np.max(np.abs(sig))) + 1e-10
    crest = peak / (rms + 1e-10)

    # 영점 교차율
    zcr = float(np.sum(np.diff(np.sign(sig)) != 0) / len(sig))

    # HF 에너지 비율 (>cutoff Hz 성분)
    fft_mag = np.abs(np.fft.rfft(sig))
    freqs   = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    cutoff  = 200  # Hz
    hf_ratio = float(np.sum(fft_mag[freqs >= cutoff] ** 2) /
                     (np.sum(fft_mag ** 2) + 1e-10))

    # 스펙트럼 무게중심
    centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10))

    return dict(rms=rms, crest_factor=crest, zcr=zcr, hf_ratio=hf_ratio,
                centroid_hz=centroid)


# ── 실제 데이터에서 roughness별 평균 신호 추출 ───────────────────────────────

def load_reference_signals(npz_path: Path) -> dict:
    """NPZ에서 roughness별 Y 신호를 가져와 roughness → mean_signal dict 반환."""
    npz = np.load(npz_path, allow_pickle=True)
    X_all = np.concatenate([npz["X_train"], npz["X_val"], npz["X_test"]], axis=0)
    Y_all = np.concatenate([npz["Y_train"], npz["Y_val"], npz["Y_test"]], axis=0)

    ref = {}
    roughness_raw = X_all[:, 3, 0] * 100.0
    for r in ROUGHNESS_LEVELS:
        mask = np.abs(roughness_raw - r) < 1
        if mask.sum() == 0:
            continue
        mean_sig = Y_all[mask].mean(axis=0)
        ref[r] = mean_sig
    return ref


# ── 시각화 ────────────────────────────────────────────────────────────────────

def make_colormap(n: int):
    cmap = plt.colormaps["plasma"]
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_waveforms(signals: dict, run_id: str, save_path: Path):
    """파형을 roughness별로 겹쳐서 그린다."""
    levels = sorted(signals.keys())
    colors = make_colormap(len(levels))
    t = np.arange(OUTPUT_STEPS) / SAMPLE_RATE * 1000  # ms

    fig, ax = plt.subplots(figsize=(10, 4))
    for color, r in zip(colors, levels):
        ax.plot(t, signals[r], color=color, label=f"R={r}", alpha=0.85, lw=1.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title(f"Generated signals by roughness — {run_id}")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → {save_path}")


def plot_fft(signals: dict, run_id: str, save_path: Path):
    """FFT 스펙트럼을 roughness별로 그린다."""
    levels = sorted(signals.keys())
    colors = make_colormap(len(levels))

    fig, ax = plt.subplots(figsize=(10, 4))
    for color, r in zip(colors, levels):
        mag = np.abs(np.fft.rfft(signals[r]))
        freqs = np.fft.rfftfreq(OUTPUT_STEPS, d=1.0 / SAMPLE_RATE)
        ax.plot(freqs, mag, color=color, label=f"R={r}", alpha=0.85, lw=1.5)
    ax.axvline(200, color="gray", ls="--", lw=1, label="200 Hz cutoff")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"FFT spectrum by roughness — {run_id}")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → {save_path}")


def plot_metrics(metrics_by_r: dict, run_id: str, save_path: Path):
    """RMS, HF ratio, ZCR, Crest Factor를 roughness별 막대 그래프로 그린다."""
    levels = sorted(metrics_by_r.keys())
    keys = ["rms", "hf_ratio", "zcr", "crest_factor", "centroid_hz"]
    labels = ["RMS", "HF Energy Ratio (>200Hz)", "Zero-Crossing Rate",
              "Crest Factor", "Spectral Centroid (Hz)"]

    fig, axes = plt.subplots(1, len(keys), figsize=(18, 4))
    colors = make_colormap(len(levels))

    for ax, key, label in zip(axes, keys, labels):
        vals = [metrics_by_r[r][key] for r in levels]
        bars = ax.bar([str(r) for r in levels], vals, color=colors)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Roughness")
        ax.tick_params(axis="x", labelsize=8)
        # 값 레이블
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"Signal metrics by roughness — {run_id}", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → {save_path}")


def plot_combined(signals: dict, ref_signals: dict | None,
                  metrics_by_r: dict, run_id: str, save_path: Path):
    """파형 + FFT + 지표를 한 장에 담은 종합 리포트."""
    levels = sorted(signals.keys())
    colors = make_colormap(len(levels))
    t  = np.arange(OUTPUT_STEPS) / SAMPLE_RATE * 1000
    fs = SAMPLE_RATE

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 파형 ─────────────────────────────────────────────────────────────────
    ax_wave = fig.add_subplot(gs[0, :2])
    for color, r in zip(colors, levels):
        ax_wave.plot(t, signals[r], color=color, label=f"R={r}", alpha=0.85, lw=1.4)
    ax_wave.set_xlabel("Time (ms)"); ax_wave.set_ylabel("Accel (m/s²)")
    ax_wave.set_title("Generated waveforms")
    ax_wave.legend(ncol=4, fontsize=7); ax_wave.grid(True, alpha=0.3)

    # ── 참조 신호 (데이터 평균) ───────────────────────────────────────────────
    if ref_signals:
        ax_ref = fig.add_subplot(gs[0, 2])
        for color, r in zip(colors, levels):
            if r in ref_signals:
                ax_ref.plot(t, ref_signals[r], color=color,
                            label=f"R={r}", alpha=0.85, lw=1.4)
        ax_ref.set_xlabel("Time (ms)"); ax_ref.set_ylabel("Accel (m/s²)")
        ax_ref.set_title("Reference (data mean)")
        ax_ref.legend(ncol=2, fontsize=7); ax_ref.grid(True, alpha=0.3)

    # ── FFT ──────────────────────────────────────────────────────────────────
    ax_fft = fig.add_subplot(gs[1, :2])
    for color, r in zip(colors, levels):
        mag   = np.abs(np.fft.rfft(signals[r]))
        freqs = np.fft.rfftfreq(OUTPUT_STEPS, d=1.0 / fs)
        ax_fft.plot(freqs, mag, color=color, label=f"R={r}", alpha=0.85, lw=1.4)
    ax_fft.axvline(200, color="gray", ls="--", lw=1, label="200Hz")
    ax_fft.set_xlabel("Freq (Hz)"); ax_fft.set_ylabel("Magnitude")
    ax_fft.set_title("FFT spectrum")
    ax_fft.legend(ncol=4, fontsize=7); ax_fft.grid(True, alpha=0.3)

    # ── RMS & HF 비율 라인 ───────────────────────────────────────────────────
    ax_rms = fig.add_subplot(gs[1, 2])
    rms_vals    = [metrics_by_r[r]["rms"]      for r in levels]
    hf_vals     = [metrics_by_r[r]["hf_ratio"] for r in levels]
    ax_rms.plot(levels, rms_vals,  "o-", color="steelblue",  label="RMS")
    ax_rms2 = ax_rms.twinx()
    ax_rms2.plot(levels, hf_vals,  "s--", color="tomato", label="HF ratio")
    ax_rms.set_xlabel("Roughness"); ax_rms.set_ylabel("RMS", color="steelblue")
    ax_rms2.set_ylabel("HF ratio (>200Hz)", color="tomato")
    ax_rms.set_title("RMS & HF ratio vs Roughness")
    ax_rms.grid(True, alpha=0.3)
    lines1, labels1 = ax_rms.get_legend_handles_labels()
    lines2, labels2 = ax_rms2.get_legend_handles_labels()
    ax_rms.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # ── ZCR & Crest ───────────────────────────────────────────────────────────
    ax_zcr = fig.add_subplot(gs[2, 0])
    zcr_vals = [metrics_by_r[r]["zcr"] for r in levels]
    ax_zcr.bar([str(r) for r in levels], zcr_vals,
               color=[c for c in colors])
    for i, v in enumerate(zcr_vals):
        ax_zcr.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax_zcr.set_title("Zero-Crossing Rate"); ax_zcr.set_xlabel("Roughness")

    ax_cr = fig.add_subplot(gs[2, 1])
    cr_vals = [metrics_by_r[r]["crest_factor"] for r in levels]
    ax_cr.bar([str(r) for r in levels], cr_vals,
              color=[c for c in colors])
    for i, v in enumerate(cr_vals):
        ax_cr.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax_cr.set_title("Crest Factor"); ax_cr.set_xlabel("Roughness")

    ax_cen = fig.add_subplot(gs[2, 2])
    cen_vals = [metrics_by_r[r]["centroid_hz"] for r in levels]
    ax_cen.bar([str(r) for r in levels], cen_vals,
               color=[c for c in colors])
    for i, v in enumerate(cen_vals):
        ax_cen.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax_cen.set_title("Spectral Centroid (Hz)"); ax_cen.set_xlabel("Roughness")

    fig.suptitle(
        f"Roughness signal evaluation — Run {run_id}\n"
        f"(fixed force/velocity, roughness varied)",
        fontsize=12,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


# ── CSV 저장 ──────────────────────────────────────────────────────────────────

def save_csv(metrics_by_r: dict, save_path: Path):
    import csv
    keys = ["rms", "hf_ratio", "zcr", "crest_factor", "centroid_hz"]
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["roughness"] + keys)
        for r in sorted(metrics_by_r.keys()):
            writer.writerow([r] + [f"{metrics_by_r[r][k]:.6f}" for k in keys])
    print(f"  → {save_path}")


# ── 데이터에서 force/vel 대표값 추출 ─────────────────────────────────────────

def get_representative_force_vel(npz_path: Path) -> tuple[float, float]:
    """train 데이터의 median force, velocity를 반환한다."""
    npz = np.load(npz_path, allow_pickle=True)
    X = npz["X_train"]
    force_med = float(np.median(X[:, 1, -1]))  # ch1: force (마지막 타임스텝)
    vel_med   = float(np.median(X[:, 2, -1]))  # ch2: velocity
    return force_med, vel_med


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Roughness별 생성 신호 평가")
    parser.add_argument("--run-id", type=str, default=None,
                        help="평가할 run ID (예: 20260610-001). 생략 시 최신 run 사용.")
    parser.add_argument("--force",  type=float, default=None,
                        help="고정 force 값 (기본: 데이터 median)")
    parser.add_argument("--vel",    type=float, default=None,
                        help="고정 velocity 값 (기본: 데이터 median)")
    parser.add_argument("--compare-data", action="store_true",
                        help="NPZ 데이터의 평균 신호와 비교 플롯 추가")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"출력 루트 디렉토리 (기본: {OUT_DIR})")
    args = parser.parse_args()

    out_dir = args.out_dir

    # ── run 디렉토리 결정 ─────────────────────────────────────────────────────
    if args.run_id:
        run_dir = out_dir / "runs" / args.run_id
        run_id  = args.run_id
    else:
        run_dir = get_latest_run_dir(out_dir)
        run_id  = run_dir.name
    print(f"[eval] run: {run_id}  ({run_dir})")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    model, x_mean, x_std, y_mean, y_std = load_model(run_dir, DEVICE)
    print(f"[eval] 모델 로드 완료  (in_ch={MODEL_IN_CH}, output_steps={OUTPUT_STEPS})")

    # x_mean/x_std 형태 보정 (load_model_from_pt는 다양한 shape 반환 가능)
    if x_mean.ndim == 1:
        x_mean = x_mean[np.newaxis, :, np.newaxis]
        x_std  = x_std [np.newaxis, :, np.newaxis]
    elif x_mean.ndim == 2:
        x_mean = x_mean[np.newaxis]
        x_std  = x_std [np.newaxis]

    # ── force/velocity 대표값 ─────────────────────────────────────────────────
    npz_path = out_dir / "inference_cache_allinone.npz"
    if args.force is None or args.vel is None:
        if npz_path.exists():
            force_med, vel_med = get_representative_force_vel(npz_path)
        else:
            force_med, vel_med = 0.0, 0.0
            print("[eval] NPZ 없음 → force=0, vel=0 사용")
    force_val = args.force if args.force is not None else force_med
    vel_val   = args.vel   if args.vel   is not None else vel_med
    print(f"[eval] 고정 조건: force={force_val:.4f}, vel={vel_val:.4f}")

    # ── 신호 생성 ─────────────────────────────────────────────────────────────
    signals = {}
    metrics_by_r = {}
    print("[eval] roughness별 신호 생성...")
    for r in ROUGHNESS_LEVELS:
        x_t = build_input(r, force_val, vel_val, x_mean, x_std)
        sig = generate_signal(model, x_t, DEVICE, y_mean, y_std)
        signals[r] = sig
        metrics_by_r[r] = compute_metrics(sig)
        m = metrics_by_r[r]
        print(f"  R={r:3d}: RMS={m['rms']:.4f}  HF={m['hf_ratio']:.3f}"
              f"  ZCR={m['zcr']:.3f}  Crest={m['crest_factor']:.2f}"
              f"  Centroid={m['centroid_hz']:.1f}Hz")

    # ── 참조 신호 (선택) ──────────────────────────────────────────────────────
    ref_signals = None
    if args.compare_data and npz_path.exists():
        print("[eval] 참조 신호 로드 중...")
        ref_signals = load_reference_signals(npz_path)

    # ── 플롯 저장 ─────────────────────────────────────────────────────────────
    eval_dir = run_dir / "roughness_eval"
    eval_dir.mkdir(exist_ok=True)
    print(f"\n[eval] 저장 디렉토리: {eval_dir}")

    plot_waveforms(signals, run_id, eval_dir / "waveforms.png")
    plot_fft(signals, run_id, eval_dir / "fft_spectrum.png")
    plot_metrics(metrics_by_r, run_id, eval_dir / "metrics_bar.png")
    plot_combined(signals, ref_signals, metrics_by_r, run_id,
                  eval_dir / "roughness_eval_combined.png")
    save_csv(metrics_by_r, eval_dir / "metrics.csv")

    print(f"\n[eval] 완료! → {eval_dir}/roughness_eval_combined.png")


if __name__ == "__main__":
    main()
