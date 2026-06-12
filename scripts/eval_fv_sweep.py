"""
scripts/eval_fv_sweep.py
========================
힘(force)과 속도(velocity)를 변화시킬 때 모델 출력이 어떻게 달라지는지 확인.

교수님 제안 방식:
  힘/속도를 극단값/중간값/최솟값으로 바꾸면서 Temporal + Spectral 플롯 생성.

Usage:
    python -m scripts.eval_fv_sweep
    python -m scripts.eval_fv_sweep --run-id 20260610-004
    python -m scripts.eval_fv_sweep --run-id 20260610-004 --roughness 45
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUT_DIR, DEVICE, INPUT_STEPS, OUTPUT_STEPS
from src.inference import load_model_from_pt

MODEL_IN_CH = 3
SAMPLE_RATE = 1000   # OUTPUT_STEPS=40 -> 40ms window

# sweep 구간 (min / 25% / 50% / 75% / max)
FORCE_LEVELS = [0.0, 0.5, 1.0, 2.0, 4.0]   # N
VEL_LEVELS   = [0.0, 0.02, 0.05, 0.10, 0.20] # m/s


def get_latest_run_dir(out_dir: Path) -> Path:
    runs = sorted((out_dir / "runs").glob("*-*"))
    if not runs:
        raise FileNotFoundError(f"runs/ 없음: {out_dir / 'runs'}")
    return runs[-1]


def build_input(force_val, vel_val, roughness_norm, x_mean, x_std):
    T = INPUT_STEPS
    acc   = np.zeros(T, dtype=np.float32)
    force = np.full(T, force_val, dtype=np.float32)
    vel   = np.full(T, vel_val,   dtype=np.float32)
    X = np.stack([acc, force, vel, np.full(T, roughness_norm)], axis=0)
    Xn = X.copy()
    Xn[:3] = (X[:3] - x_mean[0, :3, 0:1]) / (x_std[0, :3, 0:1] + 1e-8)
    return torch.tensor(Xn[np.newaxis], dtype=torch.float32)


def infer(model, x_t, device, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        out = model(x_t.to(device)).cpu().numpy()[0]
    return out * y_std + y_mean


def compute_rms(sig):
    return float(np.sqrt(np.mean(sig ** 2)))


def compute_hf(sig, fs=SAMPLE_RATE, cutoff=200):
    mag = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    return float(np.sum(mag[freqs >= cutoff] ** 2) / (np.sum(mag ** 2) + 1e-10))


# ── Force sweep plot ───────────────────────────────────────────────────────────

def plot_force_sweep(model, x_mean, x_std, y_mean, y_std, device,
                     roughness, vel_fixed, out_dir):
    t_ms = np.arange(OUTPUT_STEPS) / SAMPLE_RATE * 1000
    freqs = np.fft.rfftfreq(OUTPUT_STEPS, 1.0 / SAMPLE_RATE)
    r_norm = roughness / 100.0

    signals = []
    for f in FORCE_LEVELS:
        xt = build_input(f, vel_fixed, r_norm, x_mean, x_std)
        signals.append(infer(model, xt, device, y_mean, y_std))

    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(FORCE_LEVELS)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Force sweep  (R={roughness}, vel={vel_fixed:.3f} m/s)", fontsize=12)

    ax0, ax1, ax2 = axes
    rms_list, hf_list = [], []
    for i, (sig, f) in enumerate(zip(signals, FORCE_LEVELS)):
        label = f"F={f:.1f}N"
        ax0.plot(t_ms, sig, color=cmap[i], label=label)
        mag = np.abs(np.fft.rfft(sig))
        ax1.plot(freqs, mag, color=cmap[i], label=label)
        rms_list.append(compute_rms(sig))
        hf_list.append(compute_hf(sig))

    ax0.set_xlabel("Time (ms)"); ax0.set_ylabel("Accel"); ax0.legend(fontsize=7)
    ax0.set_title("Waveform")
    ax1.set_xlabel("Freq (Hz)"); ax1.set_ylabel("Magnitude"); ax1.legend(fontsize=7)
    ax1.set_title("FFT spectrum")

    ax2.plot(FORCE_LEVELS, rms_list, "o-b", label="RMS")
    ax2r = ax2.twinx()
    ax2r.plot(FORCE_LEVELS, hf_list, "s--r", label="HF ratio")
    ax2.set_xlabel("Force (N)"); ax2.set_ylabel("RMS", color="b")
    ax2r.set_ylabel("HF ratio (>200Hz)", color="r")
    ax2.set_title("RMS & HF ratio vs Force")
    lines1, _ = ax2.get_legend_handles_labels()
    lines2, _ = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, ["RMS", "HF ratio"], fontsize=8)

    fig.tight_layout()
    path = out_dir / "fv_sweep_force.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")

    # 수치 출력
    print(f"\n  [Force sweep]  R={roughness}  vel={vel_fixed:.3f}")
    print(f"  {'Force':>8}  {'RMS':>8}  {'HF ratio':>10}")
    for f, r, h in zip(FORCE_LEVELS, rms_list, hf_list):
        print(f"  {f:>8.2f}  {r:>8.4f}  {h:>10.4f}")


# ── Velocity sweep plot ────────────────────────────────────────────────────────

def plot_vel_sweep(model, x_mean, x_std, y_mean, y_std, device,
                   roughness, force_fixed, out_dir):
    t_ms = np.arange(OUTPUT_STEPS) / SAMPLE_RATE * 1000
    freqs = np.fft.rfftfreq(OUTPUT_STEPS, 1.0 / SAMPLE_RATE)
    r_norm = roughness / 100.0

    signals = []
    for v in VEL_LEVELS:
        xt = build_input(force_fixed, v, r_norm, x_mean, x_std)
        signals.append(infer(model, xt, device, y_mean, y_std))

    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(VEL_LEVELS)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Velocity sweep  (R={roughness}, force={force_fixed:.2f} N)", fontsize=12)

    ax0, ax1, ax2 = axes
    rms_list, hf_list = [], []
    for i, (sig, v) in enumerate(zip(signals, VEL_LEVELS)):
        label = f"v={v:.2f}"
        ax0.plot(t_ms, sig, color=cmap[i], label=label)
        mag = np.abs(np.fft.rfft(sig))
        ax1.plot(freqs, mag, color=cmap[i], label=label)
        rms_list.append(compute_rms(sig))
        hf_list.append(compute_hf(sig))

    ax0.set_xlabel("Time (ms)"); ax0.set_ylabel("Accel"); ax0.legend(fontsize=7)
    ax0.set_title("Waveform")
    ax1.set_xlabel("Freq (Hz)"); ax1.set_ylabel("Magnitude"); ax1.legend(fontsize=7)
    ax1.set_title("FFT spectrum")

    ax2.plot(VEL_LEVELS, rms_list, "o-b", label="RMS")
    ax2r = ax2.twinx()
    ax2r.plot(VEL_LEVELS, hf_list, "s--r", label="HF ratio")
    ax2.set_xlabel("Velocity (m/s)"); ax2.set_ylabel("RMS", color="b")
    ax2r.set_ylabel("HF ratio (>200Hz)", color="r")
    ax2.set_title("RMS & HF ratio vs Velocity")
    lines1, _ = ax2.get_legend_handles_labels()
    lines2, _ = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, ["RMS", "HF ratio"], fontsize=8)

    fig.tight_layout()
    path = out_dir / "fv_sweep_vel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")

    print(f"\n  [Vel sweep]  R={roughness}  force={force_fixed:.2f}")
    print(f"  {'Vel':>8}  {'RMS':>8}  {'HF ratio':>10}")
    for v, r, h in zip(VEL_LEVELS, rms_list, hf_list):
        print(f"  {v:>8.3f}  {r:>8.4f}  {h:>10.4f}")


# ── 2D heatmap (force x velocity) ─────────────────────────────────────────────

def plot_fv_heatmap(model, x_mean, x_std, y_mean, y_std, device,
                    roughness, out_dir):
    r_norm = roughness / 100.0
    rms_grid = np.zeros((len(VEL_LEVELS), len(FORCE_LEVELS)))

    for vi, v in enumerate(VEL_LEVELS):
        for fi, f in enumerate(FORCE_LEVELS):
            xt = build_input(f, v, r_norm, x_mean, x_std)
            sig = infer(model, xt, device, y_mean, y_std)
            rms_grid[vi, fi] = compute_rms(sig)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(rms_grid, aspect="auto", origin="lower",
                   extent=[FORCE_LEVELS[0]-0.1, FORCE_LEVELS[-1]+0.1,
                            VEL_LEVELS[0]-0.005, VEL_LEVELS[-1]+0.005],
                   cmap="hot")
    plt.colorbar(im, ax=ax, label="RMS")
    ax.set_xlabel("Force (N)"); ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Output RMS heatmap  (R={roughness})\n"
                 f"Goal: top-right (high F, high V) = bright")
    fig.tight_layout()
    path = out_dir / "fv_heatmap_rms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-id",    default=None,
                   help="Run ID (e.g. 20260610-004). 생략 시 최신 run 사용.")
    p.add_argument("--out-dir",   default=str(OUT_DIR))
    p.add_argument("--roughness", type=int, default=45,
                   help="고정 roughness 값 (sweep 시 사용)")
    p.add_argument("--force-fixed",  type=float, default=1.0,
                   help="vel sweep 시 고정 force")
    p.add_argument("--vel-fixed",    type=float, default=0.05,
                   help="force sweep 시 고정 velocity")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if args.run_id:
        run_dir = out_dir / "runs" / args.run_id
    else:
        run_dir = get_latest_run_dir(out_dir)

    print(f"[RUN] {run_dir.name}")
    device = DEVICE

    pt_path = run_dir / "best_model.pt"
    model, x_mean, x_std, y_mean, y_std, _ = load_model_from_pt(
        pt_path, device=device, in_ch=MODEL_IN_CH, output_steps=OUTPUT_STEPS,
    )

    save_dir = run_dir / "fv_sweep"
    save_dir.mkdir(exist_ok=True)

    print("\n[1] Force sweep ...")
    plot_force_sweep(model, x_mean, x_std, y_mean, y_std, device,
                     args.roughness, args.vel_fixed, save_dir)

    print("\n[2] Velocity sweep ...")
    plot_vel_sweep(model, x_mean, x_std, y_mean, y_std, device,
                   args.roughness, args.force_fixed, save_dir)

    print("\n[3] F x V heatmap ...")
    plot_fv_heatmap(model, x_mean, x_std, y_mean, y_std, device,
                    args.roughness, save_dir)

    print(f"\n[DONE] 결과: {save_dir}")
    print("  fv_sweep_force.png  — force 변화에 따른 파형/FFT/RMS")
    print("  fv_sweep_vel.png    — velocity 변화에 따른 파형/FFT/RMS")
    print("  fv_heatmap_rms.png  — F x V 격자에서 RMS 히트맵")
    print("\n  판정 기준:")
    print("  - force/velocity 증가 -> RMS 단조증가  -> 방향성 OK")
    print("  - F=0, V=0 에서 RMS ~ 0                -> zero-output OK")


if __name__ == "__main__":
    main()
