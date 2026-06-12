"""
scripts/analyze_velocity_spectrum.py
=====================================
Velocity별 FFT 스펙트럼 분석.

원본 측정 데이터(NPZ)에서 velocity bin별로 평균 스펙트럼을 계산하여
velocity → spectral 관계를 파악합니다.

출력:
  pt_files/analysis/velocity_spectrum/
    spectrum_per_velocity.png      — roughness별 velocity bin 스펙트럼
    metrics_per_velocity.png       — centroid / HF ratio vs velocity
    heatmap_rms_vel_roughness.png  — RMS heatmap (velocity × roughness)
    velocity_metrics.csv           — 수치 요약

사용법:
  python -m scripts.analyze_velocity_spectrum
  python -m scripts.analyze_velocity_spectrum --npz pt_files/inference_cache_allinone.npz
  python -m scripts.analyze_velocity_spectrum --min-force 0.3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE   = 8000
INPUT_STEPS   = 400
OUTPUT_STEPS  = 40
DEFAULT_NPZ   = Path("pt_files/inference_cache_allinone.npz")
ROUGHNESS_ALL = [5, 12, 23, 45, 58, 66, 100]

# velocity bins: (label, v_min, v_max)
VEL_BINS = [
    ("V<0.01",   0.000, 0.010),
    ("0.01-0.03", 0.010, 0.030),
    ("0.03-0.06", 0.030, 0.060),
    ("0.06-0.10", 0.060, 0.100),
    ("0.10-0.15", 0.100, 0.150),
    ("V>0.15",   0.150, 1.000),
]
VEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ── helpers ───────────────────────────────────────────────────────────────────
def compute_metrics(signal):
    """centroid_hz, hf_ratio, rms for a 1-D signal."""
    x = np.asarray(signal, dtype=np.float32).ravel()
    r = float(np.sqrt(np.mean(x ** 2)))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / SAMPLE_RATE)
    mag   = np.abs(np.fft.rfft(x))
    mag_sum = mag.sum() + 1e-8
    centroid = float(np.sum(freqs * mag) / mag_sum)
    hf_mask  = freqs > 200
    hf_ratio = float(mag[hf_mask].sum() / mag_sum)
    return r, centroid, hf_ratio


def mean_spectrum(signals, n_fft=None):
    """Average FFT magnitude over a list of signals (all same length)."""
    if not signals:
        return None, None
    n = n_fft or len(signals[0])
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    stack = np.stack([np.abs(np.fft.rfft(s[:n], n=n)) for s in signals], axis=0)
    return freqs, stack.mean(axis=0)


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    npz_path = Path(args.npz) if args.npz else DEFAULT_NPZ
    if not npz_path.exists():
        print(f"[ERROR] NPZ not found: {npz_path}"); sys.exit(1)

    out_dir = Path(args.out_dir) / "analysis" / "velocity_spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[NPZ]  {npz_path}")
    npz  = np.load(npz_path, allow_pickle=True)
    cols = [str(c) for c in npz["window_meta_columns"].tolist()]
    wm   = pd.DataFrame({c: npz[f"window_meta__{c}"] for c in cols})

    # merge all splits
    X_parts, Y_parts = [], []
    offset = 0
    wm["arr_idx"] = -1
    for sp in ["train", "val", "test"]:
        mask = wm["split"] == sp
        n = mask.sum()
        if n == 0:
            continue
        wm.loc[mask, "arr_idx"] = np.arange(n) + offset
        X_parts.append(npz[f"X_{sp}"])
        Y_parts.append(npz[f"Y_{sp}"])
        offset += n

    X_all = np.concatenate(X_parts, axis=0)   # (N, ch, INPUT_STEPS)
    Y_all = np.concatenate(Y_parts, axis=0)   # (N, OUTPUT_STEPS)
    print(f"[DATA] X={X_all.shape}  Y={Y_all.shape}  windows={len(wm)}")

    # channel indices: X[:,0,:]=acc  X[:,1,:]=force  X[:,2,:]=vel
    # mean force and mean velocity per window
    wm["_force_mean"] = np.abs(X_all[:, 1, :]).mean(axis=1)
    wm["_vel_mean"]   = np.abs(X_all[:, 2, :]).mean(axis=1)
    wm["_arr_idx"]    = wm["arr_idx"]

    # filter: only active-contact windows (force > min_force)
    active = wm[wm["_force_mean"] >= args.min_force].copy()
    print(f"[FILTER] min_force={args.min_force} N → {len(active)} windows")

    # ── 1. collect signals per (roughness, vel_bin) ──────────────────────────
    records = []   # {roughness, vel_bin_label, rms, centroid, hf_ratio}
    spectra = {}   # (roughness, vel_bin_label) → list of Y signals

    for _, row in active.iterrows():
        r   = int(row["roughness"])
        vi  = int(row["_arr_idx"])
        vm  = float(row["_vel_mean"])
        sig = Y_all[vi].astype(np.float32)

        # find velocity bin
        bin_label = None
        for lbl, vmin, vmax in VEL_BINS:
            if vmin <= vm < vmax:
                bin_label = lbl
                break
        if bin_label is None:
            continue

        rms_v, cent, hf = compute_metrics(sig)
        records.append(dict(roughness=r, vel_bin=bin_label,
                            rms=rms_v, centroid_hz=cent, hf_ratio=hf))
        key = (r, bin_label)
        spectra.setdefault(key, []).append(sig)

    df = pd.DataFrame(records)
    print(f"[RECORDS] {len(df)} entries, "
          f"roughness={sorted(df['roughness'].unique())}")

    # summary per (roughness, vel_bin)
    summary = df.groupby(["roughness", "vel_bin"])[["rms","centroid_hz","hf_ratio"]].mean()
    summary_csv = out_dir / "velocity_metrics.csv"
    summary.to_csv(summary_csv)
    print(f"[CSV]  {summary_csv}")

    # ── 2. spectrum_per_velocity.png ─────────────────────────────────────────
    n_rough = len(ROUGHNESS_ALL)
    fig, axes = plt.subplots(n_rough, 1, figsize=(12, 3.5 * n_rough), sharex=True)
    fig.suptitle("FFT spectrum by velocity bin — original signals", fontsize=12)

    for ri, r in enumerate(ROUGHNESS_ALL):
        ax = axes[ri]
        ax.set_title(f"R={r}", fontsize=9)
        for ci, (lbl, *_) in enumerate(VEL_BINS):
            key = (r, lbl)
            if key not in spectra or len(spectra[key]) == 0:
                continue
            sigs = spectra[key]
            freqs, mag = mean_spectrum(sigs, n_fft=OUTPUT_STEPS)
            if freqs is None:
                continue
            ax.plot(freqs, mag, color=VEL_COLORS[ci], lw=1.2,
                    label=f"{lbl} (n={len(sigs)})", alpha=0.85)
        ax.axvline(200, lw=0.7, ls="--", color="gray", alpha=0.5)
        ax.set_ylabel("Magnitude", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.3, alpha=0.4)
        if ri == 0:
            ax.legend(loc="upper right", fontsize=6, ncol=2)

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=8)
    axes[-1].set_xlim(0, SAMPLE_RATE // 2)
    plt.tight_layout()
    out1 = out_dir / "spectrum_per_velocity.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out1}")

    # ── 3. metrics_per_velocity.png ──────────────────────────────────────────
    bin_order = [b[0] for b in VEL_BINS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Signal metrics vs velocity bin — original signals", fontsize=11)
    metric_names = ["rms", "centroid_hz", "hf_ratio"]
    metric_labels = ["RMS", "Spectral Centroid (Hz)", "HF Energy Ratio (>200 Hz)"]
    colors_r = plt.cm.plasma(np.linspace(0.1, 0.9, len(ROUGHNESS_ALL)))

    for ai, (mn, ml) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[ai]
        for ri, r in enumerate(ROUGHNESS_ALL):
            sub = summary.xs(r, level="roughness") if r in summary.index.get_level_values("roughness") else None
            if sub is None or len(sub) == 0:
                continue
            vals = [sub.loc[b, mn] if b in sub.index else np.nan for b in bin_order]
            ax.plot(range(len(bin_order)), vals,
                    marker="o", ms=5, lw=1.5,
                    color=colors_r[ri], label=f"R={r}")
        ax.set_xticks(range(len(bin_order)))
        ax.set_xticklabels(bin_order, rotation=30, ha="right", fontsize=7)
        ax.set_title(ml, fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=7)
        if ai == 0:
            ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    out2 = out_dir / "metrics_per_velocity.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out2}")

    # ── 4. heatmap RMS (velocity × roughness) ────────────────────────────────
    rms_grid = np.full((len(VEL_BINS), len(ROUGHNESS_ALL)), np.nan)
    for ri, r in enumerate(ROUGHNESS_ALL):
        for bi, (lbl, *_) in enumerate(VEL_BINS):
            try:
                rms_grid[bi, ri] = summary.loc[(r, lbl), "rms"]
            except KeyError:
                pass

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(rms_grid, aspect="auto", cmap="YlOrRd",
                   vmin=np.nanmin(rms_grid) * 0.9,
                   vmax=np.nanmax(rms_grid) * 1.05)
    ax.set_xticks(range(len(ROUGHNESS_ALL)))
    ax.set_xticklabels([f"R={r}" for r in ROUGHNESS_ALL])
    ax.set_yticks(range(len(VEL_BINS)))
    ax.set_yticklabels([b[0] for b in VEL_BINS])
    ax.set_xlabel("Roughness")
    ax.set_ylabel("Velocity bin")
    ax.set_title("Mean RMS — original signals  (velocity × roughness)")
    for bi in range(len(VEL_BINS)):
        for ri in range(len(ROUGHNESS_ALL)):
            v = rms_grid[bi, ri]
            if not np.isnan(v):
                ax.text(ri, bi, f"{v:.3f}", ha="center", va="center",
                        fontsize=7.5, color="black")
    plt.colorbar(im, ax=ax, label="RMS")
    plt.tight_layout()
    out3 = out_dir / "heatmap_rms_vel_roughness.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out3}")

    print(f"\n[DONE] → {out_dir}")

    # ── quick print of centroid & HF ratio summary ───────────────────────────
    print("\n── Spectral Centroid (Hz) by roughness × velocity ──")
    cent_grid = summary["centroid_hz"].unstack(level="roughness")
    print(cent_grid.to_string())
    print("\n── HF Ratio (>200 Hz) by roughness × velocity ──")
    hf_grid = summary["hf_ratio"].unstack(level="roughness")
    print(hf_grid.to_string())


def _build_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--npz",       default=None)
    p.add_argument("--out-dir",   default="pt_files")
    p.add_argument("--min-force", type=float, default=0.2,
                   help="Minimum mean force (N) to include a window")
    return p.parse_args()


if __name__ == "__main__":
    main(_build_args())
