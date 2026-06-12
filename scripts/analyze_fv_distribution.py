"""
scripts/analyze_fv_distribution.py
====================================
학습 데이터의 Force / Velocity 분포를 거칠기별 박스플롯으로 시각화.

실행:
  python -m scripts.analyze_fv_distribution
  python -m scripts.analyze_fv_distribution --npz pt_files/inference_cache_allinone.npz
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROUGHNESS_LIST = [5, 12, 23, 45, 58, 66, 100]
DEFAULT_NPZ    = Path("pt_files/inference_cache_allinone.npz")


def load_fv(npz_path):
    npz   = np.load(npz_path, allow_pickle=True)
    X_all = np.concatenate([npz["X_train"], npz["X_val"], npz["X_test"]], axis=0)
    roughness = X_all[:, 3, 0] * 100.0
    force     = X_all[:, 1, :].mean(axis=1)   # 윈도우 평균
    vel       = X_all[:, 2, :].mean(axis=1)
    return roughness, force, vel


def boxplot_stats(data):
    """matplotlib boxplot용 stats dict 직접 계산"""
    p10, p25, p50, p75, p90 = np.percentile(data, [10, 25, 50, 75, 90])
    return dict(med=p50, q1=p25, q3=p75, whislo=p10, whishi=p90,
                mean=data.mean(), fliers=[])


def make_plot(roughness, force, vel, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training data  —  Force & Velocity distribution per roughness",
                 fontsize=12, y=1.01)

    xlabels = [f"R={r}" for r in ROUGHNESS_LIST]
    positions = list(range(1, len(ROUGHNESS_LIST) + 1))

    for ax, signal, title, unit, ylim, p99_val, color in [
        (axes[0], force, "Force", "N",     (0, 7),   8.31,  "#378ADD"),
        (axes[1], vel,   "Velocity", "m/s", (0, 0.20), 0.164, "#D85A30"),
    ]:
        stats_list = []
        for r in ROUGHNESS_LIST:
            mask = np.abs(roughness - r) < 5
            stats_list.append(boxplot_stats(signal[mask]))

        bp = ax.bxp(
            stats_list,
            positions=positions,
            widths=0.5,
            showfliers=False,
            showmeans=True,
            patch_artist=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor=color, markersize=5),
            medianprops=dict(color=color, linewidth=2),
            boxprops=dict(facecolor=color + "33", edgecolor=color, linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1, linestyle="--"),
            capprops=dict(color=color, linewidth=1.5),
        )

        # p99 기준선
        ax.axhline(p99_val, color="#E24B4A", linewidth=1.2, linestyle="--", alpha=0.85,
                   label=f"p99 = {p99_val:.3f} {unit}  (training range limit)")

        ax.set_xticks(positions)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_ylabel(f"{title} ({unit})", fontsize=10)
        ax.set_title(f"{title} distribution", fontsize=10)
        ax.grid(True, axis="y", lw=0.4, alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

        # 중앙값 텍스트
        for pos, st in zip(positions, stats_list):
            ax.text(pos, st["med"] + ylim[1] * 0.015,
                    f"{st['med']:.3f}", ha="center", va="bottom",
                    fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {save_path}")


def main(args):
    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"[ERROR] NPZ not found: {npz_path}"); return

    roughness, force, vel = load_fv(npz_path)
    print(f"[DATA] total windows: {len(force)}")

    save_path = Path(args.out)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    make_plot(roughness, force, vel, save_path)


def _build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--out", default="pt_files/fv_distribution.png")
    return p.parse_args()


if __name__ == "__main__":
    main(_build_args())
