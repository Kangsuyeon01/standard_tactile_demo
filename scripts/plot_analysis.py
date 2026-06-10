"""
scripts/plot_analysis.py
========================
Generate analysis plots:
  1. LOW / MID / HIGH force-velocity + FFT for every roughness level
  2. Generated vs Actual comparison for every roughness level

Usage:
    python -m scripts.plot_analysis
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    OUT_DIR, DEVICE, OUTPUT_STEPS,
    ACC_LIM, FORCE_LIM, VEL_LIM,
)
from src.cache import activate_cache
from src.inference import load_model_from_pt
from src.visualise import plot_low_mid_high, plot_generated_vs_actual


PT_PATH    = OUT_DIR / "best_model_light_wo71.pt"
CACHE_PATH = OUT_DIR / "inference_cache_allinone.npz"


def main():
    seg_meta_df = activate_cache(CACHE_PATH)

    model, x_mean, x_std, y_mean, y_std, _ = load_model_from_pt(
        PT_PATH, device=DEVICE, in_ch=4, output_steps=OUTPUT_STEPS,
    )
    if x_mean.ndim == 1: x_mean = x_mean[:, None]
    if x_std.ndim == 1:  x_std  = x_std[:, None]

    roughness_list = sorted(seg_meta_df["roughness"].dropna().unique().tolist())

    # ── LOW / MID / HIGH (actual only) ──────────────────────────────────────
    print("\n=== LOW / MID / HIGH plots ===")
    for r in roughness_list:
        plot_low_mid_high(
            roughness=r,
            seg_meta_df=seg_meta_df,
            out_root=OUT_DIR / "force_velocity_low_mid_high",
            save=True, show=False,
            y_lim_acc=ACC_LIM,
            y_lim_force=FORCE_LIM,
            y_lim_vel=VEL_LIM,
            y_lim_fft=(0, 1000),
        )

    # ── Generated vs Actual ──────────────────────────────────────────────────
    print("\n=== Generated signal plots ===")
    for r in roughness_list:
        plot_generated_vs_actual(
            roughness=r,
            seg_meta_df=seg_meta_df,
            model=model,
            x_mean=x_mean, x_std=x_std,
            y_mean=y_mean, y_std=y_std,
            device=DEVICE,
            out_dir=OUT_DIR / "generated_plots",
            save=True, show=False,
            y_lim_acc=ACC_LIM,
            y_lim_force=FORCE_LIM,
            y_lim_vel=VEL_LIM,
            y_lim_fft=(0, 1000),
        )

    print(f"\n[DONE] Plots saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
