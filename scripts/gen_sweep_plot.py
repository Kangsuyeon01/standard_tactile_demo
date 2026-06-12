"""
gen_sweep_plot.py
생성형 모델로 (velocity × force) 스윕 플롯을 roughness별로 생성.
각 roughness 값마다 PNG 한 장: 행=velocity step, 열=force step

Usage:
    python -m scripts.gen_sweep_plot \
        --pt-path pt_files/runs/20260612-003/best_model.pt \
        --cache-path pt_files/inference_cache_allinone.npz
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from scripts.realtime import (
    load_model_from_pt, InferenceCache,
    RealtimeReferenceGuidedGenerator,
    enhance_acc_by_roughness, apply_common_output_limit,
    force_velocity_gate, roughness_to_target_rms,
)

SEG_LEN     = 4000
SR          = 8000
CHUNK       = 40
INPUT_STEPS = 400

def generate_segment(model, x_mean, x_std, y_mean, y_std,
                     guide_acc, roughness, force, velocity,
                     ref_blend=0.20, device="cpu"):
    rt = RealtimeReferenceGuidedGenerator(
        model=model, x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
        roughness=roughness, guide_acc=guide_acc,
        device=device, input_steps=INPUT_STEPS,
        output_steps=CHUNK, ref_blend=ref_blend, mode="safe",
    )
    force_arr = np.full(SEG_LEN, force, dtype=np.float32)
    vel_arr   = np.full(SEG_LEN, velocity, dtype=np.float32)
    acc, _    = rt.predict(force_arr, vel_arr, num_samples=SEG_LEN)
    # post-processing (output limit + gate)
    acc = apply_common_output_limit(acc, roughness=roughness, velocity=velocity)
    acc = force_velocity_gate(acc, force, velocity)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-path",    default="pt_files/runs/20260612-003/best_model.pt")
    ap.add_argument("--cache-path", default="pt_files/inference_cache_allinone.npz")
    ap.add_argument("--out-dir",    default="pt_files")
    ap.add_argument("--ref-blend",  type=float, default=0.20)
    args = ap.parse_args()

    device = "cpu"
    model, x_mean, x_std, y_mean, y_std = load_model_from_pt(
        args.pt_path, device=device, in_ch=3, output_steps=CHUNK)
    cache = InferenceCache(args.cache_path)

    # sweep parameters
    roughness_list = [5, 12, 23, 45, 58, 66, 100]
    vel_steps  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]  # rows
    force_steps = [0.5, 1.0, 2.0, 3.0, 5.0]                               # cols

    t_ms = np.arange(SEG_LEN) / SR * 1000.0
    n_rows, n_cols = len(vel_steps), len(force_steps)

    for roughness in roughness_list:
        print(f"  roughness={roughness} ...")
        guide_info = cache.build_reference_guide(roughness=roughness,
                                                  seg_target_len=SEG_LEN, seg_idx=10)
        guide_acc = guide_info[0]["acc"]

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.0, n_rows * 1.8),
                                 sharex=True, sharey=True)
        fig.suptitle(f"Model output sweep  |  Roughness = {roughness}",
                     fontsize=13, fontweight="bold")

        for ri, vel in enumerate(vel_steps):
            for ci, frc in enumerate(force_steps):
                ax = axes[ri][ci]
                acc = generate_segment(model, x_mean, x_std, y_mean, y_std,
                                       guide_acc, roughness, frc, vel,
                                       ref_blend=args.ref_blend, device=device)
                ax.plot(t_ms, acc, lw=0.5, color="#378ADD")
                rms_val = float(np.sqrt(np.mean(acc**2)))
                tgt     = roughness_to_target_rms(roughness, velocity=vel)
                ax.axhline(0, lw=0.3, color="#888", ls="--", alpha=0.5)
                ax.set_ylim(-2.5, 2.5)
                ax.tick_params(labelsize=5)
                ax.grid(True, lw=0.2, alpha=0.4)

                title_str = (f"V={vel:.2f} F={frc:.1f}N\n"
                             f"RMS={rms_val:.3f} tgt={tgt:.3f}")
                ax.set_title(title_str, fontsize=5.5)

                if ci == 0:
                    ax.set_ylabel(f"V={vel:.2f}", fontsize=6)
                if ri == n_rows - 1:
                    ax.set_xlabel("ms", fontsize=5)

        # column headers
        for ci, frc in enumerate(force_steps):
            axes[0][ci].annotate(f"F={frc:.1f}N", xy=(0.5, 1.35),
                                  xycoords="axes fraction", ha="center",
                                  fontsize=8, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(args.out_dir, f"sweep_roughness_{roughness:03d}.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"    saved: {out}")

    print("Done.")

if __name__ == "__main__":
    main()
