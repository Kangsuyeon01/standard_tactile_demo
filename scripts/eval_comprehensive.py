"""
scripts/eval_comprehensive.py
==============================
Compact grid evaluation: roughness x condition/force/velocity.

Outputs:
  conditions_grid.png   -- [roughness x condition]   accel + F/V panels
  force_sweep_grid.png  -- [roughness x force step]  accel + F/V panels
  vel_sweep_grid.png    -- [roughness x vel step]     accel + F/V panels
  force_ramp_grid.png   -- [roughness x 1] F ramps 0->max, shows Accel+Force+Vel
  vel_ramp_grid.png     -- [roughness x 1] V ramps 0->max, shows Accel+Force+Vel
  summary_rms.png       -- RMS heatmap
  timing.txt

Global y-axis = 70% of NPZ data range (or --acc-scale to adjust).
Each grid cell shows: top=acceleration, bottom=F/V(normalized 0-1).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUT_DIR, DEVICE, INPUT_STEPS, OUTPUT_STEPS
from src.inference import load_model_from_pt
from scripts.realtime import (InferenceCache,
                               enhance_acc_by_roughness,
                               apply_common_output_limit)

MODEL_IN_CH  = 3
SAMPLE_RATE  = 8000
N_SAMPLES    = 4000
N_WINDOWS    = N_SAMPLES // OUTPUT_STEPS   # 100 for 4000/40
ROUGHNESS_LEVELS = [5, 12, 23, 45, 58, 66, 100]

DEFAULT_CONDITIONS = [
    ("Zero", 0.0,  0.0),
    ("Low",  0.5,  0.02),
    ("Mid",  2.0,  0.07),
    ("High", 6.0,  0.15),
]
FORCE_SWEEP = [0.0, 0.3, 0.7, 1.5, 3.0, 6.0]
VEL_SWEEP   = [0.0, 0.01, 0.03, 0.07, 0.12, 0.20]

DEFAULT_ACC_LIM = (-4.4, 4.4)


# --- data helpers ---

def get_global_acc_lim(npz_path: Path, scale: float = 0.7):
    try:
        npz = np.load(npz_path, allow_pickle=True)
        Y   = np.concatenate([npz["Y_train"], npz["Y_val"]], axis=0)
        a   = max(abs(float(Y.min())), abs(float(Y.max()))) * 1.05 * scale
        return (-a, a)
    except Exception:
        return (DEFAULT_ACC_LIM[0] * scale, DEFAULT_ACC_LIM[1] * scale)


def get_fv_conditions_from_npz(npz_path: Path):
    try:
        npz = np.load(npz_path, allow_pickle=True)
        X   = np.concatenate([npz["X_train"], npz["X_val"]], axis=0)
        f_lo, f_mi, f_hi = np.percentile(X[:, 1, :].mean(1), [5, 50, 95])
        v_lo, v_mi, v_hi = np.percentile(X[:, 2, :].mean(1), [5, 50, 95])
        return [
            ("Zero", 0.0,          0.0),
            ("Low",  float(f_lo),  float(v_lo)),
            ("Mid",  float(f_mi),  float(v_mi)),
            ("High", float(f_hi),  float(v_hi)),
        ]
    except Exception:
        return DEFAULT_CONDITIONS


# --- signal generation ---

def generate_signal(model, x_mean, x_std, y_mean, y_std,
                    force_schedule, vel_schedule, roughness_norm, device,
                    n_samples=N_SAMPLES, init_acc=None):
    """
    force_schedule, vel_schedule: array of length n_windows (one value per window),
                                  or scalar (broadcast to all windows).
    init_acc: optional 1-D array of length >= INPUT_STEPS to seed acc_buf
              (use reference guide signal to match realtime.py behaviour).
    Returns: (signal, f_per_sample, v_per_sample, times)
    """
    n_windows = n_samples // OUTPUT_STEPS
    if np.isscalar(force_schedule):
        force_schedule = np.full(n_windows, force_schedule, dtype=np.float32)
    if np.isscalar(vel_schedule):
        vel_schedule   = np.full(n_windows, vel_schedule,   dtype=np.float32)
    force_schedule = np.asarray(force_schedule, dtype=np.float32)
    vel_schedule   = np.asarray(vel_schedule,   dtype=np.float32)

    if init_acc is not None:
        init_acc = np.asarray(init_acc, dtype=np.float32).ravel()
        acc_buf  = np.zeros(INPUT_STEPS, dtype=np.float32)
        n_copy   = min(INPUT_STEPS, len(init_acc))
        acc_buf[:n_copy] = init_acc[:n_copy]
    else:
        acc_buf = np.zeros(INPUT_STEPS, dtype=np.float32)
    output, times = [], []

    for i in range(n_windows):
        fv = float(force_schedule[i])
        vv = float(vel_schedule[i])
        X  = np.stack([acc_buf,
                       np.full(INPUT_STEPS, fv,             np.float32),
                       np.full(INPUT_STEPS, vv,             np.float32),
                       np.full(INPUT_STEPS, roughness_norm, np.float32)], axis=0)
        Xn = X.copy()
        Xn[:3] = (X[:3] - x_mean[0, :3, 0:1]) / (x_std[0, :3, 0:1] + 1e-8)
        x_t = torch.tensor(Xn[np.newaxis], dtype=torch.float32)
        t0  = time.perf_counter()
        model.eval()
        with torch.no_grad():
            pred_n = model(x_t.to(device)).cpu().numpy()[0]
        times.append((time.perf_counter() - t0) * 1000)
        pred = pred_n * y_std + y_mean
        output.extend(pred.tolist())
        acc_buf = np.roll(acc_buf, -OUTPUT_STEPS)
        acc_buf[-OUTPUT_STEPS:] = pred

    sig      = np.array(output, np.float32)
    f_sample = np.repeat(force_schedule, OUTPUT_STEPS)
    v_sample = np.repeat(vel_schedule,   OUTPUT_STEPS)
    return sig, f_sample, v_sample, times


def apply_postproc(sig, roughness, v_arr, rng=None):
    """Apply realtime.py post-processing: spectral shaping + RMS calibration."""
    sig = enhance_acc_by_roughness(sig, roughness, rng=rng)
    sig = apply_common_output_limit(sig, roughness, velocity=float(np.mean(v_arr)))
    return sig


# --- grid plot (2 rows per roughness row: accel + F/V) ---

def plot_grid(grid_data, row_labels, col_labels, title, out_path,
              acc_ylim, force_max, vel_max):
    """
    grid_data[r][c] = (signal, f_per_sample, v_per_sample)
    Top panel: acceleration.  Bottom panel: F (crimson) + V (green) normalized 0-1.
    """
    nrows = len(row_labels)
    ncols = len(col_labels)
    t_ms  = np.arange(N_SAMPLES) / SAMPLE_RATE * 1000
    hr    = [1.6, 0.55] * nrows

    fig, axes = plt.subplots(
        2 * nrows, ncols,
        figsize=(2.8 * ncols, 2.5 * nrows),
        gridspec_kw={"height_ratios": hr},
        sharex=True, sharey=False,
    )
    if ncols == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(title, fontsize=9, y=1.01)

    for r, rlabel in enumerate(row_labels):
        for c, clabel in enumerate(col_labels):
            sig, f_arr, v_arr = grid_data[r][c]
            ax_a = axes[2 * r,     c]
            ax_f = axes[2 * r + 1, c]
            rms  = float(np.sqrt(np.mean(sig ** 2)))

            # -- accel --
            ax_a.plot(t_ms, sig, lw=0.25, color="steelblue", rasterized=True)
            ax_a.set_ylim(*acc_ylim)
            ax_a.text(0.03, 0.97, f"RMS={rms:.3f}",
                      transform=ax_a.transAxes, fontsize=5, va="top", color="crimson")
            ax_a.tick_params(labelsize=4)
            if c == 0:
                ax_a.set_ylabel(f"R={rlabel}\nacc", fontsize=6)
            if r == 0:
                ax_a.set_title(clabel, fontsize=7.5, linespacing=1.4)
            ax_a.tick_params(axis="x", labelbottom=False)

            # -- F/V normalized --
            f_norm = f_arr / (force_max + 1e-8)
            v_norm = v_arr / (vel_max   + 1e-8)
            ax_f.plot(t_ms, f_norm, lw=0.7, color="crimson",  label="F")
            ax_f.plot(t_ms, v_norm, lw=0.7, color="seagreen", label="V")
            ax_f.set_ylim(-0.05, 1.25)
            ax_f.set_yticks([0.0, 0.5, 1.0])
            ax_f.tick_params(labelsize=3.5)
            if c == 0:
                ax_f.set_ylabel("F/V\nnorm", fontsize=5)
                ax_f.legend(fontsize=4, loc="upper left",
                            handlelength=1, framealpha=0.4, borderpad=0.3)
            if r == nrows - 1:
                ax_f.set_xlabel("ms", fontsize=6)
            else:
                ax_f.tick_params(axis="x", labelbottom=False)

    fig.tight_layout(h_pad=0.3)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# --- ramp plot (3 panels per roughness row: Accel | Force | Vel) ---

def plot_ramp_grid(ramp_data, row_labels, title, out_path,
                   acc_ylim, force_max, vel_max):
    """
    ramp_data[r] = (signal, f_per_sample, v_per_sample)
    3 columns: Acceleration | Force (N) | Velocity (m/s)
    """
    nrows = len(row_labels)
    t_ms  = np.arange(N_SAMPLES) / SAMPLE_RATE * 1000

    fig, axes = plt.subplots(
        nrows, 3,
        figsize=(13, 2.2 * nrows),
        sharex=True,
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(title, fontsize=10, y=1.01)

    col_titles = ["Acceleration (m/s^2)", "Force (N)", "Velocity (m/s)"]
    for c, ct in enumerate(col_titles):
        axes[0, c].set_title(ct, fontsize=9)

    for r, rlabel in enumerate(row_labels):
        sig, f_arr, v_arr = ramp_data[r]
        rms = float(np.sqrt(np.mean(sig ** 2)))

        axes[r, 0].plot(t_ms, sig, lw=0.4, color="steelblue", rasterized=True)
        axes[r, 0].set_ylim(*acc_ylim)
        axes[r, 0].set_ylabel(f"R={rlabel}", fontsize=8)
        axes[r, 0].text(0.02, 0.97, f"RMS={rms:.3f}",
                        transform=axes[r, 0].transAxes,
                        fontsize=7, va="top", color="crimson")

        axes[r, 1].plot(t_ms, f_arr, lw=0.9, color="crimson")
        axes[r, 1].set_ylim(-0.05, force_max * 1.15)
        axes[r, 1].set_ylabel("N", fontsize=7)

        axes[r, 2].plot(t_ms, v_arr, lw=0.9, color="seagreen")
        axes[r, 2].set_ylim(-0.001, vel_max * 1.15)
        axes[r, 2].set_ylabel("m/s", fontsize=7)

        for c in range(3):
            axes[r, c].tick_params(labelsize=6)
            if r == nrows - 1:
                axes[r, c].set_xlabel("Time (ms)", fontsize=7)

    fig.tight_layout(h_pad=0.4)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# --- summary heatmap ---

def plot_summary(all_rms, conditions, roughness_list, out_path):
    labels = [c[0] for c in conditions]
    mat    = np.array([[all_rms.get((l, r), 0.0) for r in roughness_list]
                        for l in labels])
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(mat, aspect="auto", cmap="hot")
    plt.colorbar(im, ax=ax, label="RMS")
    ax.set_xticks(range(len(roughness_list)));  ax.set_xticklabels(roughness_list)
    ax.set_yticks(range(len(labels)));          ax.set_yticklabels(labels)
    ax.set_xlabel("Roughness")
    ax.set_title("Output RMS  [condition x roughness]  "
                 "(ideal: Zero near 0, monotone increase left->right)")
    for i in range(len(labels)):
        for j in range(len(roughness_list)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if mat[i, j] < mat.max() * 0.55 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# --- main ---

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-id",    default=None)
    p.add_argument("--out-dir",   default=str(OUT_DIR))
    p.add_argument("--npz",       default=None)
    p.add_argument("--cache",     default=None,
                   help="reference guide NPZ 경로 (기본: --npz 와 동일 또는 allinone)")
    p.add_argument("--roughness", type=int, nargs="+", default=ROUGHNESS_LEVELS)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--n-warmup",  type=int, default=10)
    p.add_argument("--acc-scale", type=float, default=0.7,
                   help="Fraction of NPZ data range to use as y-axis (0.7 = 70%)")
    p.add_argument("--acc-ylim", type=float, nargs=2, default=[-1.5, 1.5],
                   metavar=("LO", "HI"),
                   help="Fixed acc y-axis limits (default: -1.5 1.5, overrides --acc-scale)")
    p.add_argument("--apply-postproc", action="store_true",
                   help="Apply realtime.py post-processing (spectral EQ + RMS calibration) "
                        "to generated signals. Results saved to eval_comprehensive_postproc/.")
    p.add_argument("--no-ref-init", action="store_true",
                   help="레퍼런스 guide 없이 zero init으로 시작 (init_acc=None 강제)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    run_dir = (out_dir / "runs" / args.run_id) if args.run_id \
              else sorted((out_dir / "runs").glob("*-*"))[-1]
    print(f"[RUN] {run_dir.name}")

    model, x_mean, x_std, y_mean, y_std, _ = load_model_from_pt(
        run_dir / "best_model.pt", device=DEVICE,
        in_ch=MODEL_IN_CH, output_steps=OUTPUT_STEPS,
    )

    npz_path   = Path(args.npz) if args.npz \
                 else out_dir / "inference_cache_participant.npz"
    conditions = get_fv_conditions_from_npz(npz_path) \
                 if npz_path.exists() else DEFAULT_CONDITIONS

    # reference guide cache: use allinone NPZ if participant NPZ doesn't exist
    cache_path = Path(args.cache) if getattr(args, "cache", None) \
                 else (npz_path if npz_path.exists()
                       else out_dir / "inference_cache_allinone.npz")
    ref_guides: dict = {}   # roughness -> np.ndarray[INPUT_STEPS]
    if cache_path.exists():
        try:
            cache = InferenceCache(cache_path)
            for r in (args.roughness if args.roughness else ROUGHNESS_LEVELS):
                try:
                    guide, info = cache.build_reference_guide(
                        roughness=r, seg_target_len=N_SAMPLES,
                        seg_idx=10, max_abs_peak=4.0)
                    ref_guides[r] = guide["acc"][:INPUT_STEPS]
                    print(f"  [guide R={r}] {info['guide_mode']}")
                except Exception as e:
                    print(f"  [guide R={r}] failed ({e}), using zero init")
        except Exception as e:
            print(f"[WARN] InferenceCache load failed ({e}), using zero init for all")
    else:
        print(f"[WARN] cache not found: {cache_path} — using zero init_acc")
    if args.acc_ylim is not None:
        acc_ylim = tuple(args.acc_ylim)
    elif npz_path.exists():
        acc_ylim = get_global_acc_lim(npz_path, scale=args.acc_scale)
    else:
        acc_ylim = (DEFAULT_ACC_LIM[0] * args.acc_scale,
                    DEFAULT_ACC_LIM[1] * args.acc_scale)

    mid_force = conditions[2][1]
    mid_vel   = conditions[2][2]
    force_max = max(FORCE_SWEEP[-1], conditions[-1][1])
    vel_max   = max(VEL_SWEEP[-1],   conditions[-1][2])

    print(f"[GLOBAL ACC YLIM x{args.acc_scale}]  {acc_ylim}")
    print("\n[CONDITIONS]")
    for l, f, v in conditions:
        print(f"  {l:5s}  F={f:.3f}  V={v:.4f}")

    for _ in range(args.n_warmup):
        generate_signal(model, x_mean, x_std, y_mean, y_std,
                        1.0, 0.05, 0.45, DEVICE, n_samples=OUTPUT_STEPS)

    postproc_tag = "_postproc" if args.apply_postproc else ""
    save_dir = run_dir / f"eval_comprehensive{postproc_tag}"
    save_dir.mkdir(exist_ok=True)
    pp_rng = np.random.default_rng(42) if args.apply_postproc else None

    roughness_list = args.roughness
    all_rms   = {}
    all_times = []
    cond_grid   = []
    fsweep_grid = []
    vsweep_grid = []
    framp_list  = []
    vramp_list  = []

    n_windows = args.n_samples // OUTPUT_STEPS
    force_ramp_sched = np.linspace(0, force_max, n_windows)
    vel_ramp_sched   = np.linspace(0, vel_max,   n_windows)

    for roughness in roughness_list:
        r_norm   = roughness / 100.0
        init_acc = None if args.no_ref_init else ref_guides.get(roughness)
        print(f"\n[R={roughness}] generating ...  init={'ref' if init_acc is not None else 'zero'}")

        # -- conditions --
        cond_row = []
        for label, f, v in conditions:
            cond_init = None if f < 0.05 else init_acc
            sig, f_arr, v_arr, times = generate_signal(
                model, x_mean, x_std, y_mean, y_std,
                f, v, r_norm, DEVICE, args.n_samples, init_acc=cond_init)
            if args.apply_postproc and f >= 0.05:
                sig = apply_postproc(sig, roughness, v_arr, rng=pp_rng)
            cond_row.append((sig, f_arr, v_arr))
            all_times.extend(times)
            all_rms[(label, roughness)] = float(np.sqrt(np.mean(sig ** 2)))
        cond_grid.append(cond_row)

        # -- force step sweep --
        fsweep_row = []
        for fv in FORCE_SWEEP:
            fsweep_init = None if fv < 0.05 else init_acc
            sig, f_arr, v_arr, times = generate_signal(
                model, x_mean, x_std, y_mean, y_std,
                fv, mid_vel, r_norm, DEVICE, args.n_samples, init_acc=fsweep_init)
            if args.apply_postproc and fv >= 0.05:
                sig = apply_postproc(sig, roughness, v_arr, rng=pp_rng)
            fsweep_row.append((sig, f_arr, v_arr))
            all_times.extend(times)
        fsweep_grid.append(fsweep_row)

        # -- velocity step sweep --
        vsweep_row = []
        for vv in VEL_SWEEP:
            vsweep_init = None if vv < 0.005 else init_acc
            sig, f_arr, v_arr, times = generate_signal(
                model, x_mean, x_std, y_mean, y_std,
                mid_force, vv, r_norm, DEVICE, args.n_samples, init_acc=vsweep_init)
            if args.apply_postproc and vv >= 0.005:
                sig = apply_postproc(sig, roughness, v_arr, rng=pp_rng)
            vsweep_row.append((sig, f_arr, v_arr))
            all_times.extend(times)
        vsweep_grid.append(vsweep_row)

        # -- force ramp (V fixed at mid) --
        sig, f_arr, v_arr, times = generate_signal(
            model, x_mean, x_std, y_mean, y_std,
            force_ramp_sched, mid_vel, r_norm, DEVICE, args.n_samples, init_acc=init_acc)
        if args.apply_postproc:
            sig = apply_postproc(sig, roughness, v_arr, rng=pp_rng)
        framp_list.append((sig, f_arr, v_arr))
        all_times.extend(times)

        # -- velocity ramp (F fixed at mid) --
        sig, f_arr, v_arr, times = generate_signal(
            model, x_mean, x_std, y_mean, y_std,
            mid_force, vel_ramp_sched, r_norm, DEVICE, args.n_samples, init_acc=init_acc)
        if args.apply_postproc:
            sig = apply_postproc(sig, roughness, v_arr, rng=pp_rng)
        vramp_list.append((sig, f_arr, v_arr))
        all_times.extend(times)

    # -- plots --
    print("\n[PLOTTING]")
    cond_labels = [
        f"{c[0]}\nF={c[1]:.2f}N  V={c[2]:.3f}"
        for c in conditions
    ]
    plot_grid(
        cond_grid,
        [str(r) for r in roughness_list],
        cond_labels,
        f"Conditions  [{run_dir.name}]",
        save_dir / "conditions_grid.png",
        acc_ylim, force_max, vel_max,
    )
    plot_grid(
        fsweep_grid,
        [str(r) for r in roughness_list],
        [f"F={f:.2f}N\n(V={mid_vel:.3f} fix)" for f in FORCE_SWEEP],
        f"Force sweep  (vel={mid_vel:.3f} fixed)  [{run_dir.name}]",
        save_dir / "force_sweep_grid.png",
        acc_ylim, force_max, vel_max,
    )
    plot_grid(
        vsweep_grid,
        [str(r) for r in roughness_list],
        [f"V={v:.3f}\n(F={mid_force:.2f} fix)" for v in VEL_SWEEP],
        f"Velocity sweep  (force={mid_force:.2f} fixed)  [{run_dir.name}]",
        save_dir / "vel_sweep_grid.png",
        acc_ylim, force_max, vel_max,
    )
    plot_ramp_grid(
        framp_list,
        [str(r) for r in roughness_list],
        f"Force ramp  0 -> {force_max:.1f} N  (vel={mid_vel:.3f} fixed)  [{run_dir.name}]",
        save_dir / "force_ramp_grid.png",
        acc_ylim, force_max, vel_max,
    )
    plot_ramp_grid(
        vramp_list,
        [str(r) for r in roughness_list],
        f"Velocity ramp  0 -> {vel_max:.3f} m/s  (force={mid_force:.2f} fixed)  [{run_dir.name}]",
        save_dir / "vel_ramp_grid.png",
        acc_ylim, force_max, vel_max,
    )
    plot_summary(all_rms, conditions, roughness_list, save_dir / "summary_rms.png")

    arr = np.array(all_times)
    tgt = OUTPUT_STEPS / SAMPLE_RATE * 1000
    ok  = "OK" if np.median(arr) < tgt else "TOO SLOW"
    txt = (f"Inference timing  ({run_dir.name})\n"
           f"  n_calls : {len(arr)}\n"
           f"  mean    : {arr.mean():.3f} ms\n"
           f"  median  : {np.median(arr):.3f} ms\n"
           f"  p95     : {np.percentile(arr, 95):.3f} ms\n"
           f"  min/max : {arr.min():.3f} / {arr.max():.3f} ms\n"
           f"  target  : {tgt:.1f} ms -> {ok}\n")
    (save_dir / "timing.txt").write_text(txt)
    print(f"\n{txt}")
    print(f"[DONE]  {save_dir}  (7 files)")


if __name__ == "__main__":
    main()
