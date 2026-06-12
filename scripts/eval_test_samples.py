"""
scripts/eval_test_samples.py
============================
Per-roughness-label evaluation using ALL data (train + val + test).

For each roughness label: 3 rows × 4 cols
  rows : Small F/V | Mid F/V | Large F/V  (based on mean force of segment)
  cols : Acceleration (orig=blue, gen=orange) | Force | Velocity | FFT

Generates one PNG per roughness label:
  <run>/test_samples/R005.png ... R100.png

Usage
-----
  python -m scripts.eval_test_samples
  python -m scripts.eval_test_samples --npz pt_files/inference_cache_allinone.npz
  python -m scripts.eval_test_samples --run-id 20260611-006
"""
import argparse, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INPUT_STEPS  = 400
OUTPUT_STEPS = 40
SAMPLE_RATE  = 8000
SEG_LEN      = 4000
ACC_YLIM     = (-1.5, 1.5)
COND_LABELS  = ["Small F/V", "Mid F/V", "Large F/V"]
DEFAULT_NPZ  = Path("pt_files/inference_cache_allinone.npz")
ROUGHNESS_ALL = [5, 12, 23, 45, 58, 66, 100]


# ── normalisation helpers ─────────────────────────────────────────────────────
def _norm_x(X, x_mean, x_std):
    Xn = X.copy()
    Xn[:, :3, :] = (X[:, :3, :] - x_mean) / (x_std + 1e-8)
    return Xn


def _denorm_y(pred_n, y_mean, y_std):
    return pred_n * float(y_std) + float(y_mean)


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(X_windows, model, x_mean, x_std, y_mean, y_std, batch_size=512):
    import torch
    device = next(model.parameters()).device
    Xn = _norm_x(X_windows, x_mean, x_std)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch_size):
            xb = torch.tensor(Xn[i:i+batch_size], dtype=torch.float32).to(device)
            outs.append(model(xb).cpu().numpy())
    return _denorm_y(np.concatenate(outs, axis=0), y_mean, y_std).astype(np.float32)


# ── signal stitching ──────────────────────────────────────────────────────────
def _stitch_x(seg_df, X_all, ch):
    seg_df = seg_df.sort_values("resampled_start").reset_index(drop=True)
    base    = int(seg_df.iloc[0]["resampled_start"])
    last_end = int(seg_df.iloc[-1]["resampled_start"]) + INPUT_STEPS
    total   = last_end - base
    sig = np.zeros(total); cnt = np.zeros(total)
    for off_abs, li in zip(seg_df["resampled_start"].astype(int),
                           seg_df["arr_idx"].astype(int)):
        off = off_abs - base
        sig[off:off+INPUT_STEPS] += X_all[li, ch, :]
        cnt[off:off+INPUT_STEPS] += 1
    cnt[cnt == 0] = 1
    return sig / cnt


def _stitch_y(seg_df, Y, output_steps=OUTPUT_STEPS, use_local=False):
    """Stitch Y outputs. If use_local=True, use sequential 0-based index."""
    seg_df = seg_df.sort_values("resampled_start").reset_index(drop=True)
    base    = int(seg_df.iloc[0]["resampled_start"])
    last_end = int(seg_df.iloc[-1]["resampled_start"]) + INPUT_STEPS + output_steps
    total   = last_end - base
    sig = np.zeros(total); cnt = np.zeros(total)
    idxs = np.arange(len(seg_df)) if use_local else seg_df["arr_idx"].astype(int).to_numpy()
    for k, (off_abs, li) in enumerate(zip(seg_df["resampled_start"].astype(int), idxs)):
        off = off_abs - base + INPUT_STEPS
        end = min(off + output_steps, total)
        n   = end - off
        sig[off:end] += Y[li if not use_local else k, :n]
        cnt[off:end] += 1
    cnt[cnt == 0] = 1
    return sig / cnt


def _fit(sig, n=SEG_LEN):
    return sig[:n] if len(sig) >= n else np.pad(sig, (0, n - len(sig)))


# ── segment helpers ───────────────────────────────────────────────────────────
def _seg_key(sub):
    if "pid" in sub.columns and "trial" in sub.columns:
        return (sub["pid"].astype(str) + "_" +
                sub["trial"].astype(str) + "_" +
                sub["seg_idx"].astype(str))
    return sub["seg_idx"].astype(str)


def _pick_median(seg_scores):
    if not seg_scores: return None
    ids = list(seg_scores.keys()); sc = np.array([seg_scores[k] for k in ids])
    return ids[int(np.argmin(np.abs(sc - float(np.median(sc)))))]


# ── one row of subplots ───────────────────────────────────────────────────────
def _plot_row(axes_row, force, vel, Y_orig, Y_gen, row_label, no_ref=False):
    t_x  = np.arange(len(force)) / SAMPLE_RATE * 1000
    t_y  = np.arange(len(Y_orig)) / SAMPLE_RATE * 1000
    freq = np.fft.rfftfreq(len(Y_gen), d=1.0 / SAMPLE_RATE)

    axes_row[0].set_ylabel(row_label, fontsize=8)

    # acc: orig (blue) + gen (orange), or gen only if no_ref
    ax = axes_row[0]
    if not no_ref:
        ax.plot(t_y, Y_orig, lw=0.7, color="steelblue",  alpha=0.85, label="orig")
    ax.plot(t_y, Y_gen,  lw=0.7, color="darkorange", alpha=0.85, label="gen")
    ax.set_ylim(*ACC_YLIM)
    ax.axhline(0, lw=0.3, color="gray", ls="--")
    rms_o = np.sqrt(np.mean(Y_orig**2));  rms_g = np.sqrt(np.mean(Y_gen**2))
    if no_ref:
        ax.text(0.02, 0.97, f"RMS={rms_g:.3f}",
                transform=ax.transAxes, fontsize=6.5, va="top",
                color="dimgray", family="monospace")
    else:
        ax.text(0.02, 0.97,
                f"orig RMS={rms_o:.3f}\n gen RMS={rms_g:.3f}",
                transform=ax.transAxes, fontsize=6.5, va="top",
                color="dimgray", family="monospace")
    ax.set_xlabel("ms", fontsize=7)

    # force
    ax = axes_row[1]
    ax.plot(t_x, force, lw=0.6, color="crimson")
    ax.set_ylim(-0.1, max(float(np.abs(force).max())*1.2, 0.5))
    ax.text(0.02, 0.97, f"mean={force.mean():.2f} N",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.set_xlabel("ms", fontsize=7)

    # velocity
    ax = axes_row[2]
    ax.plot(t_x, vel, lw=0.6, color="seagreen")
    ax.set_ylim(-0.005, max(float(vel.max())*1.2, 0.01))
    ax.text(0.02, 0.97, f"mean={vel.mean():.4f} m/s",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.set_xlabel("ms", fontsize=7)

    # FFT: gen only (+ orig if not no_ref)
    ax = axes_row[3]
    if not no_ref:
        ax.plot(freq, np.abs(np.fft.rfft(Y_orig)), lw=0.7, color="steelblue",  alpha=0.75)
    ax.plot(freq, np.abs(np.fft.rfft(Y_gen)),  lw=0.7, color="darkorange", alpha=0.75)
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.axvline(200, lw=0.6, color="gray", ls="--", alpha=0.6)
    ax.set_xlabel("Hz", fontsize=7)

    for ax in axes_row:
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.3, alpha=0.4)


# ── per-roughness plot ────────────────────────────────────────────────────────
def plot_roughness(r, X_all, Y_all, wm,
                   model, x_mean, x_std, y_mean, y_std,
                   save_dir, run_tag, no_ref=False):
    sub = wm[wm["roughness"] == r].copy()
    if len(sub) == 0:
        print(f"  [SKIP] R={r}: no windows"); return

    # unique segment key
    sub["_sk"] = _seg_key(sub)

    # mean force per segment → classify Small / Mid / Large
    seg_force = {}
    for sk, g in sub.groupby("_sk"):
        li = g["arr_idx"].astype(int).to_numpy()
        seg_force[sk] = float(np.abs(X_all[li, 1, :]).mean())

    scores   = np.array(list(seg_force.values()))
    p33, p67 = np.percentile(scores, 33), np.percentile(scores, 67)
    groups   = {
        "Small F/V": {k: v for k, v in seg_force.items() if v <= p33},
        "Mid F/V":   {k: v for k, v in seg_force.items() if p33 < v <= p67},
        "Large F/V": {k: v for k, v in seg_force.items() if v > p67},
    }

    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    title_suffix = "Orange=model" if no_ref else "Blue=original · Orange=model"
    fig.suptitle(
        f"R={r}  [{run_tag}]   {title_suffix}",
        fontsize=10)
    for c, ct in enumerate(["Acceleration (m/s²)", "Force (N)", "Velocity (m/s)", "FFT"]):
        axes[0, c].set_title(ct, fontsize=9)

    for row_idx, cond in enumerate(COND_LABELS):
        sk = _pick_median(groups[cond])
        if sk is None:
            for c in range(4): axes[row_idx, c].text(
                0.5, 0.5, "no data", ha="center", va="center",
                transform=axes[row_idx, c].transAxes, fontsize=8)
            axes[row_idx, 0].set_ylabel(cond, fontsize=8)
            continue

        seg_df = sub[sub["_sk"] == sk].sort_values("resampled_start").reset_index(drop=True)
        li_arr = seg_df["arr_idx"].astype(int).to_numpy()

        Y_gen_raw = run_inference(X_all[li_arr], model, x_mean, x_std, y_mean, y_std)

        force  = _fit(_stitch_x(seg_df, X_all, ch=1))
        vel    = _fit(_stitch_x(seg_df, X_all, ch=2))
        Y_orig = _fit(_stitch_y(seg_df, Y_all))
        Y_gen  = _fit(_stitch_y(seg_df, Y_gen_raw, use_local=True))

        _plot_row(axes[row_idx], force, vel, Y_orig, Y_gen, cond, no_ref=no_ref)

    from matplotlib.lines import Line2D
    legend_handles = []
    if not no_ref:
        legend_handles.append(Line2D([0],[0], color="steelblue",  lw=1.5, label="original"))
    legend_handles.append(Line2D([0],[0], color="darkorange", lw=1.5, label="generated"))
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    out = save_dir / f"R{r:03d}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVE] {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    import pandas as pd
    from src.inference import load_model_from_pt

    out_dir  = Path(args.out_dir)
    npz_path = Path(args.npz) if args.npz else DEFAULT_NPZ
    if not npz_path.exists():
        print(f"[ERROR] NPZ not found: {npz_path}"); sys.exit(1)

    # resolve run / checkpoint
    if args.run_id:
        run_dir = out_dir / "runs" / args.run_id
    else:
        run_dirs = sorted((out_dir / "runs").glob("*-*"))
        run_dir  = next((d for d in reversed(run_dirs)
                         if (d / "best_model.pt").exists()), None)
        if run_dir is None:
            print("[ERROR] no run with best_model.pt"); sys.exit(1)

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] checkpoint not found: {ckpt_path}"); sys.exit(1)

    save_dir = run_dir / "test_samples"
    save_dir.mkdir(exist_ok=True)
    run_tag  = run_dir.name

    print(f"[RUN]  {run_tag}")
    print(f"[NPZ]  {npz_path}")
    print(f"[CKPT] {ckpt_path}")

    from src.config import DEVICE
    model, x_mean, x_std, y_mean, y_std, _ = load_model_from_pt(
        ckpt_path, device=DEVICE, in_ch=3)
    print(f"[MODEL] y_mean={y_mean:.4f}  y_std={y_std:.4f}")

    # load NPZ — merge ALL splits into one flat array with global arr_idx
    npz  = np.load(npz_path, allow_pickle=True)
    cols = [str(c) for c in npz["window_meta_columns"].tolist()]
    wm   = pd.DataFrame({c: npz[f"window_meta__{c}"] for c in cols})

    # build global index: arr_idx = position in the merged X/Y array
    splits = ["train", "val", "test"]
    X_parts, Y_parts = [], []
    offset = 0
    wm["arr_idx"] = -1
    for sp in splits:
        mask = (wm["split"] == sp)
        n    = mask.sum()
        if n == 0: continue
        sp_local = wm.loc[mask].groupby("split").cumcount()
        wm.loc[mask, "arr_idx"] = sp_local + offset
        X_parts.append(npz[f"X_{sp}"])
        Y_parts.append(npz[f"Y_{sp}"])
        offset += n

    X_all = np.concatenate(X_parts, axis=0)
    Y_all = np.concatenate(Y_parts, axis=0)
    print(f"[DATA] X_all={X_all.shape}  Y_all={Y_all.shape}  total_windows={len(wm)}")
    print(f"       roughness={sorted(wm['roughness'].unique())}")

    roughness_list = args.roughness if args.roughness else ROUGHNESS_ALL
    for r in roughness_list:
        print(f"\n[R={r}]")
        plot_roughness(r, X_all, Y_all, wm,
                       model, x_mean, x_std, y_mean, y_std,
                       save_dir, run_tag, no_ref=args.no_ref)

    print(f"\n[DONE] → {save_dir}")


def _build_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out-dir",   default="pt_files")
    p.add_argument("--npz",       default=None)
    p.add_argument("--run-id",    default=None)
    p.add_argument("--roughness", type=int, nargs="+", default=None)
    p.add_argument("--no-ref",   action="store_true",
                   help="레퍼런스(원본 파란선) 없이 모델 출력만 표시")
    return p.parse_args()


if __name__ == "__main__":
    main(_build_args())
