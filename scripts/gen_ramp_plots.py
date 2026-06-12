"""
gen_ramp_plots.py  —  ONNX 기반 ramp 시나리오 시각화 (spectrogram 포함)

Usage:
    python -m scripts.gen_ramp_plots \
        --pt-path pt_files/runs/20260612-003/best_model.pt \
        --cache-path pt_files/inference_cache_allinone.npz
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as scipy_signal
import onnxruntime as ort

X_MEAN = np.array([-3.4580029e-05, 1.9603732e+00, 6.7262508e-02], dtype=np.float32)
X_STD  = np.array([0.61265767, 1.8065012, 0.03293957], dtype=np.float32)
Y_MEAN = np.float32(0.0006642408552579582)
Y_STD  = np.float32(0.6099205017089844)

SEG_LEN     = 4000
SR          = 8000
CHUNK       = 40
INPUT_STEPS = 400


# ── 경량 InferenceCache (torch 불필요) ────────────────────────────────────────
class InferenceCacheLite:
    def __init__(self, npz_path):
        self.npz = np.load(npz_path, allow_pickle=True)
        cols = [str(c) for c in self.npz["window_meta_columns"].tolist()]
        self.wm = pd.DataFrame({c: self.npz[f"window_meta__{c}"] for c in cols})
        splits = ["train", "val", "test"]
        self.wm["arr_idx"] = -1
        offset = 0
        for sp in splits:
            mask = self.wm["split"] == sp
            n = mask.sum()
            self.wm.loc[mask, "arr_idx"] = np.arange(n) + offset
            offset += n
        self._acc = np.concatenate([self.npz[f"X_{sp}"][:, 0, :] for sp in splits])

    def get_guide(self, roughness, seg_idx=5):
        sub = self.wm[self.wm["roughness"] == roughness].copy()
        if sub.empty:
            return np.zeros(SEG_LEN, dtype=np.float32)
        keys = (sub["pid"].astype(str) + "_" + sub["trial"].astype(str) +
                "_" + sub["seg_idx"].astype(str)).unique()
        key = keys[seg_idx % len(keys)]
        pid, trial, si = key.rsplit("_", 2)
        grp = sub[(sub["pid"].astype(str) == pid) &
                  (sub["trial"].astype(str) == trial) &
                  (sub["seg_idx"].astype(str) == si)]
        sig = np.zeros(SEG_LEN, dtype=np.float32)
        cnt = np.zeros(SEG_LEN, dtype=np.float32)
        for _, row in grp.iterrows():
            end_in = int(row["resampled_end_in"])
            s = max(0, end_in - 400); e = min(SEG_LEN, end_in)
            idx = int(row["arr_idx"]); l = e - s
            if l <= 0: continue
            sig[s:e] += self._acc[idx][:l]; cnt[s:e] += 1.
        valid = cnt > 0; sig[valid] /= cnt[valid]
        return sig


def fv_gate(sig, f_arr, v_arr, ft=0.3, vt=0.005, ff=2.0, vf=0.03):
    fg = np.clip((np.asarray(f_arr) - ft) / max(ff - ft, 1e-6), 0, 1)
    vg = np.clip((np.asarray(v_arr) - vt) / max(vf - vt, 1e-6), 0, 1)
    return (np.asarray(sig, dtype=np.float32) * fg * vg).astype(np.float32)


def generate(sess, roughness, force_arr, vel_arr, guide_acc, ref_blend=0.20):
    r_val     = np.float32(roughness / 100.)
    acc_buf   = np.array(guide_acc[:INPUT_STEPS], dtype=np.float32)
    force_buf = np.full(INPUT_STEPS, float(force_arr[0]), dtype=np.float32)
    vel_buf   = np.full(INPUT_STEPS, float(vel_arr[0]),   dtype=np.float32)
    ref_pos   = [0]

    def get_ref(n):
        s = ref_pos[0]; e = s + n; L = len(guide_acc)
        chunk = guide_acc[s:e] if e <= L else np.concatenate([guide_acc[s:L], guide_acc[:e-L]])
        ref_pos[0] = (ref_pos[0] + n) % L
        return chunk.astype(np.float32)

    out = []
    while len(out) < SEG_LEN:
        need = min(CHUNK, SEG_LEN - len(out)); i = len(out)
        fv = float(force_arr[min(i, len(force_arr) - 1)])
        vv = float(vel_arr[min(i,  len(vel_arr)   - 1)])
        X  = np.stack([acc_buf, force_buf, vel_buf,
                       np.full(INPUT_STEPS, r_val, dtype=np.float32)], 0)
        Xn = X.copy()
        Xn[:3] = (X[:3] - X_MEAN[:, None]) / (X_STD[:, None] + 1e-8)
        pred_n = sess.run(None, {"input": Xn[None].astype(np.float32)})[0][0]
        pred   = pred_n * Y_STD + Y_MEAN
        final  = (1. - ref_blend) * pred + ref_blend * get_ref(CHUNK)
        acc_buf   = np.roll(acc_buf,   -CHUNK); acc_buf[-CHUNK:]   = final
        force_buf = np.roll(force_buf, -CHUNK); force_buf[-CHUNK:] = fv
        vel_buf   = np.roll(vel_buf,   -CHUNK); vel_buf[-CHUNK:]   = vv
        out.extend(final[:need].tolist())

    return fv_gate(np.array(out, dtype=np.float32), force_arr, vel_arr)


def make_ramp_figure(sess, cache, scenarios, roughness_list,
                     title, out_path, ref_blend=0.20,
                     acc_ylim=(-1.6, 1.6), force_max=6.5, vel_max=0.12):
    """
    scenarios: list of (label, force_arr, vel_arr, color)
    각 시나리오 = 3열: [waveform+spectrogram | force | velocity]
    """
    t_ms  = np.arange(SEG_LEN) / SR * 1000.
    n_r   = len(roughness_list)
    n_s   = len(scenarios)
    col_w = [4.2, 1.6, 1.6] * n_s

    fig = plt.figure(figsize=(sum(col_w) * 1.05, n_r * 2.2))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.002)

    outer = gridspec.GridSpec(n_r, n_s * 3, figure=fig,
                              width_ratios=col_w,
                              hspace=0.55, wspace=0.22)

    for ri, roughness in enumerate(roughness_list):
        print(f"  R={roughness} ...", flush=True)
        guide_acc = cache.get_guide(roughness)

        for ci, (label, f_arr, v_arr, color) in enumerate(scenarios):
            acc     = generate(sess, roughness, f_arr, v_arr, guide_acc, ref_blend)
            rms_val = float(np.sqrt(np.mean(acc ** 2)))
            c0, c1, c2 = ci * 3, ci * 3 + 1, ci * 3 + 2

            # waveform + spectrogram
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[ri, c0],
                height_ratios=[1, 1.3], hspace=0.06)
            ax_w = fig.add_subplot(inner[0])
            ax_s = fig.add_subplot(inner[1])

            ax_w.plot(t_ms, acc, lw=0.45, color=color, rasterized=True)
            ax_w.axhline(0, lw=0.3, color="#aaa", ls="--", alpha=0.4)
            ax_w.set_ylim(*acc_ylim)
            ax_w.text(0.02, 0.94, f"RMS={rms_val:.3f}",
                      transform=ax_w.transAxes, fontsize=6, color="crimson", va="top")
            ax_w.tick_params(labelsize=5, labelbottom=False)
            ax_w.grid(True, lw=0.2, alpha=0.3)
            if ci == 0:
                ax_w.set_ylabel(f"R={roughness}\nm/s²", fontsize=7, fontweight="bold")
            if ri == 0:
                ax_w.set_title(label, fontsize=8, fontweight="bold")
                ax_w.annotate("Acceleration (m/s²)", xy=(0.5, 1.07),
                              xycoords="axes fraction", ha="center", fontsize=6)

            f_sp, t_sp, Sxx = scipy_signal.spectrogram(
                acc, fs=SR, nperseg=256, noverlap=224, window="hann", scaling="density")
            fm  = f_sp <= 800
            Sdb = 10 * np.log10(Sxx[fm] + 1e-12)
            ax_s.pcolormesh(t_sp * 1000., f_sp[fm], Sdb,
                            shading="gouraud", cmap="inferno",
                            vmin=np.percentile(Sdb, 15), vmax=np.percentile(Sdb, 99))
            ax_s.set_ylabel("Hz", fontsize=5.5)
            ax_s.tick_params(labelsize=5)
            if ri == n_r - 1:
                ax_s.set_xlabel("Time (ms)", fontsize=6)
            if ri == 0:
                ax_s.annotate("Spectrogram (0–800 Hz)", xy=(0.5, -0.3),
                              xycoords="axes fraction", ha="center", fontsize=5.5)

            # force
            ax_f = fig.add_subplot(outer[ri, c1])
            ax_f.plot(t_ms, f_arr, lw=0.9, color="#E05A3A")
            ax_f.set_ylim(-0.1, force_max)
            ax_f.set_ylabel("N", fontsize=5.5)
            ax_f.tick_params(labelsize=5)
            ax_f.grid(True, lw=0.2, alpha=0.3)
            if ri == 0: ax_f.set_title("Force (N)", fontsize=7)
            if ri == n_r - 1: ax_f.set_xlabel("Time (ms)", fontsize=6)

            # velocity
            ax_v = fig.add_subplot(outer[ri, c2])
            ax_v.plot(t_ms, v_arr, lw=0.9, color="#3AAE6E")
            ax_v.set_ylim(-0.003, vel_max)
            ax_v.set_ylabel("m/s", fontsize=5.5)
            ax_v.tick_params(labelsize=5)
            ax_v.grid(True, lw=0.2, alpha=0.3)
            if ri == 0: ax_v.set_title("Velocity (m/s)", fontsize=7)
            if ri == n_r - 1: ax_v.set_xlabel("Time (ms)", fontsize=6)

    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-path",    default="pt_files/runs/20260612-003/best_model.pt")
    ap.add_argument("--cache-path", default="pt_files/inference_cache_allinone.npz")
    ap.add_argument("--out-dir",    default="pt_files")
    ap.add_argument("--ref-blend",  type=float, default=0.20)
    args = ap.parse_args()

    onnx_path = args.pt_path.replace("best_model.pt", "best_model.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = args.pt_path  # .onnx 직접 넘긴 경우
    sess  = ort.InferenceSession(onnx_path)
    cache = InferenceCacheLite(args.cache_path)

    roughness_list = [5, 12, 23, 45, 58, 66, 100]
    vel_fix  = np.full(SEG_LEN, 0.056, dtype=np.float32)
    vel_ramp = np.linspace(0.0,  0.09,  SEG_LEN, dtype=np.float32)
    colors   = ["#378ADD", "#E05A3A", "#3AAE6E", "#9B59B6"]

    print("\n=== Force ramp by interval ===")
    make_ramp_figure(
        sess, cache,
        [("0.5→1.0 N", np.linspace(0.5, 1.0, SEG_LEN, dtype=np.float32), vel_fix, colors[0]),
         ("1.0→2.0 N", np.linspace(1.0, 2.0, SEG_LEN, dtype=np.float32), vel_fix, colors[1]),
         ("2.0→3.5 N", np.linspace(2.0, 3.5, SEG_LEN, dtype=np.float32), vel_fix, colors[2]),
         ("3.5→5.0 N", np.linspace(3.5, 5.0, SEG_LEN, dtype=np.float32), vel_fix, colors[3])],
        roughness_list,
        title="Force ramp by interval  (vel=0.056 m/s fixed)",
        out_path=os.path.join(args.out_dir, "ramp_force_intervals_spec.png"),
        ref_blend=args.ref_blend, force_max=6.0, vel_max=0.12,
    )

    print("\n=== Velocity ramp ===")
    make_ramp_figure(
        sess, cache,
        [("F=1.0N", np.full(SEG_LEN, 1.0, dtype=np.float32), vel_ramp, colors[0]),
         ("F=2.0N", np.full(SEG_LEN, 2.0, dtype=np.float32), vel_ramp, colors[1]),
         ("F=4.0N", np.full(SEG_LEN, 4.0, dtype=np.float32), vel_ramp, colors[2])],
        roughness_list,
        title="Velocity ramp  0 → 0.09 m/s  (force fixed)",
        out_path=os.path.join(args.out_dir, "ramp_velocity_spec.png"),
        ref_blend=args.ref_blend, force_max=5.0, vel_max=0.12,
    )

    print("\n=== Force + Velocity simultaneous ramp ===")
    make_ramp_figure(
        sess, cache,
        [("F 0→6N + V 0→0.09",
          np.linspace(0, 6, SEG_LEN, dtype=np.float32),
          np.linspace(0, 0.09, SEG_LEN, dtype=np.float32), colors[0]),
         ("F 1→4N + V 0.03→0.09",
          np.linspace(1, 4, SEG_LEN, dtype=np.float32),
          np.linspace(0.03, 0.09, SEG_LEN, dtype=np.float32), colors[1])],
        roughness_list,
        title="Force + Velocity simultaneous linear ramp",
        out_path=os.path.join(args.out_dir, "ramp_both_spec.png"),
        ref_blend=args.ref_blend, force_max=7.0, vel_max=0.12,
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()
