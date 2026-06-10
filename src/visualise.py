"""
Visualisation helpers.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import (
    SEG_TARGET_LEN, INPUT_STEPS, OUTPUT_STEPS, STRIDE,
    FORCE_THRESHOLD, VEL_THRESHOLD, MIN_SEG_LEN,
    MERGE_GAP, MARGIN, SMOOTH_W, POST_MERGE_MIN_LEN,
    MAX_ABS_PEAK, SAMPLE_RATE,
    ACC_LIM, FORCE_LIM, VEL_LIM, FFT_LIM,
)
from .inference import (
    get_resampled_segment_signals,
    generated_signal_from_roughness,
    choose_row_for_roughness,
    choose_reasonable_segment_id,
)


# ──────────────────────────────────────────────────────────────────────────────
# FFT helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_fft(signal, sample_rate=SAMPLE_RATE, use_window=True):
    sig = np.nan_to_num(np.asarray(signal, dtype=np.float32), nan=0.0)
    sig -= np.mean(sig)
    if len(sig) < 2:
        return None, None
    win = np.hanning(len(sig)).astype(np.float32) if use_window else np.ones(len(sig), np.float32)
    spec = np.abs(np.fft.rfft(sig * win))
    freq = np.fft.rfftfreq(len(sig), d=1.0 / sample_rate)
    return freq, spec


# ──────────────────────────────────────────────────────────────────────────────
# Simple signal / spectrum plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_signal(signal, title="Signal", label="signal", figsize=(14, 4),
                preview_signal=None, preview_label=None):
    x = np.nan_to_num(np.asarray(signal, dtype=np.float32), nan=0.0)
    plt.figure(figsize=figsize)
    plt.plot(x, label=label, linewidth=1.2, color="blue")
    if preview_signal is not None:
        plt.plot(np.asarray(preview_signal, dtype=np.float32),
                 label=preview_label or "preview", linewidth=1.2, color="orange")
    plt.title(title)
    plt.xlabel("Resampled segment index")
    plt.ylabel("Acceleration")
    plt.ylim(-4, 4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spectrum(signal, sample_rate=SAMPLE_RATE, title="Spectrum", label="signal",
                  figsize=(14, 4), max_freq=None, use_db=False):
    freq, mag = compute_fft(signal, sample_rate)
    if freq is None:
        print("[WARN] spectrum plot skipped: signal too short")
        return
    if use_db:
        mag = 20.0 * np.log10(mag + 1e-12)
    if max_freq is not None:
        keep = freq <= max_freq
        freq, mag = freq[keep], mag[keep]
    plt.figure(figsize=figsize)
    plt.plot(freq, mag, linewidth=1.2, label=label)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)" if use_db else "Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Force-velocity case table
# ──────────────────────────────────────────────────────────────────────────────

def build_fv_case_table(roughness, seg_meta_df, max_abs_peak=MAX_ABS_PEAK):
    rows = seg_meta_df[seg_meta_df["roughness"] == roughness].copy()
    if len(rows) == 0:
        return pd.DataFrame()

    case_rows = []
    for _, row in rows.iterrows():
        sigs = get_resampled_segment_signals(Path(row["path"]), max_abs_peak=max_abs_peak)
        for sid, item in sigs.items():
            acc   = np.asarray(item["acc_res"],   dtype=np.float32)
            force = np.asarray(item["force_res"], dtype=np.float32)
            vel   = np.asarray(item["vel_res"],   dtype=np.float32)
            case_rows.append({
                "roughness": roughness,
                "path": str(Path(row["path"])),
                "path_name": Path(row["path"]).name,
                "split": row.get("split", "unknown"),
                "pid": row.get("pid", "unknown"),
                "trial": row.get("trial", "unknown"),
                "seg_idx": int(sid),
                "force_mean_abs": float(np.mean(np.abs(force))),
                "vel_mean_abs":   float(np.mean(np.abs(vel))),
                "acc_peak":       float(np.max(np.abs(acc))),
                "acc_rms":        float(np.sqrt(np.mean(acc ** 2))),
            })

    if not case_rows:
        return pd.DataFrame()

    df = pd.DataFrame(case_rows)
    f = df["force_mean_abs"].to_numpy()
    v = df["vel_mean_abs"].to_numpy()
    f_n = (f - f.min()) / (f.max() - f.min() + 1e-8)
    v_n = (v - v.min()) / (v.max() - v.min() + 1e-8)
    df["fv_score"] = 0.5 * f_n + 0.5 * v_n
    return df.sort_values("fv_score").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOW / MID / HIGH + FFT plot (actual only)
# ──────────────────────────────────────────────────────────────────────────────

def plot_low_mid_high(roughness, seg_meta_df,
                      out_root="force_velocity_low_mid_high_plots",
                      save=True, show=True,
                      y_lim_acc=ACC_LIM, y_lim_force=FORCE_LIM,
                      y_lim_vel=VEL_LIM, y_lim_fft=(0, 1000),
                      max_freq=1000, sample_rate=SAMPLE_RATE):
    case_df = build_fv_case_table(roughness, seg_meta_df)
    if len(case_df) == 0:
        print(f"[SKIP] roughness={roughness} | no valid cases")
        return None

    Path(out_root).mkdir(parents=True, exist_ok=True)
    selected = [
        ("LOW",  case_df.iloc[0]),
        ("MID",  case_df.iloc[len(case_df) // 2]),
        ("HIGH", case_df.iloc[-1]),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 9))
    for ri, (label, case) in enumerate(selected):
        sigs = get_resampled_segment_signals(Path(case["path"]), max_abs_peak=MAX_ABS_PEAK)
        sid = int(case["seg_idx"])
        if sid not in sigs:
            continue
        item = sigs[sid]
        acc   = np.asarray(item["acc_res"],   dtype=np.float32)
        force = np.asarray(item["force_res"], dtype=np.float32)
        vel   = np.asarray(item["vel_res"],   dtype=np.float32)
        freq, spec = compute_fft(acc, sample_rate)
        keep = freq <= max_freq

        for ax, arr, ttl, lbl, ylim in [
            (axes[ri, 0], acc,   f"{label} Acc",   "Acceleration", y_lim_acc),
            (axes[ri, 1], force, "Force",           "Force",        y_lim_force),
            (axes[ri, 2], vel,   "Velocity",        "Velocity",     y_lim_vel),
        ]:
            ax.plot(arr, linewidth=1.0)
            ax.set_title(ttl); ax.set_ylabel(lbl); ax.set_xlabel("Sample index")
            if ylim: ax.set_ylim(*ylim)

        axes[ri, 3].plot(freq[keep], spec[keep], color="purple", linewidth=1.2)
        axes[ri, 3].set_title("FFT Spectrum")
        axes[ri, 3].set_xlabel("Frequency (Hz)")
        axes[ri, 3].set_ylabel("Magnitude")
        axes[ri, 3].set_xlim(0, max_freq)
        if y_lim_fft: axes[ri, 3].set_ylim(*y_lim_fft)

    fig.suptitle(f"Roughness {roughness} | LOW-MID-HIGH + FFT", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        sp = Path(out_root) / f"r_{roughness}_fft.png"
        fig.savefig(sp, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {sp}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return case_df


# ──────────────────────────────────────────────────────────────────────────────
# Generated signal plot (actual vs generated + FFT)
# ──────────────────────────────────────────────────────────────────────────────

def plot_generated_vs_actual(
    roughness, seg_meta_df, model,
    x_mean, x_std, y_mean, y_std,
    device="cpu",
    out_dir="generated_plots",
    save=True, show=True,
    y_lim_acc=ACC_LIM, y_lim_force=FORCE_LIM,
    y_lim_vel=VEL_LIM, y_lim_fft=(0, 1000),
    max_freq=1000, sample_rate=SAMPLE_RATE,
    random_seed=1234,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        gen_sig, row, used_seg_idx, seed_info = generated_signal_from_roughness(
            roughness=roughness,
            split_df=seg_meta_df,
            model=model,
            x_mean=x_mean, x_std=x_std,
            y_mean=y_mean, y_std=y_std,
            device=device,
            random_seed=random_seed,
        )
    except Exception as e:
        print(f"[SKIP] r={roughness} | {e}")
        return

    # Reference signals
    ref_force = ref_vel = np.zeros_like(gen_sig, dtype=np.float32)
    if row is not None:
        sigs = get_resampled_segment_signals(Path(row["path"]), max_abs_peak=MAX_ABS_PEAK)
        if used_seg_idx in sigs:
            ref_force = np.asarray(sigs[used_seg_idx]["force_res"], dtype=np.float32)
            ref_vel   = np.asarray(sigs[used_seg_idx]["vel_res"],   dtype=np.float32)

    freq, spec = compute_fft(gen_sig, sample_rate)
    keep = freq <= max_freq

    fig, axes = plt.subplots(1, 4, figsize=(26, 4))
    axes[0].plot(gen_sig,   color="#FF8C00", linewidth=1.4, label="Generated")
    axes[0].set_title(f"Generated | r={roughness}")
    axes[0].set_ylabel("Acceleration")
    if y_lim_acc: axes[0].set_ylim(*y_lim_acc)

    axes[1].plot(ref_force, color="blue", linewidth=1.2)
    axes[1].set_title("Reference Force")
    axes[1].set_ylabel("Force")
    if y_lim_force: axes[1].set_ylim(*y_lim_force)

    axes[2].plot(ref_vel,   color="blue", linewidth=1.2)
    axes[2].set_title("Reference Velocity")
    axes[2].set_ylabel("Velocity")
    if y_lim_vel: axes[2].set_ylim(*y_lim_vel)

    axes[3].plot(freq[keep], spec[keep], color="#FF8C00", linewidth=1.4, label="Generated FFT")
    axes[3].set_title("FFT Spectrum")
    axes[3].set_xlabel("Frequency (Hz)")
    axes[3].set_ylabel("Magnitude")
    axes[3].set_xlim(0, max_freq)
    if y_lim_fft: axes[3].set_ylim(*y_lim_fft)

    info = (
        f"mode={seed_info.get('seed_mode','NA')}\n"
        f"lower={seed_info.get('lower_r','NA')}\n"
        f"upper={seed_info.get('upper_r','NA')}\n"
        f"alpha={seed_info.get('alpha_used','NA')}\n"
        f"seg={used_seg_idx}"
    )
    axes[0].text(0.01, 0.95, info, transform=axes[0].transAxes, fontsize=8, va="top")

    plt.tight_layout()
    if save:
        sp = Path(out_dir) / f"generated_r_{roughness}.png"
        fig.savefig(sp, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {sp}")
    if show:
        plt.show()
    else:
        plt.close(fig)
