"""
scripts/demo.py
===============
Interactive vibration demo loop.
  a <roughness> [seg_idx]  → actual waveform from cache
  g <roughness> [seg_idx]  → model-generated waveform
  p                         → precompute signal bank
  quit                      → exit

Usage:
    python -m scripts.demo
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    OUT_DIR, DEVICE, SAMPLE_RATE, PLAY_SECONDS, VOLTAGE_SCALE, AO_CHANNEL,
    SEG_TARGET_LEN, INPUT_STEPS, OUTPUT_STEPS, STRIDE, MAX_ABS_PEAK,
    FORCE_THRESHOLD, VEL_THRESHOLD, MIN_SEG_LEN,
    MERGE_GAP, MARGIN, SMOOTH_W, POST_MERGE_MIN_LEN,
)
from src.cache import activate_cache
from src.inference import (
    load_model_from_pt,
    generated_signal_from_roughness,
    get_resampled_segment_signals,
    choose_row_for_roughness,
    choose_reasonable_segment_id,
    apply_output_limit,
    prepare_waveform_for_daq,
    run_waveform_vibration,
)
from src.visualise import plot_signal, plot_spectrum


PT_PATH    = OUT_DIR / "best_model_light_wo71.pt"
CACHE_PATH = OUT_DIR / "inference_cache_allinone.npz"
BANK_PATH  = OUT_DIR / "cache" / "generated_signal_bank.npz"


# ──────────────────────────────────────────────────────────────────────────────

def parse_command(cmd: str):
    parts = cmd.strip().split()
    if len(parts) < 2:
        raise ValueError("Format: a <roughness>  or  g <roughness>")
    mode = parts[0].lower()
    if mode not in ("a", "g"):
        raise ValueError("First token must be 'a' or 'g'")
    r = float(parts[1])
    roughness = int(r) if r.is_integer() else r
    seg_idx = int(parts[2]) if len(parts) >= 3 else None
    return mode, roughness, seg_idx


def precompute_bank(seg_meta_df, model, x_mean, x_std, y_mean, y_std,
                    roughness_list, n_variants=10):
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    bank = {}
    for r in roughness_list:
        signals = []
        print(f"\n[PRECOMPUTE] roughness={r}")
        for k in range(n_variants):
            seed = int(float(r) * 1000) + k
            try:
                sig, _, _, _ = generated_signal_from_roughness(
                    roughness=r, split_df=seg_meta_df,
                    model=model, x_mean=x_mean, x_std=x_std,
                    y_mean=y_mean, y_std=y_std, device=DEVICE,
                    random_seed=seed, max_abs_peak=MAX_ABS_PEAK,
                )
                signals.append(sig.astype(np.float32))
            except Exception as e:
                print(f"  [WARN] seed={seed}: {e}")
        if signals:
            bank[f"r_{r}"] = np.stack(signals)
    np.savez_compressed(BANK_PATH, **bank)
    print(f"\n[SAVE] signal bank -> {BANK_PATH}")
    return np.load(BANK_PATH, allow_pickle=False)


def actual_vib(roughness, seg_meta_df, seg_idx=None, show_plot=True):
    row = choose_row_for_roughness(seg_meta_df, roughness)
    if row is None:
        raise ValueError(f"No row found for roughness={roughness}")
    sigs = get_resampled_segment_signals(Path(row["path"]), max_abs_peak=MAX_ABS_PEAK)
    if not sigs:
        raise RuntimeError("No valid segments")
    sid = choose_reasonable_segment_id(sigs, seg_idx, MAX_ABS_PEAK)
    signal = apply_output_limit(sigs[sid]["acc_res"], roughness)

    print(f"[ACTUAL] r={roughness} | pid={row.get('pid')} | "
          f"trial={row.get('trial')} | seg={sid}")

    if show_plot:
        plot_signal(signal, title=f"Actual | r={roughness} | seg={sid}", label="Actual")
        plot_spectrum(signal, title=f"Actual spectrum | r={roughness}", max_freq=1000)

    waveform = prepare_waveform_for_daq(signal, SAMPLE_RATE, PLAY_SECONDS, VOLTAGE_SCALE)
    run_waveform_vibration(waveform, SAMPLE_RATE, AO_CHANNEL)
    return waveform


def generated_vib(roughness, seg_meta_df, model, x_mean, x_std, y_mean, y_std,
                   seg_idx=None, show_plot=True):
    sig, row, used_seg_idx, seed_info = generated_signal_from_roughness(
        roughness=roughness, split_df=seg_meta_df,
        model=model, x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std, device=DEVICE,
        seg_idx=seg_idx, max_abs_peak=MAX_ABS_PEAK,
    )
    print(f"[GENERATED] r={roughness} | mode={seed_info.get('seed_mode')} | "
          f"lower={seed_info.get('lower_r')} | upper={seed_info.get('upper_r')} | "
          f"alpha={seed_info.get('alpha_used')} | seg={used_seg_idx}")

    if show_plot:
        plot_signal(sig, title=f"Generated | r={roughness} | seg={used_seg_idx}", label="Generated")
        plot_spectrum(sig, title=f"Generated spectrum | r={roughness}", max_freq=1000)

    waveform = prepare_waveform_for_daq(sig, SAMPLE_RATE, PLAY_SECONDS, VOLTAGE_SCALE)
    run_waveform_vibration(waveform, SAMPLE_RATE, AO_CHANNEL)
    return waveform


# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Activate cache ───────────────────────────────────────────────────────
    seg_meta_df = activate_cache(CACHE_PATH)

    # ── Load model ───────────────────────────────────────────────────────────
    model, x_mean, x_std, y_mean, y_std, _ = load_model_from_pt(
        PT_PATH, device=DEVICE, in_ch=4, output_steps=OUTPUT_STEPS,
    )
    if x_mean.ndim == 1: x_mean = x_mean[:, None]
    if x_std.ndim == 1:  x_std  = x_std[:, None]

    # ── Load or skip signal bank ─────────────────────────────────────────────
    if BANK_PATH.exists():
        SIGNAL_BANK = np.load(BANK_PATH, allow_pickle=False)
        print(f"[LOAD] signal bank: {BANK_PATH}")
    else:
        SIGNAL_BANK = None
        print("[WARN] Signal bank not found. Use 'p' to precompute.")

    avail = sorted(seg_meta_df["roughness"].unique().tolist())
    print("=" * 60)
    print("COMMANDS:  a <roughness> [seg]  |  g <roughness> [seg]  |  p  |  quit")
    print(f"AVAILABLE ROUGHNESS: {avail}")
    print("=" * 60)

    while True:
        try:
            cmd = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue
        if cmd.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        try:
            # Precompute signal bank
            if cmd.lower() in ("p", "precompute"):
                bank_rlist = [1] + list(range(5, 101, 5))
                SIGNAL_BANK = precompute_bank(
                    seg_meta_df, model, x_mean, x_std, y_mean, y_std,
                    roughness_list=bank_rlist, n_variants=10,
                )
                print("[DONE] signal bank ready")
                continue

            mode, roughness, seg_idx = parse_command(cmd)

            if mode == "a":
                actual_vib(roughness, seg_meta_df, seg_idx=seg_idx)

            elif mode == "g":
                key = f"r_{roughness}"
                if SIGNAL_BANK is not None and key in SIGNAL_BANK:
                    signals = SIGNAL_BANK[key]
                    idx     = np.random.randint(0, len(signals))
                    signal  = signals[idx]
                    print(f"[FAST] r={roughness} | variant={idx}")
                    plot_signal(signal, title=f"Fast generated | r={roughness}", label="Generated")
                    waveform = prepare_waveform_for_daq(signal, SAMPLE_RATE, PLAY_SECONDS, VOLTAGE_SCALE)
                    run_waveform_vibration(waveform, SAMPLE_RATE, AO_CHANNEL)
                else:
                    generated_vib(roughness, seg_meta_df, model,
                                  x_mean, x_std, y_mean, y_std, seg_idx=seg_idx)

        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
