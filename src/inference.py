"""
Inference utilities:
  - Segment signal retrieval from cache
  - Roughness interpolation helpers
  - Signal generation (exact + unseen labels)
  - Post-processing
  - DAQ waveform output
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (
    SEG_TARGET_LEN, INPUT_STEPS, OUTPUT_STEPS, STRIDE,
    FORCE_THRESHOLD, VEL_THRESHOLD, MIN_SEG_LEN,
    MERGE_GAP, MARGIN, SMOOTH_W, POST_MERGE_MIN_LEN,
    MAX_ABS_PEAK,
)
from .data import resample_1d, normalize_roughness_value, moving_average
from .cache import filter_window_meta, collect_xy_from_window_meta

# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_windows(X, x_mean, x_std, eps=1e-8):
    return (X - x_mean) / (x_std + eps)


def denormalize_y(pred_n, y_mean, y_std):
    return pred_n * y_std + y_mean


# ──────────────────────────────────────────────────────────────────────────────
# Model prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict_windows_denorm(X_windows, model, x_mean, x_std, y_mean, y_std,
                           device="cpu", batch_size=256):
    if X_windows is None or len(X_windows) == 0:
        return None
    Xn = normalize_windows(X_windows, x_mean, x_std)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch_size):
            xb = torch.tensor(Xn[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(xb).cpu().numpy())
    pred_n = np.concatenate(outs, axis=0)
    return denormalize_y(pred_n, y_mean, y_std).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Model + stats loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model_from_pt(pt_path, device="cpu", in_ch=4, output_steps=40):
    from .model import LiteSeq2SeqCNNGRU_AttnPool
    ckpt = torch.load(pt_path, map_location=device)
    model = LiteSeq2SeqCNNGRU_AttnPool(in_ch=in_ch, output_steps=output_steps).to(device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    def _to_numpy(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        return np.asarray(v, dtype=np.float32)

    if not isinstance(ckpt, dict) or any(k not in ckpt for k in ("x_mean", "x_std", "y_mean", "y_std")):
        raise ValueError("pt file must contain x_mean/x_std/y_mean/y_std")

    x_mean = _to_numpy(ckpt["x_mean"])
    x_std  = _to_numpy(ckpt["x_std"])
    y_mean = _to_numpy(ckpt["y_mean"])
    y_std  = _to_numpy(ckpt["y_std"])
    config = ckpt.get("config", {})
    return model, x_mean, x_std, y_mean, y_std, config


# ──────────────────────────────────────────────────────────────────────────────
# Segment signal retrieval from NPZ cache
# ──────────────────────────────────────────────────────────────────────────────

def _reconstruct_channel(X_seg, win_meta_seg, ch, seg_target_len=SEG_TARGET_LEN):
    sig_sum   = np.zeros(seg_target_len, dtype=np.float32)
    sig_count = np.zeros(seg_target_len, dtype=np.float32)
    win_meta_seg = win_meta_seg.reset_index(drop=True)
    for i, row in win_meta_seg.iterrows():
        end_in = int(row["resampled_end_in"])
        start  = max(0, end_in - X_seg.shape[-1])
        end    = min(seg_target_len, end_in)
        length = end - start
        if length <= 0:
            continue
        sig_sum[start:end]   += X_seg[i, ch, :length]
        sig_count[start:end] += 1.0
    out = np.zeros(seg_target_len, dtype=np.float32)
    valid = sig_count > 0
    out[valid] = sig_sum[valid] / sig_count[valid]
    return out


def get_resampled_segment_signals(csv_path, seg_target_len=SEG_TARGET_LEN,
                                   max_abs_peak=MAX_ABS_PEAK, **_ignored):
    """Retrieve acc/force/vel signals per segment from the NPZ cache."""
    wm = filter_window_meta(csv_path=csv_path, roughness=None)
    if len(wm) == 0:
        return {}

    X_file, _, win_meta_df = collect_xy_from_window_meta(wm)
    if X_file is None or len(X_file) == 0:
        return {}
    if "seg_idx" not in win_meta_df.columns:
        win_meta_df["seg_idx"] = 0

    result = {}
    for sid in sorted(win_meta_df["seg_idx"].dropna().unique()):
        idx = win_meta_df[win_meta_df["seg_idx"] == sid].index.to_numpy()
        X_seg   = X_file[idx]
        meta_s  = win_meta_df.iloc[idx].reset_index(drop=True)

        acc_res   = _reconstruct_channel(X_seg, meta_s, 0, seg_target_len)
        force_res = _reconstruct_channel(X_seg, meta_s, 1, seg_target_len)
        vel_res   = _reconstruct_channel(X_seg, meta_s, 2, seg_target_len)

        peak = float(np.max(np.abs(acc_res)))
        if max_abs_peak is not None and peak > max_abs_peak:
            continue

        result[int(sid)] = {
            "acc_res": acc_res, "force_res": force_res, "vel_res": vel_res,
            "raw_start": None, "raw_end": None, "raw_len": None, "peak": peak,
        }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Window preparation from cache (replaces original process_one_csv_to_windows)
# ──────────────────────────────────────────────────────────────────────────────

def process_one_csv_from_cache(csv_path, roughness, max_abs_peak=MAX_ABS_PEAK,
                                seg_target_len=SEG_TARGET_LEN, **_ignored):
    wm = filter_window_meta(csv_path=csv_path, roughness=roughness)
    if len(wm) == 0:
        wm = filter_window_meta(csv_path=csv_path, roughness=None)
    if len(wm) == 0:
        return None, None, None, None, []

    X_file, Y_file, win_meta_df = collect_xy_from_window_meta(wm)
    if X_file is None or len(X_file) == 0:
        return None, None, None, None, []

    if X_file.shape[1] >= 4:
        X_file[:, 3, :] = normalize_roughness_value(roughness)

    if max_abs_peak is not None and "seg_idx" in win_meta_df.columns:
        keep = []
        for sid in sorted(win_meta_df["seg_idx"].dropna().unique()):
            idx = win_meta_df[win_meta_df["seg_idx"] == sid].index.to_numpy()
            if float(np.max(np.abs(X_file[idx, 0, :]))) <= max_abs_peak:
                keep.extend(idx.tolist())
        if not keep:
            return None, None, None, None, []
        keep = np.asarray(sorted(keep), dtype=int)
        X_file      = X_file[keep]
        Y_file      = Y_file[keep]
        win_meta_df = win_meta_df.iloc[keep].reset_index(drop=True)

    segments = sorted(win_meta_df["seg_idx"].dropna().unique().tolist())
    return X_file, Y_file, None, win_meta_df.reset_index(drop=True), segments


# ──────────────────────────────────────────────────────────────────────────────
# Segment reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_one_segment(pred_windows_seg, win_meta_seg, seg_target_len=SEG_TARGET_LEN):
    pred_sum   = np.zeros(seg_target_len, dtype=np.float32)
    pred_count = np.zeros(seg_target_len, dtype=np.float32)
    win_meta_seg = win_meta_seg.reset_index(drop=True)
    for i, row in win_meta_seg.iterrows():
        s = int(row["resampled_end_in"])
        e = int(row["resampled_end_out"])
        pred_sum[s:e]   += pred_windows_seg[i]
        pred_count[s:e] += 1.0
    pred_signal = np.full(seg_target_len, np.nan, dtype=np.float32)
    valid = pred_count > 0
    pred_signal[valid] = pred_sum[valid] / pred_count[valid]
    return pred_signal, pred_count


# ──────────────────────────────────────────────────────────────────────────────
# Row selection helpers
# ──────────────────────────────────────────────────────────────────────────────

def _available_roughnesses(split_df):
    vals = sorted(split_df["roughness"].dropna().unique().tolist())
    return [int(v) if float(v).is_integer() else float(v) for v in vals]


def choose_row_for_roughness(seg_meta_df, roughness, trial=None,
                              prefer_split_order=("train", "val", "test")):
    rows = seg_meta_df[seg_meta_df["roughness"] == roughness].copy()
    if len(rows) == 0:
        return None
    if trial is not None:
        sub = rows[rows["trial"] == trial]
        if len(sub) > 0:
            rows = sub
    if "split" in rows.columns:
        for sp in prefer_split_order:
            sub = rows[rows["split"] == sp]
            if len(sub) > 0:
                rows = sub; break
    if "peak" in rows.columns and len(rows) > 1:
        target = rows["peak"].median()
        return rows.loc[(rows["peak"] - target).abs().idxmin()]
    return rows.iloc[0]


def choose_bracketing_rows(split_df, roughness, trial=None):
    avail = _available_roughnesses(split_df)
    if not avail:
        return None, None, None, None
    lower_cands = [r for r in avail if float(r) <= float(roughness)]
    upper_cands = [r for r in avail if float(r) >= float(roughness)]
    lower_r = max(lower_cands) if lower_cands else None
    upper_r = min(upper_cands) if upper_cands else None
    lower_row = choose_row_for_roughness(split_df, lower_r, trial) if lower_r is not None else None
    upper_row = choose_row_for_roughness(split_df, upper_r, trial) if upper_r is not None else None
    return lower_row, upper_row, lower_r, upper_r


def choose_two_nearest_rows(split_df, roughness, trial=None):
    avail = _available_roughnesses(split_df)
    if not avail:
        return None
    top2 = sorted(avail, key=lambda x: abs(float(x) - float(roughness)))[:2]
    rows = [(r, choose_row_for_roughness(split_df, r, trial)) for r in top2]
    rows = [(r, row) for r, row in rows if row is not None]
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0], rows[0]
    return rows[0], rows[1]


def choose_reasonable_segment_id(seg_signal_dict, preferred_seg_idx=None,
                                  max_abs_peak=MAX_ABS_PEAK, max_rms=None):
    if not seg_signal_dict:
        return None
    valid_ids = []
    for sid, item in seg_signal_dict.items():
        x = np.asarray(item["acc_res"], dtype=np.float32)
        peak = float(np.max(np.abs(x)))
        rms  = float(np.sqrt(np.mean(x ** 2)))
        ok = peak <= max_abs_peak
        if max_rms is not None:
            ok = ok and rms <= max_rms
        if ok:
            valid_ids.append(sid)
    if preferred_seg_idx is not None and preferred_seg_idx in valid_ids:
        return int(preferred_seg_idx)
    if valid_ids:
        peaks = [float(np.max(np.abs(seg_signal_dict[s]["acc_res"]))) for s in valid_ids]
        target = float(np.median(peaks))
        return int(min(valid_ids, key=lambda s: abs(float(np.max(np.abs(seg_signal_dict[s]["acc_res"]))) - target)))
    return int(min(seg_signal_dict.keys(),
                   key=lambda s: float(np.max(np.abs(seg_signal_dict[s]["acc_res"])))))


# ──────────────────────────────────────────────────────────────────────────────
# Signal post-processing
# ──────────────────────────────────────────────────────────────────────────────

def fill_leading_gap(pred_signal, input_steps=INPUT_STEPS,
                     output_steps=OUTPUT_STEPS, mode="repeat_first_pred"):
    x = np.nan_to_num(np.asarray(pred_signal, dtype=np.float32), nan=0.0)
    valid_idx = np.where(~np.isnan(np.asarray(pred_signal)))[0]
    if len(valid_idx) == 0 or valid_idx[0] <= 0:
        return x
    fv = int(valid_idx[0])
    first_block = np.nan_to_num(x[fv:min(fv + output_steps, len(x))], nan=0.0)
    if len(first_block) == 0:
        return x
    if mode == "repeat_first_pred":
        rep = int(np.ceil(fv / len(first_block)))
        x[:fv] = np.tile(first_block, rep)[:fv]
    return np.nan_to_num(x, nan=0.0)


def blend_preview_and_generated(pred_signal, preview_signal,
                                  preview_len=INPUT_STEPS, fade_len=120):
    pred = np.nan_to_num(np.asarray(pred_signal, dtype=np.float32), nan=0.0)
    if preview_signal is None or np.all(np.isnan(preview_signal)):
        return pred
    prev = np.asarray(preview_signal, dtype=np.float32)
    n = len(pred)
    p_len = min(int(preview_len), n)
    pred[:p_len] = np.nan_to_num(prev[:p_len], nan=0.0)
    last = float(prev[np.where(~np.isnan(prev[:p_len]))[0][-1]]) if np.any(~np.isnan(prev[:p_len])) else 0.0
    fe = min(p_len + int(fade_len), n)
    if fe > p_len:
        a = np.linspace(0, 1, fe - p_len, dtype=np.float32)
        bridge = np.full(fe - p_len, last, dtype=np.float32)
        pred[p_len:fe] = (1 - a) * bridge + a * pred[p_len:fe]
    return pred.astype(np.float32)


def hard_clip_by_median(signal, ratio=5.0):
    med = np.median(np.abs(signal)) + 1e-8
    return np.clip(signal, -med * ratio, med * ratio)


def roughness_to_target_rms(roughness, rms_min=0.5, rms_max=1.8, gamma=0.8):
    r = float(np.clip(float(roughness) / 100.0, 0, 1))
    return rms_min + (rms_max - rms_min) * (r ** gamma)


def apply_output_limit(signal, roughness):
    x = np.nan_to_num(np.asarray(signal, dtype=np.float32), nan=0.0)
    x -= np.mean(x)
    x  = hard_clip_by_median(x, ratio=4.5)
    cur_rms = np.sqrt(np.mean(x ** 2)) + 1e-8
    target  = roughness_to_target_rms(roughness, rms_min=0.20, rms_max=0.30, gamma=0.85)
    scale = float(np.clip((1.0 - 0.2) + 0.2 * (target / cur_rms), 0.88, 1.15))
    return (x * scale).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Generated signal from roughness
# ──────────────────────────────────────────────────────────────────────────────

def generated_signal_from_roughness(
    roughness,
    split_df,
    model,
    x_mean, x_std, y_mean, y_std,
    device="cpu",
    seg_target_len=SEG_TARGET_LEN,
    input_steps=INPUT_STEPS,
    output_steps=OUTPUT_STEPS,
    stride=STRIDE,
    force_threshold=FORCE_THRESHOLD,
    vel_threshold=VEL_THRESHOLD,
    min_seg_len=MIN_SEG_LEN,
    merge_gap=MERGE_GAP,
    margin=MARGIN,
    smooth_w=SMOOTH_W,
    post_merge_min_len=POST_MERGE_MIN_LEN,
    batch_size=256,
    trial=None,
    seg_idx=None,
    noise_sigma=0.035,
    drift_scale=0.10,
    smooth_kernel=31,
    noise_on="acc",
    max_abs_peak=MAX_ABS_PEAK,
    random_seed=1234,
    alpha_jitter_std=0.05,
):
    row = choose_row_for_roughness(split_df, roughness, trial=trial)
    preferred_seg_idx = seg_idx

    if row is not None:
        csv_path = Path(row["path"])
        if preferred_seg_idx is None:
            preferred_seg_idx = int(row.get("seg_idx", 0))
        X_file, _, _, win_meta_df, _ = process_one_csv_from_cache(
            csv_path=csv_path, roughness=roughness,
            max_abs_peak=max_abs_peak, seg_target_len=seg_target_len,
        )
        if X_file is None or len(X_file) == 0:
            raise RuntimeError(f"{csv_path.name}: no valid windows")
        seed_info = {
            "seed_mode": "exact_actual_head",
            "requested_roughness": roughness,
            "lower_r": roughness, "upper_r": roughness, "alpha_used": 0.0,
        }
    else:
        lower_row, upper_row, lower_r, upper_r = choose_bracketing_rows(split_df, roughness, trial)
        if lower_row is None or upper_row is None:
            pair = choose_two_nearest_rows(split_df, roughness, trial)
            if pair is None:
                raise ValueError(f"No reference rows found for roughness={roughness}")
            (r1, row1), (r2, row2) = pair
            lower_r, upper_r = sorted([r1, r2])
            lower_row = choose_row_for_roughness(split_df, lower_r, trial)
            upper_row = choose_row_for_roughness(split_df, upper_r, trial)

        denom = float(upper_r) - float(lower_r)
        alpha = float(np.clip(0.0 if abs(denom) < 1e-8 else (float(roughness) - float(lower_r)) / denom, 0, 1))

        X_low, _, _, meta_low, _ = process_one_csv_from_cache(
            csv_path=Path(lower_row["path"]), roughness=roughness,
            max_abs_peak=max_abs_peak, seg_target_len=seg_target_len,
        )
        X_up, _, _, meta_up, _ = process_one_csv_from_cache(
            csv_path=Path(upper_row["path"]), roughness=roughness,
            max_abs_peak=max_abs_peak, seg_target_len=seg_target_len,
        )
        if X_low is None or X_up is None:
            raise RuntimeError("Could not build interpolated windows")

        n = min(len(X_low), len(X_up))
        X_file = (1.0 - alpha) * X_low[:n] + alpha * X_up[:n]
        X_file[:, 3, :] = normalize_roughness_value(roughness)
        win_meta_df = meta_low.iloc[:n].reset_index(drop=True)
        row = lower_row
        seed_info = {
            "seed_mode": "interpolated_windows_actual_head",
            "requested_roughness": roughness,
            "lower_r": lower_r, "upper_r": upper_r, "alpha_used": alpha,
        }

    pred_file = predict_windows_denorm(
        X_file, model, x_mean, x_std, y_mean, y_std,
        device=device, batch_size=batch_size,
    )
    if pred_file is None or len(pred_file) == 0:
        raise RuntimeError("Prediction failed")

    win_meta_df = win_meta_df.reset_index(drop=True)
    avail_segs = sorted(win_meta_df["seg_idx"].unique().tolist())
    use_seg_idx = int(preferred_seg_idx) if preferred_seg_idx in avail_segs else int(np.random.choice(avail_segs))

    seg_win_meta = win_meta_df[win_meta_df["seg_idx"] == use_seg_idx].copy()
    pred_windows_seg = pred_file[seg_win_meta.index.to_numpy()]
    pred_signal, _ = reconstruct_one_segment(
        pred_windows_seg, seg_win_meta.reset_index(drop=True), seg_target_len
    )
    pred_signal = fill_leading_gap(pred_signal, input_steps, output_steps)

    # Replace leading INPUT_STEPS with actual reference
    seg_dict = get_resampled_segment_signals(Path(row["path"]), max_abs_peak=max_abs_peak)
    if seg_dict:
        ref_sid = choose_reasonable_segment_id(seg_dict, use_seg_idx, max_abs_peak)
        if ref_sid is not None:
            ref_acc = seg_dict[ref_sid]["acc_res"]
            head = min(input_steps, len(ref_acc), len(pred_signal))
            pred_signal[:head] = ref_acc[:head]
            fade = min(60, len(pred_signal) - head)
            if fade > 0:
                a = np.linspace(0, 1, fade, dtype=np.float32)
                bridge = np.full(fade, float(pred_signal[head - 1]), dtype=np.float32)
                pred_signal[head:head + fade] = ((1 - a) * bridge + a * pred_signal[head:head + fade])

    alpha = seed_info.get("alpha_used", 0.0) or 0.0
    pred_signal *= 1.0 + 0.25 * float(alpha) * (1.0 - float(alpha)) * 4.0
    pred_signal = np.nan_to_num(pred_signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return pred_signal, row, use_seg_idx, seed_info


# ──────────────────────────────────────────────────────────────────────────────
# DAQ waveform preparation & playback
# ──────────────────────────────────────────────────────────────────────────────

def prepare_waveform_for_daq(signal, sample_rate=8000, seconds=5.0,
                              voltage_scale=1.0, clip_min=-5.0, clip_max=5.0):
    x = np.nan_to_num(np.asarray(signal, dtype=np.float32), nan=0.0)
    target_len = int(sample_rate * seconds)
    repeat_n = int(np.ceil(target_len / len(x)))
    waveform = np.tile(x, repeat_n)[:target_len].astype(np.float32)
    waveform -= np.mean(waveform)
    waveform  = np.clip(waveform * float(voltage_scale), clip_min, clip_max)
    return waveform.astype(np.float32)


def run_waveform_vibration(waveform, sample_rate=8000, ao_channel="Dev1/ao0"):
    """Output waveform via NI-DAQmx (requires nidaqmx package)."""
    import time
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    waveform = np.asarray(waveform, dtype=np.float32)
    with nidaqmx.Task() as task:
        task.ao_channels.add_ao_voltage_chan(ao_channel)
        task.timing.cfg_samp_clk_timing(
            sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=len(waveform),
        )
        task.write(waveform, auto_start=True)
        print(f"[VIB] Output started | {len(waveform)} samples | sr={sample_rate}")
        while not task.is_task_done():
            time.sleep(0.01)
    print("[VIB] Output complete.")
