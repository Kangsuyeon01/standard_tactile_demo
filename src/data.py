"""
Data utilities:
  - CSV file table builder
  - Segment detection
  - Resampling
  - Window generation
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EXCLUDE_ROUGHNESS, ROUGHNESS_SPLIT,
    SPLIT_MODE, PARTICIPANT_SPLIT,
    SEG_TARGET_LEN, INPUT_STEPS, OUTPUT_STEPS, STRIDE,
    FORCE_THRESHOLD, VEL_THRESHOLD, MIN_SEG_LEN,
    MERGE_GAP, MARGIN, SMOOTH_W, POST_MERGE_MIN_LEN,
    MAX_ABS_PEAK,
)


# ──────────────────────────────────────────────────────────────────────────────
# File table
# ──────────────────────────────────────────────────────────────────────────────

def parse_meta_from_path(path: Path) -> dict:
    """Extract pid / roughness / trial from filename."""
    pattern = r"data_participant_(\d+)_roughness_(\d+)_trial_(\d+)"
    m = re.fullmatch(pattern, path.stem)
    if m is None:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return {
        "pid": int(m.group(1)),
        "roughness": int(m.group(2)),
        "trial": int(m.group(3)),
        "trial_name": path.stem,
        "path": str(path),
    }


def get_split_from_roughness(roughness: int) -> str:
    r = int(roughness)
    if r in EXCLUDE_ROUGHNESS:
        return "unused"
    for split_name, values in ROUGHNESS_SPLIT.items():
        if r in values:
            return split_name
    return "unused"


def get_split_from_participant(pid: int, all_pids: list) -> str:
    """참가자 ID 기반 split 결정.

    1) config.py 의 PARTICIPANT_SPLIT 에 명시된 PID 우선 적용.
    2) val/test 목록이 모두 비어있으면 자동 비율 분리:
       sorted PID 기준 상위 15% → test, 그 앞 15% → val, 나머지 → train.
    """
    # 명시적 설정 우선
    for split_name in ("test", "val"):
        ids = PARTICIPANT_SPLIT.get(split_name, [])
        if ids and pid in ids:
            return split_name

    # 자동 비율 분리
    if not PARTICIPANT_SPLIT.get("val") and not PARTICIPANT_SPLIT.get("test"):
        sorted_pids = sorted(set(all_pids))
        n = len(sorted_pids)
        idx = sorted_pids.index(pid)
        test_start = int(n * 0.70)   # 상위 30% → test+val
        val_start  = int(n * 0.55)   # 중간 15% → val
        if idx >= test_start:
            return "test"
        elif idx >= val_start:
            return "val"
        else:
            return "train"

    return "train"


def build_split_file_table(root_dir: Path, split_mode: str = None) -> pd.DataFrame:
    """CSV 파일 목록을 스캔해 split 컬럼을 채운 DataFrame 반환.

    split_mode:
        None      → config.py 의 SPLIT_MODE 사용
        "roughness"  → 거칠기 기반 (원래 방식)
        "participant" → 참가자 ID 기반 (모든 거칠기 학습 가능)
    """
    if split_mode is None:
        split_mode = SPLIT_MODE

    rows = []
    for csv_path in sorted(root_dir.rglob("*.csv")):
        try:
            meta = parse_meta_from_path(csv_path)
            rows.append(meta)
        except Exception as e:
            print(f"[SKIP] {csv_path} -> {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if split_mode == "participant":
        all_pids = df["pid"].tolist()
        df["split"] = df["pid"].apply(
            lambda pid: (
                "unused" if df.loc[df["pid"] == pid, "roughness"].iloc[0]
                            in EXCLUDE_ROUGHNESS
                else get_split_from_participant(pid, all_pids)
            )
        )
        # exclude_roughness 처리 (roughness 기준으로도 unused 적용)
        df.loc[df["roughness"].isin(EXCLUDE_ROUGHNESS), "split"] = "unused"
        print(f"[SPLIT MODE] participant-based split")
        print(f"  unique PIDs : {sorted(df['pid'].unique())}")
        for sp in ("train", "val", "test", "unused"):
            pids = sorted(df.loc[df["split"] == sp, "pid"].unique())
            print(f"  {sp:8s}: PIDs={pids}  files={len(df[df['split']==sp])}")
    else:
        df["split"] = df["roughness"].apply(get_split_from_roughness)
        print(f"[SPLIT MODE] roughness-based split")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Signal helpers
# ──────────────────────────────────────────────────────────────────────────────

def moving_average(x: np.ndarray, w: int = 101) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if w <= 1:
        return x.copy()
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def resample_1d(x: np.ndarray, target_len: int = 4000) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == target_len:
        return x.copy()
    if len(x) < 2:
        fill = float(x[0]) if len(x) == 1 else 0.0
        return np.full(target_len, fill, dtype=np.float32)
    old_idx = np.linspace(0.0, 1.0, len(x))
    new_idx = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def normalize_roughness_value(roughness, scale_max: float = 100.0) -> np.float32:
    return np.float32(float(roughness) / float(scale_max))


# ──────────────────────────────────────────────────────────────────────────────
# Segment detection
# ──────────────────────────────────────────────────────────────────────────────

def find_valid_segments(
    force, vel,
    force_threshold=FORCE_THRESHOLD,
    vel_threshold=VEL_THRESHOLD,
    min_len=MIN_SEG_LEN,
    merge_gap=MERGE_GAP,
    margin=MARGIN,
    smooth_w=SMOOTH_W,
    post_merge_min_len=POST_MERGE_MIN_LEN,
) -> list:
    force = np.asarray(force, dtype=np.float32)
    vel   = np.asarray(vel,   dtype=np.float32)

    force_s = moving_average(force, smooth_w)
    vel_s   = moving_average(np.abs(vel), smooth_w)
    valid_mask = (force_s > force_threshold) & (vel_s > vel_threshold)

    segments, start = [], None
    for i, flag in enumerate(valid_mask):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            if i - start >= min_len:
                segments.append([start, i])
            start = None
    if start is not None:
        if len(valid_mask) - start >= min_len:
            segments.append([start, len(valid_mask)])

    if not segments:
        return []

    # Merge nearby segments
    merged = [segments[0]]
    for s, e in segments[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Apply margin & post-merge min length filter
    n = len(force)
    final = []
    for s, e in merged:
        if e - s < post_merge_min_len:
            continue
        final.append((max(0, s - margin), min(n, e + margin)))
    return final


# ──────────────────────────────────────────────────────────────────────────────
# Window building
# ──────────────────────────────────────────────────────────────────────────────

def build_windows_from_resampled_segment(
    acc_res, force_res, vel_res, roughness,
    input_steps=INPUT_STEPS,
    output_steps=OUTPUT_STEPS,
    stride=STRIDE,
):
    """
    Returns
    -------
    X : np.ndarray  [N, 4, input_steps]  channels: acc, force, vel, roughness
    Y : np.ndarray  [N, output_steps]
    meta_list : list of dicts
    """
    X_list, Y_list, meta_list = [], [], []
    total_len = len(acc_res)
    max_start = total_len - input_steps - output_steps
    if max_start < 0:
        return None, None, []

    roughness_ch = np.full(input_steps, normalize_roughness_value(roughness), dtype=np.float32)

    for start in range(0, max_start + 1, stride):
        x_end = start + input_steps
        y_end = x_end + output_steps
        X = np.stack([
            acc_res[start:x_end],
            force_res[start:x_end],
            vel_res[start:x_end],
            roughness_ch,
        ], axis=0).astype(np.float32)
        Y = acc_res[x_end:y_end].astype(np.float32)
        X_list.append(X)
        Y_list.append(Y)
        meta_list.append({
            "resampled_start": start,
            "resampled_end_in": x_end,
            "resampled_end_out": y_end,
        })

    if not X_list:
        return None, None, []
    return np.stack(X_list), np.stack(Y_list), meta_list


def process_one_csv_to_windows(
    csv_path,
    roughness,
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
    max_abs_peak=MAX_ABS_PEAK,
):
    """Process a single CSV into X/Y windows."""
    df = pd.read_csv(csv_path, usecols=["Acceleration", "Force", "Velocity"])
    acc   = df["Acceleration"].to_numpy(dtype=np.float32)
    force = df["Force"].to_numpy(dtype=np.float32)
    vel   = df["Velocity"].to_numpy(dtype=np.float32)

    segments = find_valid_segments(
        force, vel,
        force_threshold=force_threshold,
        vel_threshold=vel_threshold,
        min_len=min_seg_len,
        merge_gap=merge_gap,
        margin=margin,
        smooth_w=smooth_w,
        post_merge_min_len=post_merge_min_len,
    )

    X_all, Y_all, seg_rows, win_rows, all_peaks = [], [], [], [], []

    for seg_idx, (s, e) in enumerate(segments):
        acc_seg   = acc[s:e]
        force_seg = force[s:e]
        vel_seg   = vel[s:e]
        if len(acc_seg) < 10:
            continue

        acc_res   = resample_1d(acc_seg,   target_len=seg_target_len)
        force_res = resample_1d(force_seg, target_len=seg_target_len)
        vel_res   = resample_1d(vel_seg,   target_len=seg_target_len)

        peak = float(np.max(np.abs(acc_res)))
        all_peaks.append(peak)

        if max_abs_peak is not None and peak > max_abs_peak:
            continue

        X_seg, Y_seg, win_meta = build_windows_from_resampled_segment(
            acc_res, force_res, vel_res, roughness,
            input_steps=input_steps,
            output_steps=output_steps,
            stride=stride,
        )
        if X_seg is None or len(X_seg) == 0:
            continue

        seg_rows.append({
            "path": str(csv_path), "seg_idx": seg_idx,
            "raw_seg_start": s, "raw_seg_end": e, "raw_seg_len": e - s,
            "seg_target_len": seg_target_len, "peak": peak,
            "num_windows": len(X_seg),
        })
        for w_idx, m in enumerate(win_meta):
            win_rows.append({
                "path": str(csv_path), "seg_idx": seg_idx,
                "window_idx": w_idx,
                "raw_seg_start": s, "raw_seg_end": e, "raw_seg_len": e - s,
                "seg_target_len": seg_target_len, "peak": peak,
                **m,
            })

        X_all.append(X_seg)
        Y_all.append(Y_seg)

    seg_df = pd.DataFrame(seg_rows)
    win_df = pd.DataFrame(win_rows)

    if not X_all:
        return None, None, seg_df, win_df, segments, all_peaks

    X_file = np.concatenate(X_all, axis=0).astype(np.float32)
    Y_file = np.concatenate(Y_all, axis=0).astype(np.float32)
    return X_file, Y_file, seg_df, win_df, segments, all_peaks


def build_all_windows(
    split_df: pd.DataFrame,
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
    max_abs_peak=MAX_ABS_PEAK,
):
    """Process all CSVs and return data_dict + metadata DataFrames."""
    X_dict = {"train": [], "val": [], "test": []}
    Y_dict = {"train": [], "val": [], "test": []}
    seg_meta_list, win_meta_list = [], []
    all_peaks_total = []

    use_df = split_df[split_df["split"].isin(["train", "val", "test"])].copy()
    print(f"[PROCESS FILES] num files: {len(use_df)}")

    for _, row in use_df.iterrows():
        split = row["split"]
        csv_path = Path(row["path"])
        roughness = row["roughness"]

        try:
            X_file, Y_file, seg_df, win_df, segments, peaks = process_one_csv_to_windows(
                csv_path=csv_path,
                roughness=roughness,
                seg_target_len=seg_target_len,
                input_steps=input_steps,
                output_steps=output_steps,
                stride=stride,
                force_threshold=force_threshold,
                vel_threshold=vel_threshold,
                min_seg_len=min_seg_len,
                merge_gap=merge_gap,
                margin=margin,
                smooth_w=smooth_w,
                post_merge_min_len=post_merge_min_len,
                max_abs_peak=max_abs_peak,
            )

            all_peaks_total.extend(peaks)

            for df_, lst in [(seg_df, seg_meta_list), (win_df, win_meta_list)]:
                if len(df_) > 0:
                    df_ = df_.copy()
                    df_["split"] = split
                    df_["pid"] = row["pid"]
                    df_["roughness"] = roughness
                    df_["trial"] = row["trial"]
                    df_["trial_name"] = row["trial_name"]
                    lst.append(df_)

            if X_file is None or len(X_file) == 0:
                print(f"[SKIP] {split:5s} | {csv_path.name} | segments={len(segments)} | windows=0")
                continue

            X_dict[split].append(X_file)
            Y_dict[split].append(Y_file)
            print(f"[OK]   {split:5s} | {csv_path.name} | segments={len(segments)} | windows={len(X_file)}")

        except Exception as e:
            print(f"[ERROR] {csv_path} -> {e}")

    data_dict = {}
    for sp in ["train", "val", "test"]:
        if not X_dict[sp]:
            raise ValueError(f"No data collected for split={sp}")
        data_dict[f"X_{sp}"] = np.concatenate(X_dict[sp], axis=0).astype(np.float32)
        data_dict[f"Y_{sp}"] = np.concatenate(Y_dict[sp], axis=0).astype(np.float32)

    seg_meta_df = pd.concat(seg_meta_list, ignore_index=True) if seg_meta_list else pd.DataFrame()
    window_meta_df = pd.concat(win_meta_list, ignore_index=True) if win_meta_list else pd.DataFrame()
    all_peaks = np.asarray(all_peaks_total, dtype=np.float32)

    return data_dict, seg_meta_df, window_meta_df, all_peaks
