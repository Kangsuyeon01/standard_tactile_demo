"""
scripts/preprocess.py
=====================
Step 1 – Build the split file table, extract windows from every CSV,
         and save a single all-in-one NPZ cache (+ CSV metadata files).

Usage:
    # 거칠기 기반 split (기본)
    python -m scripts.preprocess

    # 참가자 기반 split (모든 거칠기를 학습에 활용)
    python -m scripts.preprocess --split-mode participant
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ROOT_DIR, OUT_DIR,
    SEG_TARGET_LEN, INPUT_STEPS, OUTPUT_STEPS, STRIDE,
    FORCE_THRESHOLD, VEL_THRESHOLD, MIN_SEG_LEN,
    MERGE_GAP, MARGIN, SMOOTH_W, POST_MERGE_MIN_LEN,
    MAX_ABS_PEAK,
)
from src.data import build_split_file_table, build_all_windows
from src.cache import save_cache


def build_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--split-mode",
        choices=["roughness", "participant"],
        default=None,
        help=(
            "split 방식. 기본은 config.py 의 SPLIT_MODE 값 사용.\n"
            "  roughness   : 거칠기 값으로 train/val/test 분리 (원래 방식)\n"
            "  participant : 참가자 ID로 분리 (모든 거칠기 학습 가능)"
        ),
    )
    p.add_argument(
        "--out-npz",
        type=str,
        default=None,
        help="저장할 NPZ 경로. 기본: pt_files/inference_cache_allinone.npz",
    )
    return p.parse_args()


def main():
    args = build_args()
    npz_path = Path(args.out_npz) if args.out_npz else OUT_DIR / "inference_cache_allinone.npz"

    # ── 1. Build file table ────────────────────────────────────────────────
    # pilot-data 폴더 존재 확인
    if not ROOT_DIR.exists():
        raise FileNotFoundError(
            f"데이터 폴더를 찾을 수 없습니다: {ROOT_DIR.resolve()}\n"
            f"  config.py 의 ROOT_DIR 경로를 확인하거나,\n"
            f"  프로젝트 루트(roughness_model/)에서 실행하세요.\n"
            f"  현재 작업 디렉토리: {Path('.').resolve()}"
        )

    split_df = build_split_file_table(ROOT_DIR, split_mode=args.split_mode)

    if split_df.empty or "split" not in split_df.columns:
        csv_count = len(list(ROOT_DIR.rglob("*.csv")))
        raise RuntimeError(
            f"CSV 파일 파싱 실패: {ROOT_DIR.resolve()} 에서 {csv_count}개 발견됐지만\n"
            f"  파일명 형식이 맞지 않습니다.\n"
            f"  필요한 형식: data_participant_{{N}}_roughness_{{R}}_trial_{{T}}.csv"
        )

    split_df.to_csv(OUT_DIR / "raw_file_table.csv", index=False)
    split_df.to_csv(OUT_DIR / "roughness_split_table.csv", index=False)

    print(f"\n[FILE TABLE] total: {len(split_df)}")
    print(split_df["split"].value_counts(dropna=False))

    # ── 2. Build windows ──────────────────────────────────────────────────
    data_dict, seg_meta_df, window_meta_df, all_peaks = build_all_windows(
        split_df=split_df,
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
    )

    print("\n[WINDOW SHAPES]")
    for sp in ("train", "val", "test"):
        print(f"  X_{sp}: {data_dict[f'X_{sp}'].shape}  Y_{sp}: {data_dict[f'Y_{sp}'].shape}")

    # ── 3. Peak summary plot ───────────────────────────────────────────────
    print(f"\n[PEAKS] total segments: {len(all_peaks)}")
    print(f"  > {MAX_ABS_PEAK}G: {int(np.sum(all_peaks > MAX_ABS_PEAK))}")
    print(f"  max: {float(np.max(all_peaks)):.3f}  median: {float(np.median(all_peaks)):.3f}")

    plt.figure(figsize=(7, 4))
    plt.hist(all_peaks, bins=50, alpha=0.7)
    plt.axvline(MAX_ABS_PEAK, linestyle="--", label=f"{MAX_ABS_PEAK}G")
    plt.title("Peak distribution before filter")
    plt.xlabel("Peak |acc|"); plt.ylabel("Count")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "peak_distribution.png", dpi=150)
    plt.show()

    # ── 4. Save metadata CSVs ─────────────────────────────────────────────
    seg_meta_df.to_csv(OUT_DIR / "segment_meta_table.csv", index=False)
    window_meta_df.to_csv(OUT_DIR / "window_meta_table.csv", index=False)

    # ── 5. Save NPZ cache ─────────────────────────────────────────────────
    save_cache(
        data_dict, seg_meta_df, window_meta_df, all_peaks,
        npz_path,
    )

    print("\n[DONE] Preprocessing complete.")
    print(f"  NPZ saved : {npz_path.resolve()}")
    print(f"  Output dir: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
