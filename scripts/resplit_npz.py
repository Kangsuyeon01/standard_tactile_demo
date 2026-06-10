"""
scripts/resplit_npz.py
======================
기존 NPZ 캐시를 재전처리 없이 참가자(participant) 기반으로 재분할.

원리:
  NPZ 안의 window_meta__pid / window_meta__split 컬럼을 이용해
  각 윈도우에 원래 어떤 split이었는지, PID가 무엇인지 알 수 있다.
  전체 X/Y 를 합친 뒤 PID로 새로 train/val/test 를 나눠 저장한다.

Usage:
    # 자동 비율 분리 (PID 정렬 기준 train 70% / val 15% / test 15%)
    python -m scripts.resplit_npz

    # 특정 PID를 val/test 로 지정
    python -m scripts.resplit_npz --val-pids 8 9 --test-pids 10 11

    # 입출력 경로 직접 지정
    python -m scripts.resplit_npz \\
        --in-npz  pt_files/inference_cache_allinone.npz \\
        --out-npz pt_files/inference_cache_participant.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUT_DIR, EXCLUDE_ROUGHNESS


# ── 참가자 → split 결정 ───────────────────────────────────────────────────────

def assign_participant_splits(pids: np.ndarray,
                              val_pids: list, test_pids: list) -> np.ndarray:
    """
    각 윈도우의 PID 배열을 받아 새 split 문자열 배열을 반환.

    val_pids / test_pids 가 비어 있으면 자동 비율 분리:
      - sorted unique PIDs 의 하위 70% → train
      - 다음 15% → val
      - 상위 15% → test
    """
    unique_pids = sorted(set(pids.tolist()))
    n = len(unique_pids)

    if not val_pids and not test_pids:
        val_start  = int(n * 0.70)
        test_start = int(n * 0.85)
        pid_to_split = {}
        for i, pid in enumerate(unique_pids):
            if i >= test_start:
                pid_to_split[pid] = "test"
            elif i >= val_start:
                pid_to_split[pid] = "val"
            else:
                pid_to_split[pid] = "train"
    else:
        val_set  = set(val_pids)
        test_set = set(test_pids)
        pid_to_split = {}
        for pid in unique_pids:
            if pid in test_set:
                pid_to_split[pid] = "test"
            elif pid in val_set:
                pid_to_split[pid] = "val"
            else:
                pid_to_split[pid] = "train"

    new_splits = np.array([pid_to_split[p] for p in pids.tolist()])
    return new_splits, pid_to_split


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--in-npz",  type=str, default=None,
                        help="원본 NPZ 경로 (기본: pt_files/inference_cache_allinone.npz)")
    parser.add_argument("--out-npz", type=str, default=None,
                        help="저장 경로 (기본: pt_files/inference_cache_participant.npz)")
    parser.add_argument("--val-pids",  type=int, nargs="+", default=[],
                        help="val 로 지정할 참가자 ID 목록 (예: --val-pids 8 9)")
    parser.add_argument("--test-pids", type=int, nargs="+", default=[],
                        help="test 로 지정할 참가자 ID 목록 (예: --test-pids 10 11)")
    args = parser.parse_args()

    in_path  = Path(args.in_npz)  if args.in_npz  else OUT_DIR / "inference_cache_allinone.npz"
    out_path = Path(args.out_npz) if args.out_npz else OUT_DIR / "inference_cache_participant.npz"

    if not in_path.exists():
        raise FileNotFoundError(f"NPZ 파일 없음: {in_path}")

    print(f"[LOAD] {in_path}")
    npz = np.load(in_path, allow_pickle=True)

    # ── 1. X/Y 전체 합치기 ────────────────────────────────────────────────────
    X_parts, Y_parts, split_labels = [], [], []
    for sp in ("train", "val", "test"):
        X = npz[f"X_{sp}"].astype(np.float32)
        Y = npz[f"Y_{sp}"].astype(np.float32)
        X_parts.append(X)
        Y_parts.append(Y)
        split_labels.extend([sp] * len(X))
        print(f"  loaded {sp}: {X.shape}")

    X_all = np.concatenate(X_parts, axis=0)
    Y_all = np.concatenate(Y_parts, axis=0)
    split_labels = np.array(split_labels)
    N = len(X_all)
    print(f"  total windows: {N}")

    # ── 2. window_meta 에서 PID 복원 ──────────────────────────────────────────
    # window_meta 컬럼들은 모든 split의 윈도우가 file-processing 순서로 저장됨
    # "split" 컬럼과 split별 cumcount를 이용해 X_all 과 매핑
    meta_split_key = "window_meta__split"
    meta_pid_key   = "window_meta__pid"
    meta_rough_key = "window_meta__roughness"

    if meta_split_key not in npz or meta_pid_key not in npz:
        raise RuntimeError(
            "NPZ 에 window_meta__split 또는 window_meta__pid 컬럼이 없습니다.\n"
            "이 NPZ는 구버전입니다. preprocess.py 를 다시 실행해야 합니다."
        )

    meta_splits    = npz[meta_split_key].astype(str)
    meta_pids      = npz[meta_pid_key].astype(int)
    meta_roughness = npz[meta_rough_key].astype(int) if meta_rough_key in npz \
                     else np.zeros(len(meta_splits), dtype=int)

    # window_meta 행을 X_all 인덱스에 매핑
    # X_all = [X_train(cumcount 0..n_train-1), X_val(...), X_test(...)]
    # window_meta는 file-처리 순서 → split별 cumcount로 X_all 인덱스 계산
    from collections import defaultdict
    split_offset = {"train": 0,
                    "val":   len(X_parts[0]),
                    "test":  len(X_parts[0]) + len(X_parts[1])}
    split_counter = defaultdict(int)

    meta_global_idx = np.full(len(meta_splits), -1, dtype=int)
    for i, sp in enumerate(meta_splits):
        sp = str(sp)
        if sp in split_offset:
            meta_global_idx[i] = split_offset[sp] + split_counter[sp]
            split_counter[sp] += 1

    valid_mask = meta_global_idx >= 0
    print(f"  window_meta rows: {len(meta_splits)}  valid: {valid_mask.sum()}")

    # 전체 윈도우 PID 배열 (X_all 순서)
    window_pids      = np.zeros(N, dtype=int)
    window_roughness = np.zeros(N, dtype=int)
    for i in range(len(meta_splits)):
        if valid_mask[i]:
            gidx = meta_global_idx[i]
            window_pids[gidx]      = meta_pids[i]
            window_roughness[gidx] = meta_roughness[i]

    # ── 3. 참가자 기반 split 재배정 ───────────────────────────────────────────
    # EXCLUDE_ROUGHNESS 는 어떤 split 에도 넣지 않음
    unique_pids = sorted(set(window_pids.tolist()))
    print(f"\n  발견된 PID 목록: {unique_pids}")

    new_splits, pid_to_split = assign_participant_splits(
        window_pids, args.val_pids, args.test_pids,
    )

    print("\n[PARTICIPANT SPLIT]")
    for sp in ("train", "val", "test"):
        sp_pids = sorted([p for p, s in pid_to_split.items() if s == sp])
        roughnesses = sorted(set(window_roughness[new_splits == sp].tolist()))
        n_win = int((new_splits == sp).sum())
        print(f"  {sp:5s}: PIDs={sp_pids}  roughnesses={roughnesses}  windows={n_win}")

    # EXCLUDE_ROUGHNESS 제거
    exclude_mask = np.isin(window_roughness, EXCLUDE_ROUGHNESS)
    new_splits[exclude_mask] = "unused"
    print(f"  excluded (roughness {EXCLUDE_ROUGHNESS}): {exclude_mask.sum()} windows")

    # ── 4. 새 split 배열로 분리 ───────────────────────────────────────────────
    data_new = {}
    for sp in ("train", "val", "test"):
        mask = new_splits == sp
        if mask.sum() == 0:
            raise ValueError(f"split='{sp}' 에 해당하는 윈도우가 없습니다. PID 지정을 확인하세요.")
        data_new[f"X_{sp}"] = X_all[mask]
        data_new[f"Y_{sp}"] = Y_all[mask]

    print("\n[NEW SHAPES]")
    for sp in ("train", "val", "test"):
        print(f"  X_{sp}: {data_new[f'X_{sp}'].shape}")

    # ── 5. 나머지 메타 컬럼 그대로 복사 ─────────────────────────────────────
    save_dict = dict(data_new)
    # all_peaks 등 기타 배열 유지
    for key in npz.files:
        if key.startswith("X_") or key.startswith("Y_"):
            continue
        save_dict[key] = npz[key]

    np.savez_compressed(out_path, **save_dict)
    print(f"\n[SAVE] {out_path}")
    print("[DONE] 재분할 완료. 이제 학습 시 --out-dir 와 NPZ 경로를 지정하세요:")
    print(f"  python -m scripts.train --note \"participant split\"")
    print(f"  (train.py 의 NPZ_PATH 를 {out_path} 로 수정하거나,")
    print(f"   main() 에서 npz_path 를 인자로 받도록 --npz 옵션 추가 가능)")


if __name__ == "__main__":
    main()
