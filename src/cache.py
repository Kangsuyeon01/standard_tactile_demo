"""
NPZ cache: save and load all-in-one data arrays + metadata.
"""
from pathlib import Path
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

def save_cache(
    data_dict: dict,
    seg_meta_df: pd.DataFrame,
    window_meta_df: pd.DataFrame,
    all_peaks: np.ndarray,
    out_path,
) -> None:
    """Compress everything into a single .npz file."""
    save_dict = {
        "X_train": data_dict["X_train"].astype(np.float32),
        "Y_train": data_dict["Y_train"].astype(np.float32),
        "X_val":   data_dict["X_val"].astype(np.float32),
        "Y_val":   data_dict["Y_val"].astype(np.float32),
        "X_test":  data_dict["X_test"].astype(np.float32),
        "Y_test":  data_dict["Y_test"].astype(np.float32),
        "all_peaks": np.asarray(all_peaks, dtype=np.float32),
        "seg_meta_columns":    np.array(seg_meta_df.columns.astype(str)),
        "window_meta_columns": np.array(window_meta_df.columns.astype(str)),
    }
    for col in seg_meta_df.columns:
        save_dict[f"seg_meta__{col}"] = seg_meta_df[col].to_numpy()
    for col in window_meta_df.columns:
        save_dict[f"window_meta__{col}"] = window_meta_df[col].to_numpy()

    np.savez_compressed(out_path, **save_dict)
    print(f"[SAVE] cache -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Load helpers
# ──────────────────────────────────────────────────────────────────────────────

def _restore_df(npz, prefix: str, columns_key: str) -> pd.DataFrame:
    cols = [str(c) for c in npz[columns_key].tolist()]
    data = {}
    for col in cols:
        key = f"{prefix}__{col}"
        if key in npz:
            data[col] = npz[key]
    return pd.DataFrame(data)


# Module-level cache state
_CACHE_DATA: dict | None = None
_CACHE_SEG_META: pd.DataFrame | None = None
_CACHE_WIN_META: pd.DataFrame | None = None
_CACHE_PEAKS: np.ndarray | None = None


def activate_cache(cache_path) -> pd.DataFrame:
    """
    Load the all-in-one NPZ into module-level variables.
    Returns a copy of seg_meta_df for convenience.
    """
    global _CACHE_DATA, _CACHE_SEG_META, _CACHE_WIN_META, _CACHE_PEAKS

    npz = np.load(cache_path, allow_pickle=True)

    _CACHE_DATA = {
        "train": {"X": npz["X_train"].astype(np.float32), "Y": npz["Y_train"].astype(np.float32)},
        "val":   {"X": npz["X_val"].astype(np.float32),   "Y": npz["Y_val"].astype(np.float32)},
        "test":  {"X": npz["X_test"].astype(np.float32),  "Y": npz["Y_test"].astype(np.float32)},
    }
    _CACHE_SEG_META = _restore_df(npz, "seg_meta",    "seg_meta_columns")
    _CACHE_WIN_META = _restore_df(npz, "window_meta", "window_meta_columns")
    _CACHE_PEAKS    = npz["all_peaks"] if "all_peaks" in npz else None

    if "split" not in _CACHE_WIN_META.columns:
        raise KeyError("window_meta must contain a 'split' column")

    _CACHE_WIN_META["_local_idx"] = _CACHE_WIN_META.groupby("split").cumcount()

    if "path" in _CACHE_WIN_META.columns:
        _CACHE_WIN_META["_path_name"] = (
            _CACHE_WIN_META["path"].astype(str)
            .str.replace("\\", "/", regex=False)
            .str.split("/").str[-1]
        )
    if "path" in _CACHE_SEG_META.columns:
        _CACHE_SEG_META["_path_name"] = (
            _CACHE_SEG_META["path"].astype(str)
            .str.replace("\\", "/", regex=False)
            .str.split("/").str[-1]
        )

    print(f"[LOAD] cache activated: {cache_path}")
    return _CACHE_SEG_META.copy()


def _require_cache():
    if _CACHE_DATA is None or _CACHE_WIN_META is None:
        raise RuntimeError("Call activate_cache() first.")


def get_cache_xy(split: str):
    _require_cache()
    return _CACHE_DATA[split]["X"], _CACHE_DATA[split]["Y"]


def filter_window_meta(csv_path=None, roughness=None) -> pd.DataFrame:
    _require_cache()
    wm = _CACHE_WIN_META
    if roughness is not None and "roughness" in wm.columns:
        wm = wm[wm["roughness"] == roughness]
    if csv_path is not None and "_path_name" in wm.columns:
        target = Path(csv_path).name
        wm = wm[wm["_path_name"] == target]
    return wm.reset_index(drop=True)


def collect_xy_from_window_meta(wm: pd.DataFrame):
    """Gather X/Y arrays and aligned window_meta from cache."""
    X_list, Y_list, meta_list = [], [], []
    for split in ("train", "val", "test"):
        sub = wm[wm["split"] == split].copy()
        if len(sub) == 0:
            continue
        X_sp, Y_sp = get_cache_xy(split)
        idx = sub["_local_idx"].astype(int).to_numpy()
        valid = (idx >= 0) & (idx < len(X_sp))
        idx = idx[valid]
        sub = sub.iloc[np.where(valid)[0]].copy()
        if len(idx) == 0:
            continue
        X_list.append(X_sp[idx])
        Y_list.append(Y_sp[idx])
        meta_list.append(sub.drop(columns=["_local_idx"], errors="ignore"))

    if not X_list:
        return None, None, None

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    Y = np.concatenate(Y_list, axis=0).astype(np.float32)
    meta = pd.concat(meta_list, ignore_index=True)
    if "seg_idx" not in meta.columns:
        meta["seg_idx"] = 0
    return X, Y, meta.reset_index(drop=True)
