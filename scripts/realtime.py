import argparse
import math
import socket
import struct
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

rng = np.random.default_rng(1234)

GLOVE_FILTERS = {
    "Glove5_CH1": {
        "b": [7.67676360e+02, -1.97780692e+03, 1.83154863e+03, -6.00931356e+02],
        "a": [1.00000000e+00, -2.64603348e+00, 2.49329490e+00, -8.32688261e-01]
    },
    "Glove5_CH2": {
        "b": [7.94313966e+02, -2.02599120e+03, 1.89545144e+03, -6.40054344e+02],
        "a": [1.00000000e+00, -2.59317456e+00, 2.44663879e+00, -8.35278615e-01]
    },
    "Glove3_CH1": {
        "b": [1.55210927e+04, -3.98107251e+04, 3.67055147e+04, -1.19260871e+04],
        "a": [1.00000000e+00, -2.63140512e+00, 2.47158320e+00, -8.24619541e-01]
    },
    "Glove3_CH2": {
        "b": [1.57457636e+04, -4.02056477e+04, 3.77560241e+04, -1.29552109e+04],
        "a": [1.00000000e+00, -2.57841523e+00, 2.42910457e+00, -8.41029514e-01]
    },
    "Glove3_CH3": {
        "b": [1.60521457e+04, -4.09245512e+04, 3.81455229e+04, -1.24051471e+04],
        "a": [1.00000000e+00, -2.61042951e+00, 2.45841029e+00, -8.29104257e-01]
    }
}

filter_states = {}
for ch_name, coeffs in GLOVE_FILTERS.items():
    filter_states[ch_name] = {
        "x_prev": [0.0] * (len(coeffs["b"]) - 1),
        "y_prev": [0.0] * (len(coeffs["a"]) - 1)
    }

# =========================================================
# 0) Small utilities
# =========================================================
def to_1d_float32(x):
    return np.asarray(x, dtype=np.float32).reshape(-1).astype(np.float32)


def moving_average(x, w):
    x = np.asarray(x, dtype=np.float32)
    if w is None or int(w) <= 1:
        return x.astype(np.float32)
    w = int(w)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def normalize_roughness_value(roughness, scale_max=100.0):
    return np.float32(float(roughness) / float(scale_max))


def rms(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x ** 2) + 1e-8))


def safe_basename(path_like):
    s = str(path_like)
    s = s.replace("\\", "/")
    return s.split("/")[-1]


def hard_clip_by_median(signal, ratio=5.0):
    x = np.asarray(signal, dtype=np.float32).copy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(np.abs(x)) + 1e-8
    limit = float(med * ratio)
    if limit <= 1e-8:
        return x.astype(np.float32)
    return np.clip(x, -limit, limit).astype(np.float32)


def roughness_to_target_rms(roughness, velocity=0.05, rms_min=0.18, rms_max=0.90, gamma=0.60):
    r = np.clip(float(roughness) / 100.0, 0.0, 1.0)
    base_rms = rms_min + (rms_max - rms_min) * (r ** gamma)

    # Velocity-dependent gain — derived from analyze_velocity_spectrum.py
    # Rough surfaces (R=100) show ~43% RMS increase over 0.01→0.15 m/s
    # Smooth surfaces (R=5) show ~0% change
    # Formula: vel_gain_coeff = 0.45 * r^1.5
    #   r=0.05 → 0.005 (~0%)  r=0.45 → 0.136 (~14%)  r=1.0 → 0.45 (~45%)
    v = np.clip(float(velocity), 0.0, 0.25)
    vel_gain_coeff = 0.45 * (r ** 1.5)
    v_norm = np.clip((v - 0.01) / 0.14, 0.0, 1.0)
    return float(base_rms * (1.0 + vel_gain_coeff * v_norm))


def force_velocity_gate(signal, force, velocity,
                        force_thresh=0.3, vel_thresh=0.005,
                        force_full=2.0,  vel_full=0.03):
    """
    force 또는 velocity가 낮을 때 출력 amplitude를 0 쪽으로 감쇄.
    - force < force_thresh  또는  vel < vel_thresh → 거의 0
    - force >= force_full   AND  vel >= vel_full   → gain=1 (감쇄 없음)
    두 축의 gain을 곱해서 최종 gate를 만든다.
    """
    f = float(force)
    v = float(velocity)
    f_gain = float(np.clip((f - force_thresh) / max(force_full - force_thresh, 1e-6), 0.0, 1.0))
    v_gain = float(np.clip((v - vel_thresh)   / max(vel_full   - vel_thresh,   1e-6), 0.0, 1.0))
    gate   = f_gain * v_gain
    return (np.asarray(signal, dtype=np.float32) * gate).astype(np.float32)


def apply_common_output_limit(signal, roughness, velocity=0.05):
    x = np.asarray(signal, dtype=np.float32).copy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    x = hard_clip_by_median(x, ratio=4.5)

    cur_rms = rms(x)
    target = roughness_to_target_rms(roughness, velocity=velocity)
    raw_scale = target / (cur_rms + 1e-8)
    scale = float(np.clip(raw_scale, 0.20, 3.0))
    return (x * scale).astype(np.float32)


def enhance_acc_by_roughness(acc, roughness, rng=None, smooth_w=5):
    """
    Data-driven spectral shaping based on measured reference signals.

    Measured pattern (metrics_reference.csv):
      smooth (R=5):   centroid≈126 Hz, HF_ratio=0.040  →  buzzy / HF-dominant
      mid    (R=23):  centroid≈134 Hz, HF_ratio=0.052  →  most HF
      rough  (R=100): centroid≈111 Hz, HF_ratio=0.027  →  thuddy / LF-dominant

    Strategy:
      LF gain  ∝  roughness   (rough → bigger slow oscillation)
      HF gain  ∝  1-roughness (smooth → more fine buzz)
      Noise texture: smooth→fine HF noise, rough→coarser LF noise
    """
    if rng is None:
        rng = np.random.default_rng()
    x = to_1d_float32(acc).copy()
    r = np.clip(float(roughness) / 100.0, 0.0, 1.0)

    # ── LF / HF split ─────────────────────────────────────────────────────────
    # smooth_w=5 at 8000 Hz → LPF cutoff ~500 Hz
    slow = moving_average(x, smooth_w)   # LF ≤ ~500 Hz
    fast = x - slow                      # HF > ~500 Hz

    # ── spectral gains (calibrated to measured data) ──────────────────────────
    # r=0 (smooth): lf_gain=0.80, hf_gain=1.40  →  buzzy texture
    # r=1 (rough):  lf_gain=1.40, hf_gain=0.70  →  thuddy texture
    lf_gain = 0.50 + 1.50 * r            # 0.50 → 2.00  (rough → strong thud)
    hf_gain = 2.00 - 1.50 * r            # 2.00 → 0.50  (smooth → strong buzz)

    x = slow * lf_gain + fast * hf_gain

    # ── roughness-textured noise ───────────────────────────────────────────────
    # smooth → fine HF noise;  rough → coarser LF noise
    noise_raw = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    noise_w   = max(1, int(1 + 9 * r))   # r=0 → w=1 (HF noise), r=1 → w=10 (LF noise)
    noise_lf  = moving_average(noise_raw, noise_w)
    noise_hf  = noise_raw - noise_lf

    sigma_lf = 0.008 * r                 # rough surface: LF noise component
    sigma_hf = 0.008 * (1.0 - r)         # smooth surface: HF noise component
    x = x + noise_lf * sigma_lf + noise_hf * sigma_hf

    x = hard_clip_by_median(x, ratio=5.0)
    return x.astype(np.float32)

def acc_to_uint16_wave(acc_signal, device_num, channel_num):
    global GLOVE_FILTERS, filter_states
    
    channel_name = f"Glove{3 + device_num*2}_CH{channel_num+1}"

    b = GLOVE_FILTERS[channel_name]["b"]
    a = GLOVE_FILTERS[channel_name]["a"]
    x_prev = filter_states[channel_name]["x_prev"]
    y_prev = filter_states[channel_name]["y_prev"]

    acc_signal = np.asarray(acc_signal, dtype=np.float32).reshape(-1)
    filtered_signal = np.zeros_like(acc_signal)

    minn = 0
    maxx = 4095
    offset = 2048
    if device_num==0:
        maxx = 65535
        offset = 32768

    for s_idx in range(len(acc_signal)):
        x_curr = float(acc_signal[s_idx])

        y_curr = b[0] * x_curr
        
        for i in range(1, len(b)):
            y_curr += b[i] * x_prev[i - 1]
            
        for i in range(1, len(a)):
            y_curr -= a[i] * y_prev[i - 1]

        if math.isnan(y_curr) or math.isinf(y_curr):
            y_curr = 0.0
            x_prev = [0.0] * len(x_prev)
            y_prev = [0.0] * len(y_prev)

        if len(x_prev) > 0:
            x_prev = [x_curr] + x_prev[:-1]
            y_prev = [y_curr] + y_prev[:-1]

        filtered_signal[s_idx] = max(minn, min(maxx, int(y_curr)+offset))

    filter_states[channel_name]["x_prev"] = x_prev
    filter_states[channel_name]["y_prev"] = y_prev

    return filtered_signal.astype(np.int32)


# =========================================================
# 1) Model definition: FiLM roughness conditioning
# =========================================================
class LiteSeq2SeqCNNGRU_AttnPool(nn.Module):
    def __init__(self, in_ch=3, output_steps=40):
        # in_ch: dynamic 채널 수 (acc, force, vel). roughness(ch3)는 FiLM으로 별도 처리.
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 32 * 2),   # gamma 32 + beta 32
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, 24, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.attn = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, output_steps),
        )

    def forward(self, x):
        r  = x[:, 3, 0:1]              # [B, 1] roughness 0~1
        h  = self.conv1(x[:, :3, :])   # [B, 24, T]
        h  = self.conv2(h)             # [B, 32, T]
        film  = self.film_net(r)                  # [B, 64]
        gamma = film[:, :32].unsqueeze(2)         # [B, 32, 1]
        beta  = film[:, 32:].unsqueeze(2)         # [B, 32, 1]
        h     = gamma * h + beta                  # FiLM 적용
        h  = h.transpose(1, 2)                    # [B, T, 32]
        h, _ = self.gru(h)                        # [B, T, 32]
        w  = torch.softmax(self.attn(h), dim=1)   # [B, T, 1]
        ctx = (h * w).sum(dim=1)                  # [B, 32]
        return self.head(ctx)                     # [B, 40]


# =========================================================
# ONNX Runtime wrapper (drop-in replacement for PyTorch model)
# =========================================================
class _ONNXModel:
    """onnxruntime-based inference wrapper — same call signature as nn.Module."""
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1   # single-threaded fastest for small models
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._inp = self._sess.get_inputs()[0].name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self._sess.run(None, {self._inp: x.cpu().numpy()})[0]
        return torch.from_numpy(out)

    def eval(self):   return self
    def to(self, *a, **kw): return self


def load_model_from_pt(pt_path, device="cpu", in_ch=3, output_steps=40,
                       onnx_path=None, use_compile=False):
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"PT file not found: {pt_path}")

    ckpt = torch.load(
                        pt_path,
                        map_location=device,
                        weights_only=False
                    )
    model = LiteSeq2SeqCNNGRU_AttnPool(in_ch=in_ch, output_steps=output_steps).to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    # filter out training-only keys (e.g. roughness_head) not present in inference model
    model_keys = set(model.state_dict().keys())
    filtered   = {k: v for k, v in state.items() if k in model_keys}
    missing    = model_keys - set(filtered.keys())
    if missing:
        print(f"[WARN] missing keys in checkpoint: {missing}")
    model.load_state_dict(filtered, strict=True)
    model.eval()

    # ONNX: auto-export if onnx_path given but file doesn't exist yet
    if onnx_path is not None:
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            print(f"[ONNX] Exporting to {onnx_path} ...")
            dummy = torch.randn(1, 4, 400)
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=["input"], output_names=["output"],
                opset_version=17,
            )
            print(f"[ONNX] Saved ({onnx_path.stat().st_size//1024} KB)")
        model = _ONNXModel(str(onnx_path))
        print(f"[ONNX] Loaded {onnx_path}")
    elif use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[COMPILE] torch.compile applied")
        except Exception as e:
            print(f"[COMPILE] torch.compile failed ({e}), using eager mode")

    if not isinstance(ckpt, dict):
        raise ValueError("pt 파일이 dict가 아니라서 x_mean/x_std/y_mean/y_std를 읽을 수 없습니다.")

    def as_np(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        return np.asarray(v, dtype=np.float32)

    x_mean = as_np(ckpt.get("x_mean"))
    x_std = as_np(ckpt.get("x_std"))
    y_mean = as_np(ckpt.get("y_mean"))
    y_std = as_np(ckpt.get("y_std"))

    if x_mean is None or x_std is None or y_mean is None or y_std is None:
        raise ValueError("pt 파일 안에 x_mean/x_std/y_mean/y_std가 없습니다.")

    if x_mean.ndim == 1:
        x_mean = x_mean[:, None]
    if x_std.ndim == 1:
        x_std = x_std[:, None]

    return model, x_mean, x_std, y_mean, y_std


# =========================================================
# 2) NPZ cache and reference guide
# =========================================================
def restore_df_from_npz(npz, prefix, columns_key):
    cols = [str(c) for c in npz[columns_key].tolist()]
    data = {}
    for col in cols:
        key = f"{prefix}__{col}"
        if key in npz:
            data[col] = npz[key]
    return pd.DataFrame(data)


class InferenceCache:
    def __init__(self, cache_path):
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"cache npz not found: {cache_path}")

        npz = np.load(cache_path, allow_pickle=True)
        self.data = {
            "train": {"X": npz["X_train"].astype(np.float32), "Y": npz["Y_train"].astype(np.float32)},
            "val": {"X": npz["X_val"].astype(np.float32), "Y": npz["Y_val"].astype(np.float32)},
            "test": {"X": npz["X_test"].astype(np.float32), "Y": npz["Y_test"].astype(np.float32)},
        }
        self.seg_meta = restore_df_from_npz(npz, "seg_meta", "seg_meta_columns")
        self.win_meta = restore_df_from_npz(npz, "window_meta", "window_meta_columns")

        if "split" not in self.win_meta.columns:
            raise KeyError("window_meta에 split 컬럼이 필요합니다.")

        self.win_meta["_local_idx"] = self.win_meta.groupby("split").cumcount()
        if "path" in self.win_meta.columns:
            self.win_meta["_path_name"] = self.win_meta["path"].map(safe_basename)
        if "path" in self.seg_meta.columns:
            self.seg_meta["_path_name"] = self.seg_meta["path"].map(safe_basename)

        print(f"[LOAD] cache: {cache_path}")

    def available_roughnesses(self):
        if "roughness" not in self.seg_meta.columns:
            return []
        vals = sorted(self.seg_meta["roughness"].dropna().unique().tolist())
        out = []
        for v in vals:
            fv = float(v)
            out.append(int(fv) if fv.is_integer() else fv)
        return out

    def choose_row_for_roughness(self, roughness, prefer_split_order=("train", "val", "test")):
        if "roughness" not in self.seg_meta.columns:
            return None
        rows = self.seg_meta[self.seg_meta["roughness"] == roughness].copy()
        if len(rows) == 0:
            return None
        if "split" in rows.columns:
            for sp in prefer_split_order:
                sub = rows[rows["split"] == sp].copy()
                if len(sub) > 0:
                    rows = sub
                    break
        if "peak" in rows.columns and len(rows) > 1:
            target_peak = rows["peak"].median()
            idx = (rows["peak"] - target_peak).abs().idxmin()
            return rows.loc[idx]
        return rows.iloc[0]

    def choose_bracketing_rows(self, roughness):
        avail = self.available_roughnesses()
        if len(avail) == 0:
            return None, None, None, None
        lower_candidates = [r for r in avail if float(r) <= float(roughness)]
        upper_candidates = [r for r in avail if float(r) >= float(roughness)]
        lower_r = max(lower_candidates) if lower_candidates else None
        upper_r = min(upper_candidates) if upper_candidates else None
        lower_row = self.choose_row_for_roughness(lower_r) if lower_r is not None else None
        upper_row = self.choose_row_for_roughness(upper_r) if upper_r is not None else None
        return lower_row, upper_row, lower_r, upper_r

    def choose_nearest_row(self, roughness):
        avail = self.available_roughnesses()
        if len(avail) == 0:
            return None, None
        nearest = min(avail, key=lambda r: abs(float(r) - float(roughness)))
        return nearest, self.choose_row_for_roughness(nearest)

    def _filter_window_meta(self, csv_path=None, roughness=None):
        wm = self.win_meta
        if roughness is not None and "roughness" in wm.columns:
            wm = wm[wm["roughness"] == roughness]
        if csv_path is not None and "_path_name" in wm.columns:
            target_name = safe_basename(csv_path)
            wm = wm[wm["_path_name"] == target_name]
        return wm.reset_index(drop=True)

    def _collect_xy_from_window_meta(self, wm):
        x_list = []
        y_list = []
        meta_list = []
        for split in ["train", "val", "test"]:
            sub = wm[wm["split"] == split].copy()
            if len(sub) == 0:
                continue
            X_split = self.data[split]["X"]
            Y_split = self.data[split]["Y"]
            idx = sub["_local_idx"].astype(int).to_numpy()
            valid = (idx >= 0) & (idx < len(X_split))
            valid_pos = np.where(valid)[0]
            idx = idx[valid]
            sub = sub.iloc[valid_pos].copy()
            if len(idx) == 0:
                continue
            x_list.append(X_split[idx])
            y_list.append(Y_split[idx])
            meta_list.append(sub.drop(columns=["_local_idx"], errors="ignore"))
        if len(x_list) == 0:
            return None, None, None
        X = np.concatenate(x_list, axis=0).astype(np.float32)
        Y = np.concatenate(y_list, axis=0).astype(np.float32)
        meta = pd.concat(meta_list, ignore_index=True)
        if "seg_idx" not in meta.columns:
            meta["seg_idx"] = 0
        return X, Y, meta.reset_index(drop=True)

    @staticmethod
    def _reconstruct_cached_input_channel(X_seg, win_meta_seg, channel_idx, seg_target_len=4000):
        sig_sum = np.zeros(seg_target_len, dtype=np.float32)
        sig_count = np.zeros(seg_target_len, dtype=np.float32)
        win_meta_seg = win_meta_seg.reset_index(drop=True)
        for i, row in win_meta_seg.iterrows():
            end_in = int(row["resampled_end_in"])
            start = end_in - X_seg.shape[-1]
            end = end_in
            start = max(0, start)
            end = min(seg_target_len, end)
            length = end - start
            if length <= 0:
                continue
            sig_sum[start:end] += X_seg[i, channel_idx, :length]
            sig_count[start:end] += 1.0
        out = np.zeros(seg_target_len, dtype=np.float32)
        valid = sig_count > 0
        out[valid] = sig_sum[valid] / sig_count[valid]
        return out.astype(np.float32)

    def get_resampled_segment_signals(self, csv_path, seg_target_len=4000, max_abs_peak=None):
        wm = self._filter_window_meta(csv_path=csv_path, roughness=None)
        if len(wm) == 0:
            print(f"[WARN] no window_meta matched for path: {safe_basename(csv_path)}")
            return {}
        X_file, _, meta = self._collect_xy_from_window_meta(wm)
        if X_file is None or len(X_file) == 0:
            return {}

        seg_signal_dict = {}
        for seg_idx in sorted(meta["seg_idx"].dropna().unique().tolist()):
            idx = meta[meta["seg_idx"] == seg_idx].index.to_numpy()
            if len(idx) == 0:
                continue
            X_seg = X_file[idx]
            meta_seg = meta.iloc[idx].reset_index(drop=True)
            acc = self._reconstruct_cached_input_channel(X_seg, meta_seg, 0, seg_target_len)
            force = self._reconstruct_cached_input_channel(X_seg, meta_seg, 1, seg_target_len)
            vel = self._reconstruct_cached_input_channel(X_seg, meta_seg, 2, seg_target_len)
            peak = float(np.max(np.abs(acc)))
            if max_abs_peak is not None and peak > max_abs_peak:
                continue
            seg_signal_dict[int(seg_idx)] = {
                "acc": acc,
                "force": force,
                "vel": vel,
                "peak": peak,
            }
        return seg_signal_dict

    def choose_reasonable_segment_id(self, seg_signal_dict, preferred_seg_idx=None, max_abs_peak=4.0):
        if not seg_signal_dict:
            return None
        valid_ids = []
        stats = []
        for sid, item in seg_signal_dict.items():
            acc = np.asarray(item["acc"], dtype=np.float32)
            peak = float(np.max(np.abs(acc)))
            rr = rms(acc)
            ok = peak <= max_abs_peak
            stats.append((sid, peak, rr, ok))
            if ok:
                valid_ids.append(sid)
        if preferred_seg_idx is not None and int(preferred_seg_idx) in valid_ids:
            return int(preferred_seg_idx)
        if valid_ids:
            peaks = [row[1] for row in stats if row[0] in valid_ids]
            target_peak = float(np.median(peaks))
            return int(min(valid_ids, key=lambda sid: abs(seg_signal_dict[sid]["peak"] - target_peak)))
        return int(min(seg_signal_dict.keys(), key=lambda sid: seg_signal_dict[sid]["peak"]))

    def load_reference_from_row(self, row, seg_target_len=4000, seg_idx=10, max_abs_peak=4.0):
        if row is None or "path" not in row.index:
            return None, None
        csv_path = Path(row["path"])
        segs = self.get_resampled_segment_signals(csv_path, seg_target_len=seg_target_len, max_abs_peak=max_abs_peak)
        if len(segs) == 0 and max_abs_peak is not None:
            print(f"[WARN] no segment under max_abs_peak={max_abs_peak} | retry without peak filter: {safe_basename(csv_path)}")
            segs = self.get_resampled_segment_signals(csv_path, seg_target_len=seg_target_len, max_abs_peak=None)
        if len(segs) == 0:
            return None, None
        sid = self.choose_reasonable_segment_id(segs, preferred_seg_idx=seg_idx, max_abs_peak=max_abs_peak)
        if sid is None or sid not in segs:
            return None, None
        return segs[sid], sid

    def build_reference_guide(self, roughness, seg_target_len=4000, seg_idx=10, max_abs_peak=4.0):
        exact_row = self.choose_row_for_roughness(roughness)
        if exact_row is not None:
            ref, sid = self.load_reference_from_row(exact_row, seg_target_len, seg_idx, max_abs_peak)
            if ref is not None:
                info = {"guide_mode": "exact_actual_reference", "requested": float(roughness), "seg_idx": sid}
                return ref, info

        lower_row, upper_row, lower_r, upper_r = self.choose_bracketing_rows(roughness)
        refs = []
        if lower_row is not None:
            low_ref, low_sid = self.load_reference_from_row(lower_row, seg_target_len, seg_idx, max_abs_peak)
            if low_ref is not None:
                refs.append(("low", lower_r, low_ref, low_sid))
        if upper_row is not None:
            up_ref, up_sid = self.load_reference_from_row(upper_row, seg_target_len, seg_idx, max_abs_peak)
            if up_ref is not None:
                refs.append(("up", upper_r, up_ref, up_sid))

        if len(refs) >= 2 and lower_r is not None and upper_r is not None and float(lower_r) != float(upper_r):
            _, _, low_ref, low_sid = refs[0]
            _, _, up_ref, up_sid = refs[1]
            denom = float(upper_r) - float(lower_r)
            alpha = float(np.clip((float(roughness) - float(lower_r)) / denom, 0.0, 1.0))
            guide = {
                "acc": ((1.0 - alpha) * low_ref["acc"] + alpha * up_ref["acc"]).astype(np.float32),
                "force": ((1.0 - alpha) * low_ref["force"] + alpha * up_ref["force"]).astype(np.float32),
                "vel": ((1.0 - alpha) * low_ref["vel"] + alpha * up_ref["vel"]).astype(np.float32),
            }
            info = {"guide_mode": "interpolated_reference", "requested": float(roughness), "alpha": alpha}
            return guide, info

        if len(refs) == 1:
            _, used_r, ref, sid = refs[0]
            info = {"guide_mode": "single_side_reference_fallback", "requested": float(roughness), "seg_idx": sid}
            return ref, info

        near_r, near_row = self.choose_nearest_row(roughness)
        near_ref, near_sid = self.load_reference_from_row(near_row, seg_target_len, seg_idx, max_abs_peak)
        if near_ref is not None:
            info = {"guide_mode": "nearest_actual_reference", "requested": float(roughness), "seg_idx": near_sid}
            return near_ref, info

        raise RuntimeError("reference guide failed: no usable segment found")


# =========================================================
# 3) Realtime reference-guided generator
# =========================================================
class RealtimeReferenceGuidedGenerator:
    def __init__(
        self,
        model,
        x_mean,
        x_std,
        y_mean,
        y_std,
        roughness,
        guide_acc,
        device="cpu",
        input_steps=400,
        output_steps=40,
        ref_blend=0.20,
        mode="safe",
    ):
        self.model = model.eval()
        self.x_mean = torch.tensor(x_mean, dtype=torch.float32, device=device)
        self.x_std = torch.tensor(x_std, dtype=torch.float32, device=device)
        self.y_mean = y_mean
        self.y_std = y_std
        self.roughness = float(roughness)
        self.device = device
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.ref_blend = float(ref_blend)
        self.mode = str(mode).lower()
        self.ref_pos = 0

        self.guide_acc = to_1d_float32(guide_acc)
        if len(self.guide_acc) < self.input_steps:
            self.guide_acc = np.pad(self.guide_acc, (0, self.input_steps - len(self.guide_acc)))

        self.acc_buf = np.zeros(self.input_steps, dtype=np.float32)
        self.acc_buf[:] = self.guide_acc[:self.input_steps]
        self.force_buf = np.zeros(self.input_steps, dtype=np.float32)
        self.vel_buf = np.zeros(self.input_steps, dtype=np.float32)

    def update_roughness(self, roughness):
        self.roughness = float(roughness)

    def get_guide_chunk(self, n):
        ref = self.guide_acc
        length = len(ref)
        if length <= 0:
            return np.zeros(n, dtype=np.float32)
        s = self.ref_pos
        e = s + n
        if e <= length:
            chunk = ref[s:e]
        else:
            chunk = np.concatenate([ref[s:length], ref[0 : e - length]])
        self.ref_pos = (self.ref_pos + n) % length
        return chunk.astype(np.float32)

    def predict(self, force_samples, vel_samples, num_samples=40):
        force_samples = to_1d_float32(force_samples)
        vel_samples = to_1d_float32(vel_samples)
        num_samples = int(num_samples)
        out_final = []

        while len(out_final) < num_samples:
            need = min(self.output_steps, num_samples - len(out_final))
            
            f_val = float(force_samples[min(len(out_final), len(force_samples) - 1)])
            v_val = float(vel_samples[min(len(out_final), len(vel_samples) - 1)])
            
            self.force_buf = np.roll(self.force_buf, -self.output_steps)
            self.force_buf[-self.output_steps:] = f_val
            self.vel_buf = np.roll(self.vel_buf, -self.output_steps)
            self.vel_buf[-self.output_steps:] = v_val

            rough_val = normalize_roughness_value(self.roughness)
            X = np.stack([self.acc_buf, self.force_buf, self.vel_buf, 
                          np.full(self.input_steps, rough_val, dtype=np.float32)], axis=0)
            X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)
            # ch0~2만 정규화, ch3(roughness 0~1)는 FiLM을 위해 원본 그대로 유지
            Xn = X_tensor.clone()
            Xn[:, :3, :] = (X_tensor[:, :3, :] - self.x_mean) / (self.x_std + 1e-8)

            with torch.no_grad():
                pred_n = self.model(Xn).cpu().numpy()[0]

            pred = pred_n * self.y_std + self.y_mean
            pred = pred[:self.output_steps]

            if self.mode == "pure":
                final = pred
            else:
                guide = self.get_guide_chunk(self.output_steps)
                final = (1.0 - self.ref_blend) * pred + self.ref_blend * guide

            self.acc_buf = np.roll(self.acc_buf, -self.output_steps)
            self.acc_buf[-self.output_steps:] = final

            out_final.extend(final[:need].tolist())

        return np.asarray(out_final, dtype=np.float32), None

    def override_last_output(self, signal):
        """post-processing(gate 등) 적용 후 신호를 acc_buf에 반영.
        gate로 줄어든 출력이 다음 예측의 입력에도 반영되도록 한다."""
        n = min(len(signal), self.input_steps)
        self.acc_buf[-n:] = np.asarray(signal[:n], dtype=np.float32)

# =========================================================
# 5) Live plot mode
# =========================================================
def run_live_plot(args):
    """
    --live-plot 모드: 슬라이더로 roughness/force/velocity를 실시간 조절하면서
    모델 가속도 출력을 rolling window로 시각화.

    실행 예:
      python -m scripts.realtime --live-plot \\
          --pt-path pt_files/runs/20260612-003/best_model.pt \\
          --cache-path pt_files/inference_cache_allinone.npz
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider

    device  = "cpu"
    win_samp = int(args.plot_window)
    SR       = 8000

    # ── 공유 상태 (슬라이더 콜백에서 갱신) ──────────────────────────────────
    state = {
        "roughness": float(args.roughness_val),
        "force":     float(args.force_val),
        "vel":       float(args.vel_val),
        "reload":    False,   # roughness 변경 시 guide 재로드 필요
    }

    print("[LIVE] loading cache …")
    cache = InferenceCache(args.cache_path)
    pt_path = args.pt_path
    onnx_path = getattr(args, "onnx_path", None)
    if onnx_path is not None and pt_path == "pt_files/best_model_light.pt":
        candidate = Path(onnx_path).parent / "best_model.pt"
        if candidate.exists():
            pt_path = str(candidate)
    model, x_mean, x_std, y_mean, y_std = load_model_from_pt(
        pt_path, device=device, in_ch=3,
        output_steps=args.output_steps,
        onnx_path=onnx_path,
    )

    def _make_generator(roughness):
        guide, info = cache.build_reference_guide(
            roughness=roughness,
            seg_target_len=args.seg_target_len,
            seg_idx=args.seg_idx,
            max_abs_peak=args.max_abs_peak,
        )
        print(f"[LIVE] guide rebuilt  R={roughness:.0f}  mode={info.get('guide_mode')}")
        return RealtimeReferenceGuidedGenerator(
            model=model, x_mean=x_mean, x_std=x_std,
            y_mean=y_mean, y_std=y_std,
            roughness=roughness, guide_acc=guide["acc"],
            device=device, input_steps=args.input_steps,
            output_steps=args.output_steps,
            ref_blend=args.ref_blend, mode=args.mode,
        )

    rt_holder = [_make_generator(state["roughness"])]

    # ── 레이아웃: 위=파형+RMS, 아래=슬라이더 3개 ─────────────────────────────
    fig = plt.figure(figsize=(13, 7.5))
    fig.patch.set_facecolor("#1e1e2e")

    ax_acc = fig.add_axes([0.07, 0.45, 0.88, 0.46])   # 파형
    ax_rms = fig.add_axes([0.07, 0.28, 0.88, 0.12])   # RMS history

    ax_sl_r = fig.add_axes([0.15, 0.17, 0.72, 0.03])  # roughness 슬라이더
    ax_sl_f = fig.add_axes([0.15, 0.11, 0.72, 0.03])  # force 슬라이더
    ax_sl_v = fig.add_axes([0.15, 0.05, 0.72, 0.03])  # velocity 슬라이더

    for ax in (ax_acc, ax_rms):
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555577")

    # ── 파형 플롯 ────────────────────────────────────────────────────────────
    acc_buf = deque([0.0] * win_samp, maxlen=win_samp)
    t_ms    = np.arange(win_samp) / SR * 1000.0
    (line_acc,) = ax_acc.plot(t_ms, list(acc_buf), lw=0.6, color="#ff9d3a", alpha=0.9)
    ax_acc.set_ylim(-2.0, 2.0)
    ax_acc.set_xlim(0, t_ms[-1])
    ax_acc.axhline(0, lw=0.4, color="#888899", ls="--")
    ax_acc.set_ylabel("Acceleration (m/s²)", color="white", fontsize=9)
    ax_acc.set_xlabel("ms", color="white", fontsize=8)
    ax_acc.grid(True, lw=0.3, alpha=0.35, color="#444466")
    title_txt = ax_acc.set_title("", color="white", fontsize=10)

    # ── RMS 이력 ─────────────────────────────────────────────────────────────
    rms_hist_len = 200
    rms_history  = deque([0.0] * rms_hist_len, maxlen=rms_hist_len)
    (line_rms,)  = ax_rms.plot(np.arange(rms_hist_len), list(rms_history),
                               lw=1.0, color="#66ccff")
    ax_rms.set_ylim(0, 1.5)
    ax_rms.set_xlim(0, rms_hist_len)
    ax_rms.set_ylabel("RMS", color="white", fontsize=8)
    ax_rms.grid(True, lw=0.3, alpha=0.35, color="#444466")
    tgt_line = ax_rms.axhline(
        roughness_to_target_rms(state["roughness"], velocity=state["vel"]),
        lw=1.2, color="#ff6666", ls="--", alpha=0.85, label="target RMS"
    )
    ax_rms.legend(fontsize=7, facecolor="#2a2a3e", labelcolor="white",
                  edgecolor="#555577")

    # ── 슬라이더 ─────────────────────────────────────────────────────────────
    sl_style = dict(color="#3a3a5c", track_color="#555577")
    sl_r = Slider(ax_sl_r, "Roughness", 0.0,  100.0, valinit=state["roughness"],
                  valstep=1.0,  **sl_style)
    sl_f = Slider(ax_sl_f, "Force (N)", 0.0,  10.0,  valinit=state["force"],
                  valstep=0.1,  **sl_style)
    sl_v = Slider(ax_sl_v, "Vel (m/s)", 0.0,  0.25,  valinit=state["vel"],
                  valstep=0.005, **sl_style)
    for sl in (sl_r, sl_f, sl_v):
        sl.label.set_color("white")
        sl.valtext.set_color("#ffcc88")

    ROUGHNESS_RELOAD_THRESH = 2.0

    def on_roughness(val):
        new_r = float(val)
        if abs(new_r - state["roughness"]) >= ROUGHNESS_RELOAD_THRESH:
            state["reload"] = True
        state["roughness"] = new_r

    def on_force(val):
        state["force"] = float(val)

    def on_vel(val):
        state["vel"] = float(val)

    sl_r.on_changed(on_roughness)
    sl_f.on_changed(on_force)
    sl_v.on_changed(on_vel)

    # ── 애니메이션 업데이트 ───────────────────────────────────────────────────
    def update(_frame):
        r = state["roughness"]
        f = state["force"]
        v = state["vel"]

        if state["reload"]:
            state["reload"] = False
            rt_holder[0] = _make_generator(r)
            acc_buf.clear()
            acc_buf.extend([0.0] * win_samp)
            rms_history.clear()
            rms_history.extend([0.0] * rms_hist_len)
        else:
            rt_holder[0].update_roughness(r)

        force_arr = np.full(args.output_steps, f, dtype=np.float32)
        vel_arr   = np.full(args.output_steps, v, dtype=np.float32)
        acc, _    = rt_holder[0].predict(force_arr, vel_arr, num_samples=args.output_steps)

        if not args.no_enhance_roughness:
            acc = enhance_acc_by_roughness(acc, r, rng=rng)
        if not args.no_output_limit:
            acc = apply_common_output_limit(acc, r, velocity=v)
        if not args.no_force_gate:
            acc = force_velocity_gate(acc, f, v)
        rt_holder[0].override_last_output(acc)

        acc_buf.extend(acc.tolist())
        data    = np.asarray(acc_buf, dtype=np.float32)
        cur_rms = float(np.sqrt(np.mean(data ** 2)))
        tgt     = roughness_to_target_rms(r, velocity=v)

        line_acc.set_ydata(data)
        rms_history.append(cur_rms)
        line_rms.set_ydata(list(rms_history))
        tgt_line.set_ydata([tgt, tgt])

        title_txt.set_text(
            f"Live Gen  |  R={r:.0f}  F={f:.2f} N  V={v:.4f} m/s  "
            f"RMS={cur_rms:.3f}  target={tgt:.3f}  mode={args.mode}"
        )
        return line_acc, line_rms, tgt_line, title_txt

    ani = animation.FuncAnimation(   # noqa: F841
        fig, update, interval=args.plot_interval,
        blit=False, cache_frame_data=False,
    )
    print("[LIVE] window open — 슬라이더로 R/F/V 조절, 창 닫으면 종료")
    plt.show()


# =========================================================
# 6) Offline & Socket server
# =========================================================

def run_offline_generation(args):
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[OFFLINE] Starting generation for 100 iterations (4000 samples) on {device}...")

    cache = InferenceCache(args.cache_path)
    pt_path = args.pt_path
    onnx_path = getattr(args, "onnx_path", None)
    if onnx_path is not None and pt_path == "pt_files/best_model_light.pt":
        candidate = Path(onnx_path).parent / "best_model.pt"
        if candidate.exists():
            pt_path = str(candidate)
    model, x_mean, x_std, y_mean, y_std = load_model_from_pt(
        pt_path,
        device=device,
        in_ch=3,
        output_steps=args.output_steps,
        onnx_path=onnx_path,
        use_compile=getattr(args, "use_compile", False),
    )

    test_roughness = 66.0
    guide, guide_info = cache.build_reference_guide(
        roughness=test_roughness,
        seg_target_len=args.seg_target_len,
        seg_idx=args.seg_idx,
        max_abs_peak=args.max_abs_peak,
    )

    rt = RealtimeReferenceGuidedGenerator(
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        roughness=test_roughness,
        guide_acc=guide["acc"],
        device=device,
        input_steps=args.input_steps,
        output_steps=args.output_steps,
        ref_blend=args.ref_blend,
        mode=args.mode,
    )

    test_force = 1.96
    test_speed = 0.067
    iterations = 100
    num_samples_per_iter = args.output_steps
    
    raw_model_outputs = []
    filtered_signals = []

    for i in range(iterations):
        force_arr = np.full(num_samples_per_iter, test_force, dtype=np.float32)
        vel_arr = np.full(num_samples_per_iter, test_speed, dtype=np.float32)
        
        acc, _ = rt.predict(force_arr, vel_arr, num_samples=num_samples_per_iter)
        raw_model_outputs.extend(acc.tolist())

        processed_acc = acc.copy()
        if not args.no_enhance_roughness:
            processed_acc = enhance_acc_by_roughness(processed_acc, roughness=test_roughness, rng=rng)
        if args.apply_output_limit:
            processed_acc = apply_common_output_limit(processed_acc, roughness=test_roughness, velocity=test_speed)

        wave_data = acc_to_uint16_wave(
            processed_acc,
            device_num=0,
            channel_num=0,
        )
        filtered_signals.extend(wave_data.tolist())

    raw_path = f"test_raw_model_output_{test_roughness}_{test_force}_{test_speed}.txt"
    filt_path = "test_filtered_signal.txt"
    
    with open(raw_path, "w") as f:
        f.write("\n".join(map(str, raw_model_outputs)))
    
    with open(filt_path, "w") as f:
        f.write("\n".join(map(str, filtered_signals)))

    print(f"[OFFLINE] Done. Saved to {raw_path} and {filt_path}")


def recv_exact(sock, nbytes):
    data = b""
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def get_response_profile(user_id):
    if int(user_id) % 2 == 1:
        return {"name": "5Glove", "num_samples": 80}
    return {"name": "KrissGlove", "num_samples": 40}


def start_socket_server(args):
    if args.save_test_signal:
        run_offline_generation(args)
        return

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    cache = InferenceCache(args.cache_path)

    # --onnx-path 지정 시 같은 디렉토리의 best_model.pt를 pt_path로 자동 추론
    pt_path = args.pt_path
    onnx_path = getattr(args, "onnx_path", None)
    if onnx_path is not None and pt_path == "pt_files/best_model_light.pt":
        candidate = Path(onnx_path).parent / "best_model.pt"
        if candidate.exists():
            pt_path = str(candidate)
            print(f"[ONNX] auto-inferred pt_path: {pt_path}")
        else:
            print(f"[WARN] pt_path not found: {candidate}. ONNX export will fail if .onnx doesn't exist.")

    model, x_mean, x_std, y_mean, y_std = load_model_from_pt(
        pt_path, device=device, in_ch=3, output_steps=args.output_steps,
        onnx_path=onnx_path,
        use_compile=getattr(args, "use_compile", False),
    )

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.host, args.port))
    server_socket.listen(1)

    print(f"[SOCKET] Waiting for Unity on {args.host}:{args.port}...")

    try:
        while True:
            client_socket, addr = server_socket.accept()
            print(f"[SOCKET] Connected by {addr}")
            rt = None
            current_roughness = None
            try:
                while True:
                    t_start = time.perf_counter()
                    data = recv_exact(client_socket, 28)
                    if data is None: break
                    roughness, force_mag, vx, vy, vz, user_id, fingerIdx = struct.unpack("<7f", data)
                    speed = float(math.sqrt(vx**2 + vy**2 + vz**2))
                    profile = get_response_profile(user_id)
                    num_samples = int(profile["num_samples"])

                    if rt is None or abs(roughness - current_roughness) > args.roughness_change_threshold:
                        guide, guide_info = cache.build_reference_guide(roughness=roughness, seg_target_len=args.seg_target_len, seg_idx=args.seg_idx, max_abs_peak=args.max_abs_peak)
                        rt = RealtimeReferenceGuidedGenerator(model=model, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, roughness=roughness, guide_acc=guide["acc"], device=device, output_steps=args.output_steps, ref_blend=args.ref_blend, mode=args.mode)
                        current_roughness = roughness
                    else:
                        rt.update_roughness(roughness)

                    t_pred = time.perf_counter()
                    acc, _ = rt.predict(np.full(num_samples, force_mag), np.full(num_samples, speed), num_samples=num_samples)
                    t_pred_end = time.perf_counter()

                    if not args.no_enhance_roughness: acc = enhance_acc_by_roughness(acc, roughness, rng=rng)
                    if not args.no_output_limit: acc = apply_common_output_limit(acc, roughness, velocity=speed)
                    if not args.no_force_gate: acc = force_velocity_gate(acc, force_mag, speed)
                    rt.override_last_output(acc)
                    wave_data = acc_to_uint16_wave(acc, device_num=int(user_id)%2, channel_num=int(fingerIdx))
                    client_socket.sendall(struct.pack(f"<{num_samples}H", *wave_data))

                    t_end = time.perf_counter()
                    print(f"[TIME] samples={num_samples} | predict={( t_pred_end - t_pred)*1000:.2f}ms | total={( t_end - t_start)*1000:.2f}ms | budget={(num_samples/8000)*1000:.2f}ms")
            except Exception as e: print(f"[ERROR] {e}")
            finally: client_socket.close()
    finally: server_socket.close()

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--pt-path", type=str, default="pt_files/best_model_light.pt")
    p.add_argument("--cache-path", type=str, default="pt_files/inference_cache_allinone.npz")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=65432)
    p.add_argument("--mode", type=str, default="safe")
    p.add_argument("--ref-blend", type=float, default=0.50)
    p.add_argument("--input-steps", type=int, default=400)
    p.add_argument("--output-steps", type=int, default=40)
    p.add_argument("--seg-target-len", type=int, default=4000)
    p.add_argument("--seg_idx", type=int, default=10)
    p.add_argument("--max-abs-peak", type=float, default=4.0)
    p.add_argument("--roughness-change-threshold", type=float, default=0.25)
    p.add_argument("--no-enhance-roughness", action="store_true")
    p.add_argument("--no-output-limit", action="store_true")
    p.add_argument("--no-force-gate", action="store_true",
                   help="force_velocity_gate 비활성화 (모델이 학습한 gating 효과만 평가할 때 사용)")
    p.add_argument("--save-test-signal", action="store_true")
    p.add_argument("--onnx-path", type=str, default=None,
                   help="ONNX 파일 경로. 지정하면 ONNX Runtime으로 추론 (3-5x 빠름). "
                        "파일이 없으면 자동 export. 예: pt_files/runs/20260610-004/best_model.onnx")
    p.add_argument("--use-compile", action="store_true",
                   help="torch.compile 적용 (PyTorch 2.0+, 약 1.5-2x 빠름)")
    # ── live plot ─────────────────────────────────────────────────────────────
    p.add_argument("--live-plot", action="store_true",
                   help="실시간 가속도 플롯 모드. 소켓 서버 없이 단독 실행.")
    p.add_argument("--roughness-val", type=float, default=58.0,
                   help="[live-plot] 거칠기 값 (0~100, default: 58)")
    p.add_argument("--force-val", type=float, default=1.96,
                   help="[live-plot] 힘 (N, default: 1.96)")
    p.add_argument("--vel-val", type=float, default=0.067,
                   help="[live-plot] 속도 (m/s, default: 0.067)")
    p.add_argument("--plot-window", type=int, default=8000,
                   help="[live-plot] 화면에 표시할 샘플 수 (default: 8000 = 1s)")
    p.add_argument("--plot-interval", type=int, default=20,
                   help="[live-plot] 애니메이션 갱신 주기 ms (default: 20)")
    return p

if __name__ == "__main__":
    _args = build_argparser().parse_args()
    if _args.live_plot:
        run_live_plot(_args)
    else:
        start_socket_server(_args)