"""
scripts/plot_postproc.py  — Post-processing 파이프라인 시각화
사용법:
  python -m scripts.plot_postproc
  python -m scripts.plot_postproc --npz pt_files/inference_cache_allinone.npz
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROUGHNESS_LIST = [5, 12, 23, 45, 58, 66, 100]
SR = 8000
COLORS = plt.cm.plasma(np.linspace(0.05, 0.95, len(ROUGHNESS_LIST)))

# ── post-processing (realtime.py 동일 로직, torch 없이) ─────────────────────
def _ma(x, w):
    if w <= 1: return x.copy()
    return np.convolve(x, np.ones(w)/w, mode="same").astype(np.float32)

def _clip_median(x, ratio=4.5):
    m = np.median(np.abs(x))
    return x if m < 1e-9 else np.clip(x, -ratio*m, ratio*m)

def enhance_acc(acc, roughness, rng=None, smooth_w=5):
    if rng is None: rng = np.random.default_rng()
    x = np.asarray(acc, dtype=np.float32).copy()
    r = np.clip(float(roughness)/100.0, 0.0, 1.0)
    slow = _ma(x, smooth_w); fast = x - slow
    lf = 0.50 + 1.50*r   # 0.50 -> 2.00
    hf = 2.00 - 1.50*r   # 2.00 -> 0.50
    x = slow*lf + fast*hf
    nr = rng.normal(0,1,x.shape).astype(np.float32)
    nw = max(1, int(1+9*r))
    nlf = _ma(nr, nw); nhf = nr - nlf
    x = x + nlf*(0.008*r) + nhf*(0.008*(1-r))
    return _clip_median(x, 5.0).astype(np.float32)

def target_rms(roughness, velocity=0.05, rms_min=0.18, rms_max=0.90, gamma=0.60):
    r = np.clip(float(roughness)/100.0, 0.0, 1.0)
    base = rms_min + (rms_max - rms_min) * (r**gamma)
    v = np.clip(float(velocity), 0.0, 0.25)
    vc = 0.45 * (r**1.5)
    vn = np.clip((v - 0.01)/0.14, 0.0, 1.0)
    return float(base * (1.0 + vc*vn))

def limit(signal, roughness, velocity=0.05):
    x = np.asarray(signal, dtype=np.float32).copy()
    x = x - np.mean(x)
    x = _clip_median(x, 4.5)
    cur = float(np.sqrt(np.mean(x**2)) + 1e-8)
    sc = float(np.clip(target_rms(roughness, velocity)/cur, 0.20, 3.0))
    return (x * sc).astype(np.float32)

# ── helpers ───────────────────────────────────────────────────────────────────
def rms(x): return float(np.sqrt(np.mean(np.asarray(x)**2)))

def spectrum(sig):
    freqs = np.fft.rfftfreq(len(sig), d=1.0/SR)
    mag   = np.abs(np.fft.rfft(sig.astype(np.float32)))
    return freqs, mag

def load_signals(npz_path, seed=42):
    import pandas as pd
    npz = np.load(npz_path, allow_pickle=True)
    cols = [str(c) for c in npz["window_meta_columns"].tolist()]
    wm   = pd.DataFrame({c: npz[f"window_meta__{c}"] for c in cols})
    rng  = np.random.default_rng(seed)
    out  = {}
    for sp in ["train","val","test"]:
        if f"Y_{sp}" not in npz: continue
        Y   = npz[f"Y_{sp}"]
        sub = wm[wm["split"]==sp]
        for r in ROUGHNESS_LIST:
            if r in out: continue
            if "roughness" not in sub.columns: continue
            rows = sub[sub["roughness"]==r]
            if len(rows)==0: continue
            idx  = int(rows.index[rng.integers(0, len(rows))])
            ai   = int(rows.loc[idx,"arr_idx"]) if "arr_idx" in rows.columns else rng.integers(0,len(Y))
            out[r] = Y[ai].astype(np.float32)
    # fallback synthetic
    for r in ROUGHNESS_LIST:
        if r not in out:
            t = np.arange(40)/SR
            out[r] = (0.5*np.sin(2*np.pi*200*t) + 0.2*rng.normal(0,1,40)).astype(np.float32)
    return out

# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = Path(args.npz) if args.npz else Path("pt_files/inference_cache_allinone.npz")

    if npz_path.exists():
        print(f"[NPZ] {npz_path}")
        base = load_signals(npz_path)
    else:
        print("[NPZ] 없음 → 합성 신호 사용")
        rng = np.random.default_rng(42)
        t = np.arange(40)/SR
        sig = (0.5*np.sin(2*np.pi*300*t)+0.2*np.sin(2*np.pi*800*t)+0.1*rng.normal(0,1,40)).astype(np.float32)
        base = {r: sig.copy() for r in ROUGHNESS_LIST}

    rng2 = np.random.default_rng(0)
    VEL  = 0.056

    raw, eq, lim = {}, {}, {}
    for r in ROUGHNESS_LIST:
        raw[r] = base[r].copy()
        eq[r]  = enhance_acc(base[r].copy(), r, rng=rng2)
        lim[r] = limit(eq[r].copy(), r, velocity=VEL)

    t_ms = np.arange(40)*1000.0/SR

    # ── Fig 1: 파형 3단 비교 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 7, figsize=(18, 7), sharey="row", sharex=True)
    fig.suptitle(f"Post-processing 단계별 파형  (vel={VEL} m/s)", fontsize=12)
    labels = ["① Raw (레퍼런스/모델 출력)", "② + Spectral EQ", "③ + RMS 스케일링"]
    for row, dct in enumerate([raw, eq, lim]):
        for ci, r in enumerate(ROUGHNESS_LIST):
            ax = axes[row, ci]
            ax.plot(t_ms, dct[r], color=COLORS[ci], lw=1.0)
            ax.axhline(0, color="gray", lw=0.4, ls="--")
            if row == 0: ax.set_title(f"R={r}", fontsize=9)
            if ci == 0:  ax.set_ylabel(labels[row], fontsize=7.5)
            ax.text(0.97, 0.94, f"RMS={rms(dct[r]):.3f}", transform=ax.transAxes,
                    fontsize=7, ha="right", va="top", color=COLORS[ci])
            ax.tick_params(labelsize=6); ax.grid(True, lw=0.3, alpha=0.4)
    axes[-1, 3].set_xlabel("Time (ms)", fontsize=8)
    plt.tight_layout()
    p1 = out_dir/"postproc_waveform.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVE] {p1}")

    # ── Fig 2: 스펙트럼 비교 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Post-processing 단계별 스펙트럼", fontsize=12)
    for ax, (title, dct) in zip(axes, [("① Raw", raw), ("② + Spectral EQ", eq), ("③ + RMS 스케일링", lim)]):
        for ci, r in enumerate(ROUGHNESS_LIST):
            freqs, mag = spectrum(dct[r])
            ax.plot(freqs, mag, color=COLORS[ci], lw=1.3, label=f"R={r}", alpha=0.85)
        ax.set_title(title, fontsize=10); ax.set_xlabel("Hz", fontsize=8); ax.set_ylabel("Magnitude", fontsize=8)
        ax.set_xlim(0, SR//2); ax.grid(True, lw=0.3, alpha=0.4); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    p2 = out_dir/"postproc_spectrum.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVE] {p2}")

    # ── Fig 3: RMS 요약 & 게인 파라미터 ─────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)
    fig.suptitle("Roughness별 RMS & 게인 파라미터", fontsize=12)

    ax1 = fig.add_subplot(gs[0])
    x = np.arange(7); w = 0.26
    ax1.bar(x-w, [rms(raw[r]) for r in ROUGHNESS_LIST], w, label="Raw",           color="#94A3B8", alpha=0.85)
    ax1.bar(x,   [rms(eq[r])  for r in ROUGHNESS_LIST], w, label="+Spectral EQ",  color="#0EA5E9", alpha=0.85)
    ax1.bar(x+w, [rms(lim[r]) for r in ROUGHNESS_LIST], w, label="+RMS scaling",  color="#10B981", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels([f"R={r}" for r in ROUGHNESS_LIST], fontsize=8)
    ax1.set_ylabel("RMS", fontsize=9); ax1.set_title("단계별 RMS", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(axis="y", lw=0.3, alpha=0.5)

    ax2 = fig.add_subplot(gs[1])
    rv  = np.linspace(0, 1, 200)
    ax2.plot(rv*100, 0.50+1.50*rv, color="#F59E0B", lw=2, label="LF gain (slow)")
    ax2.plot(rv*100, 2.00-1.50*rv, color="#6366F1", lw=2, label="HF gain (fast)")
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    for r in ROUGHNESS_LIST: ax2.axvline(r, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax2.set_xlabel("Roughness", fontsize=9); ax2.set_ylabel("Gain", fontsize=9)
    ax2.set_title("Spectral EQ 게인 커브", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(lw=0.3, alpha=0.4)

    ax3 = fig.add_subplot(gs[2])
    for vel, vc, lbl in [(0.01,"#94A3B8","v=0.01"), (0.056,"#0EA5E9","v=0.056"), (0.15,"#EF4444","v=0.15")]:
        targets = [target_rms(r, velocity=vel) for r in ROUGHNESS_LIST]
        ax3.plot(ROUGHNESS_LIST, targets, "o-", color=vc, lw=1.5, ms=5, label=lbl)
    ax3.set_xlabel("Roughness", fontsize=9); ax3.set_ylabel("Target RMS", fontsize=9)
    ax3.set_title("Target RMS 커브", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(lw=0.3, alpha=0.4)

    plt.tight_layout()
    p3 = out_dir/"postproc_summary.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVE] {p3}")
    print(f"\n[DONE] → {out_dir}")

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz",     default=None)
    p.add_argument("--out-dir", default="pt_files/analysis/postproc_vis")
    return p.parse_args()

if __name__ == "__main__":
    main(_args())
