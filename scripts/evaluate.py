"""
scripts/evaluate.py
===================
Step 3 – Post-training evaluation.
  - Test metrics (RMSE / MAE / Corr)
  - Permutation feature importance
  - Saliency heatmap

Usage:
    python -m scripts.evaluate
"""
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUT_DIR, DEVICE, BATCH_SIZE
from src.model import LiteSeq2SeqCNNGRU_AttnPool, SeqDataset
from src.inference import load_model_from_pt


FEATURE_NAMES = ["PastAcc", "Force", "Velocity", "Roughness"]


def predict_numpy(model, Xn, device, batch_size=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch_size):
            xb = torch.tensor(Xn[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def main():
    NPZ_PATH = OUT_DIR / "inference_cache_allinone.npz"
    PT_PATH  = OUT_DIR / "best_model_light_wo71.pt"

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Run preprocess.py first. Expected: {NPZ_PATH}")
    if not PT_PATH.exists():
        raise FileNotFoundError(f"Run train.py first. Expected: {PT_PATH}")

    # ── Load model & data ────────────────────────────────────────────────────
    model, x_mean, x_std, y_mean, y_std, _ = load_model_from_pt(
        PT_PATH, device=DEVICE, in_ch=4, output_steps=40,
    )
    if x_mean.ndim == 1: x_mean = x_mean[:, None]
    if x_std.ndim == 1:  x_std  = x_std[:, None]

    npz     = np.load(NPZ_PATH, allow_pickle=True)
    X_val   = npz["X_val"].astype(np.float32)
    Y_val   = npz["Y_val"].astype(np.float32)
    X_test  = npz["X_test"].astype(np.float32)
    Y_test  = npz["Y_test"].astype(np.float32)

    X_val_n  = (X_val  - x_mean) / x_std
    X_test_n = (X_test - x_mean) / x_std
    Y_val_n  = (Y_val  - y_mean) / y_std
    Y_test_n = (Y_test - y_mean) / y_std

    # ── Test metrics ─────────────────────────────────────────────────────────
    pred_test_n = predict_numpy(model, X_test_n, DEVICE)
    pred_test   = pred_test_n * y_std + y_mean
    true_test   = Y_test_n    * y_std + y_mean

    rmse = float(np.sqrt(np.mean((pred_test - true_test) ** 2)))
    mae  = float(np.mean(np.abs(pred_test - true_test)))
    pf, tf = pred_test.reshape(-1), true_test.reshape(-1)
    corr = float(np.corrcoef(pf, tf)[0, 1]) if np.std(pf) > 0 and np.std(tf) > 0 else np.nan
    print(f"[TEST]  RMSE={rmse:.4f}  MAE={mae:.4f}  Corr={corr:.4f}")

    # ── Permutation importance (validation set) ──────────────────────────────
    base_pred   = predict_numpy(model, X_val_n, DEVICE) * y_std + y_mean
    base_true   = Y_val_n * y_std + y_mean
    base_mse    = float(np.mean((base_pred - base_true) ** 2))

    importance = {}
    for ch, name in enumerate(FEATURE_NAMES):
        X_perm = X_val_n.copy()
        perm   = np.random.permutation(len(X_perm))
        X_perm[:, ch, :] = X_perm[perm, ch, :]
        pred_p = predict_numpy(model, X_perm, DEVICE) * y_std + y_mean
        importance[name] = float(np.mean((pred_p - base_true) ** 2)) - base_mse

    imp_df = pd.DataFrame({
        "Feature": list(importance.keys()),
        "MSE_increase": list(importance.values()),
    })
    print("\n[Permutation Importance]")
    print(imp_df.to_string(index=False))

    # ── Saliency ─────────────────────────────────────────────────────────────
    n_sal = min(256, len(X_val_n))
    x_sal = torch.tensor(X_val_n[:n_sal], dtype=torch.float32,
                         requires_grad=True, device=DEVICE)
    model.train()
    out = model(x_sal)
    out.mean().backward()
    saliency = np.mean(np.abs(x_sal.grad.detach().cpu().numpy()), axis=0)  # [4, T]
    sal_feature = saliency.mean(axis=1)
    model.eval()

    sal_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Mean_saliency": [float(s) for s in sal_feature],
    })
    print("\n[Saliency]")
    print(sal_df.to_string(index=False))

    # ── Plots ────────────────────────────────────────────────────────────────
    # Permutation importance bar
    plt.figure(figsize=(6, 4))
    plt.bar(imp_df["Feature"], imp_df["MSE_increase"])
    plt.title("Permutation feature importance")
    plt.xlabel("Feature"); plt.ylabel("Val MSE increase")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "permutation_importance.png", dpi=150)
    plt.show()

    # Saliency heatmap
    plt.figure(figsize=(12, 4))
    plt.imshow(saliency, aspect="auto", origin="lower")
    plt.yticks([0, 1, 2, 3], FEATURE_NAMES)
    plt.xlabel("Lag index (older → recent)")
    plt.title("Feature activation map (saliency)")
    plt.colorbar(label="Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "saliency_heatmap.png", dpi=150)
    plt.show()

    # Saliency summary bar
    plt.figure(figsize=(6, 4))
    plt.bar(sal_df["Feature"], sal_df["Mean_saliency"])
    plt.title("Average activation by feature")
    plt.xlabel("Feature"); plt.ylabel("Mean saliency")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "saliency_bar.png", dpi=150)
    plt.show()

    # ── Save summary CSV ─────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "Item": [
            "Test RMSE", "Test MAE", "Test Corr",
            *[f"Permutation importance - {n}" for n in FEATURE_NAMES],
            *[f"Mean saliency - {n}" for n in FEATURE_NAMES],
        ],
        "Value": [
            rmse, mae, corr,
            *[importance[n] for n in FEATURE_NAMES],
            *[float(sal_feature[i]) for i in range(4)],
        ],
    })
    summary.to_csv(OUT_DIR / "model_summary.csv", index=False)
    print(f"\n[DONE] Results saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
