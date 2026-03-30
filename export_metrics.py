# export_metrics.py
"""
Helper script to export model comparison metrics from your Colab notebook
into the JSON format expected by the Fraud Detection Dashboard.

USAGE (in a Colab cell, after training all models):
    # 1. Upload this script to Colab or paste its contents into a cell.
    # 2. Make sure the following variables are in scope:
    #      xgb_clf       — trained XGBoost booster / sklearn wrapper
    #      rf_clf        — trained RandomForestClassifier
    #      lr_clf        — trained LogisticRegression
    #      lgbm_clf      — trained LGBMClassifier
    #      mlp_model     — trained ImprovedMLP (torch.nn.Module)
    #      X_test        — numpy array of test features (already scaled)
    #      y_test        — numpy array of true labels
    #      scaler        — fitted StandardScaler (for MLP inference)
    #      ART_DIR       — path string where artifacts are stored, e.g.
    #                      "/content/drive/MyDrive/Colab_Notebooks/fraud_app/results"
    # 3. Run:  exec(open("export_metrics.py").read())
"""

import json
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configure output directory (edit if needed)
# ---------------------------------------------------------------------------
_OUT_DIR = Path(ART_DIR)  # type: ignore[name-defined]  # noqa: F821 - set by notebook
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
try:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
    )
    import torch
    import torch.nn as nn
except ImportError as _e:
    raise ImportError(f"Required package missing: {_e}") from _e


def _mc_predict(model, X_tensor, T=20):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    probs = []
    with torch.no_grad():
        for _ in range(T):
            probs.append(torch.sigmoid(model(X_tensor)).cpu().numpy())
    model.eval()
    return np.stack(probs).mean(axis=0).reshape(-1)


def _get_probs(clf, X, name):
    """Return probability scores for the positive (fraud) class."""
    if name == "MLP Static":
        Xt = torch.tensor(X, dtype=torch.float32)
        return _mc_predict(clf, Xt)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    # XGBoost booster
    import xgboost as xgb
    return clf.predict(xgb.DMatrix(X))


def _recall_at_fpr(fpr_arr, tpr_arr, target_fpr=0.05):
    """Interpolate TPR (recall) at the given FPR threshold."""
    idx = np.searchsorted(fpr_arr, target_fpr, side="right")
    if idx == 0:
        return float(tpr_arr[0])
    if idx >= len(fpr_arr):
        return float(tpr_arr[-1])
    # linear interpolation
    fpr0, fpr1 = fpr_arr[idx - 1], fpr_arr[idx]
    tpr0, tpr1 = tpr_arr[idx - 1], tpr_arr[idx]
    alpha = (target_fpr - fpr0) / max(fpr1 - fpr0, 1e-12)
    return float(tpr0 + alpha * (tpr1 - tpr0))


def _fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a [0, 1] metric value as a percentage string for display.

    Internal metrics are stored in [0, 1]; this helper is used only for
    printed/logged summaries that a human reads.

    Examples:
        _fmt_pct(0.8602) -> "86.02%"
        _fmt_pct(0.5)    -> "50.00%"  (threshold 0.5 == 50%)
    """
    return f"{value * 100:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Collect models — edit keys / values to match your notebook variables
# ---------------------------------------------------------------------------
MODELS = {
    "XGBoost": xgb_clf,          # type: ignore[name-defined]  # noqa: F821
    "Random Forest": rf_clf,     # type: ignore[name-defined]  # noqa: F821
    "Logistic Regression": lr_clf,  # type: ignore[name-defined]  # noqa: F821
    "LightGBM": lgbm_clf,        # type: ignore[name-defined]  # noqa: F821
    "MLP Static": mlp_model,     # type: ignore[name-defined]  # noqa: F821
}

print("Computing metrics for all models …")

metrics_out = {}
roc_out = {}
pr_out = {}
cm_out = {}

for model_name, clf in MODELS.items():
    print(f"  {model_name} …", end=" ")
    probs = _get_probs(clf, X_test, model_name)  # type: ignore[name-defined]  # noqa: F821

    roc_auc = float(roc_auc_score(y_test, probs))  # type: ignore[name-defined]  # noqa: F821
    pr_auc = float(average_precision_score(y_test, probs))  # type: ignore[name-defined]  # noqa: F821

    fpr_arr, tpr_arr, _ = roc_curve(y_test, probs)  # type: ignore[name-defined]  # noqa: F821
    recall_5fpr = _recall_at_fpr(fpr_arr, tpr_arr, target_fpr=0.05)

    threshold = 0.5
    preds = (probs >= threshold).astype(int)
    f1 = float(f1_score(y_test, preds, zero_division=0))  # type: ignore[name-defined]  # noqa: F821
    prec = float(precision_score(y_test, preds, zero_division=0))  # type: ignore[name-defined]  # noqa: F821
    rec = float(recall_score(y_test, preds, zero_division=0))  # type: ignore[name-defined]  # noqa: F821

    metrics_out[model_name] = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall_at_5pct_fpr": recall_5fpr,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
    }

    # ROC curve (downsample to ≤ 200 points for compact JSON)
    step = max(1, len(fpr_arr) // 200)
    roc_out[model_name] = {
        "fpr": fpr_arr[::step].tolist(),
        "tpr": tpr_arr[::step].tolist(),
    }

    # PR curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, probs)  # type: ignore[name-defined]  # noqa: F821
    step_pr = max(1, len(prec_arr) // 200)
    pr_out[model_name] = {
        "recall": rec_arr[::step_pr].tolist(),
        "precision": prec_arr[::step_pr].tolist(),
    }

    # Confusion matrix at threshold = 0.5
    cm = confusion_matrix(y_test, preds).tolist()  # type: ignore[name-defined]  # noqa: F821
    cm_out[model_name] = {"matrix": cm, "labels": ["Legitimate", "Fraud"]}

    print(f"ROC-AUC={_fmt_pct(roc_auc)}  PR-AUC={_fmt_pct(pr_auc)}")

# --- Add Ensemble ---
print("  Ensemble …", end=" ")
xgb_probs = _get_probs(xgb_clf, X_test, "XGBoost")  # type: ignore[name-defined]  # noqa: F821
mlp_probs = _get_probs(mlp_model, X_test, "MLP Static")  # type: ignore[name-defined]  # noqa: F821
ens_probs = 0.6 * xgb_probs + 0.4 * mlp_probs

ens_roc_auc = float(roc_auc_score(y_test, ens_probs))  # type: ignore[name-defined]  # noqa: F821
ens_pr_auc = float(average_precision_score(y_test, ens_probs))  # type: ignore[name-defined]  # noqa: F821

ens_fpr, ens_tpr, _ = roc_curve(y_test, ens_probs)  # type: ignore[name-defined]  # noqa: F821
ens_recall_5fpr = _recall_at_fpr(ens_fpr, ens_tpr)

ens_preds = (ens_probs >= 0.5).astype(int)
ens_f1 = float(f1_score(y_test, ens_preds, zero_division=0))  # type: ignore[name-defined]  # noqa: F821
ens_prec = float(precision_score(y_test, ens_preds, zero_division=0))  # type: ignore[name-defined]  # noqa: F821
ens_rec = float(recall_score(y_test, ens_preds, zero_division=0))  # type: ignore[name-defined]  # noqa: F821

metrics_out["Ensemble"] = {
    "roc_auc": ens_roc_auc,
    "pr_auc": ens_pr_auc,
    "recall_at_5pct_fpr": ens_recall_5fpr,
    "f1_score": ens_f1,
    "precision": ens_prec,
    "recall": ens_rec,
}

step_ens = max(1, len(ens_fpr) // 200)
roc_out["Ensemble"] = {"fpr": ens_fpr[::step_ens].tolist(), "tpr": ens_tpr[::step_ens].tolist()}

ens_prec_arr, ens_rec_arr, _ = precision_recall_curve(y_test, ens_probs)  # type: ignore[name-defined]  # noqa: F821
step_ens_pr = max(1, len(ens_prec_arr) // 200)
pr_out["Ensemble"] = {
    "recall": ens_rec_arr[::step_ens_pr].tolist(),
    "precision": ens_prec_arr[::step_ens_pr].tolist(),
}

ens_cm = confusion_matrix(y_test, ens_preds).tolist()  # type: ignore[name-defined]  # noqa: F821
cm_out["Ensemble"] = {"matrix": ens_cm, "labels": ["Legitimate", "Fraud"]}

print(f"ROC-AUC={_fmt_pct(ens_roc_auc)}  PR-AUC={_fmt_pct(ens_pr_auc)}")

# ---------------------------------------------------------------------------
# Write JSON files
# ---------------------------------------------------------------------------
def _save(data, filename):
    path = _OUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


_save({"_sample": False, "models": metrics_out}, "metrics_comparison.json")
_save({"_sample": False, "models": roc_out}, "roc_curves.json")
_save({"_sample": False, "models": pr_out}, "pr_curves.json")
_save({"_sample": False, "models": cm_out}, "confusion_matrices.json")

# ---------------------------------------------------------------------------
# Drift analysis (optional — requires streaming_results dict in scope)
# ---------------------------------------------------------------------------
# If you ran the streaming simulation and have per-window metrics, export them:
#
#   streaming_results = {
#       "XGBoost Adaptive": [{"window": 1, "roc_auc": 0.78, "pr_auc": 0.18}, ...],
#       "XGBoost Static":   [...],
#       "MLP Static":       [...],
#       "Ensemble":         [...],
#   }
#   adwin_events = [{"window": 3, "description": "drift detected"}, ...]
#
try:
    _streaming_results = streaming_results  # type: ignore[name-defined]  # noqa: F821
    try:
        _adwin_events = adwin_events  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        _adwin_events = []
    drift_payload = {
        "_sample": False,
        "time_windows": _streaming_results,
        "drift_events": _adwin_events,
    }
    _save(drift_payload, "drift_analysis.json")
except NameError:
    pass  # streaming_results not in scope - skip drift analysis export

print("\n✅ All metrics exported. Copy the results/ folder to your local repo and restart the Streamlit app.")
