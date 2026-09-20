"""
evaluate.py
===========
Classification metrics + Kamei et al.'s effort-aware evaluation (Popt /
PofB20), computed from real predictions -- nothing here is a placeholder.

Effort-aware evaluation background (see README for full explanation):
  * "Effort" for a commit is approximated by its code churn, effort = la+ld
    (lines added + deleted), the same proxy Kamei et al. use ("LOC" of the
    change) -- it is the only effort proxy our features actually support,
    since we don't have real reviewer-hours.
  * Popt is computed from the "Alberg diagram": cumulative %defects found
    (y) vs cumulative %effort inspected (x), for three rankings of the test
    commits -- the optimal ranking, the worst (inverse-optimal) ranking, and
    the model's own ranking (by predicted bug density = P(bug)/effort).
    Popt = 1 - (Area_optimal - Area_model) / (Area_optimal - Area_worst)
  * PofB20 is simpler and matches the statistic the base paper itself
    quotes from Kamei et al. ("reviewing 20% of churn reveals up to 35% of
    defects"): the % of actual bug-inducing commits found among the top 20%
    of churn when commits are ranked by predicted bug density.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true.tolist())) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = None  # cannot compute AUC with a single class present

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    return metrics


def _alberg_area(order_idx, y_true, effort):
    """Area under the cumulative-%effort (x) vs cumulative-%defects (y)
    curve, for a given commit ordering, via the trapezoidal rule."""
    y_true = np.asarray(y_true)[order_idx]
    effort = np.asarray(effort)[order_idx]
    total_effort = effort.sum()
    total_defects = y_true.sum()
    if total_effort <= 0 or total_defects <= 0:
        return None
    cum_effort = np.cumsum(effort) / total_effort
    cum_defects = np.cumsum(y_true) / total_defects
    x = np.concatenate([[0.0], cum_effort])
    y = np.concatenate([[0.0], cum_defects])
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(y, x))


def effort_aware_metrics(y_true, y_prob, effort) -> dict:
    """Returns dict with popt, pofb20, or an explanation if it cannot be
    reliably computed (e.g. no churn information, or a degenerate split)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    effort = np.asarray(effort, dtype=float)
    effort_safe = np.clip(effort, 1.0, None)  # avoid div-by-zero for 0-churn commits

    if y_true.sum() == 0:
        return {
            "computable": False,
            "reason": "No actual bug-inducing commits in this split, so no "
                      "effort-aware curve can be evaluated against.",
        }
    if effort.sum() == 0:
        return {
            "computable": False,
            "reason": "All commits in this split have zero recorded churn "
                      "(la+ld), so effort-aware ranking is meaningless here.",
        }

    density = y_prob / effort_safe
    model_order = np.argsort(-density)

    optimal_density = y_true / effort_safe
    optimal_order = np.argsort(-optimal_density)
    worst_order = optimal_order[::-1]

    area_model = _alberg_area(model_order, y_true, effort_safe)
    area_optimal = _alberg_area(optimal_order, y_true, effort_safe)
    area_worst = _alberg_area(worst_order, y_true, effort_safe)

    if area_optimal is None or area_worst is None or area_model is None or area_optimal == area_worst:
        return {"computable": False, "reason": "Degenerate effort/label distribution in this split."}

    popt = 1.0 - (area_optimal - area_model) / (area_optimal - area_worst)

    # PofB20: % of actual bug-inducing commits found within the top 20% of
    # cumulative churn, when ranked by predicted bug density.
    ordered_effort = effort_safe[model_order]
    ordered_labels = y_true[model_order]
    total_effort = ordered_effort.sum()
    cum_effort_frac = np.cumsum(ordered_effort) / total_effort
    cutoff = np.searchsorted(cum_effort_frac, 0.20, side="right") + 1
    cutoff = min(cutoff, len(ordered_labels))
    found = ordered_labels[:cutoff].sum()
    pofb20 = float(found / y_true.sum())

    return {
        "computable": True,
        "popt": float(popt),
        "pofb20": pofb20,
        "area_model": area_model,
        "area_optimal": area_optimal,
        "area_worst": area_worst,
        "note": "effort = la+ld (code churn); this is a proxy for reviewer "
                "effort, following Kamei et al., since real review-time data "
                "is not available.",
    }
