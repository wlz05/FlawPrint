# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from engine.conformal import validate_labels
from engine.results import prediction_rows, score_rows

def valid_probabilities(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != labels.shape or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Invalid evaluation probabilities")
    return result

def calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    labels = validate_labels(labels)
    probabilities = valid_probabilities(probabilities, labels)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("Bin count must be positive")
    indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    result = 0.0
    for index in range(bins):
        chosen = indices == index
        if chosen.any():
            result += chosen.mean() * abs(labels[chosen].mean() - probabilities[chosen].mean())
    return float(result)

def binary(labels: np.ndarray, scores: np.ndarray, probabilities: np.ndarray | None = None, predictions: np.ndarray | None = None) -> dict:
    labels, scores = validate_labels(labels), np.asarray(scores, dtype=float)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid evaluation labels or scores")
    predictions = (scores >= 0).astype(int) if predictions is None else validate_labels(predictions)
    if predictions.shape != labels.shape:
        raise ValueError("Prediction dimensions must match evaluation labels")
    both = len(np.unique(labels)) == 2
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    result = {"samples": len(labels), "fake_fraction": float(labels.mean()), "accuracy": float(np.mean(predictions == labels)), "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)) if both else None, "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)), "fake_f1": float(f1_score(labels, predictions, zero_division=0)), "fake_precision": float(precision_score(labels, predictions, zero_division=0)), "fake_recall": float(recall_score(labels, predictions, zero_division=0)), "auc": float(roc_auc_score(labels, scores)) if both else None, "average_precision": float(average_precision_score(labels, scores)) if both else None, "tp": true_positive, "fp": false_positive, "fn": false_negative, "tn": true_negative}
    # Report each class under the same decision threshold.
    result.update(real_precision=float(precision_score(labels, predictions, pos_label=0, zero_division=0)), real_recall=float(recall_score(labels, predictions, pos_label=0, zero_division=0)), real_f1=float(f1_score(labels, predictions, pos_label=0, zero_division=0)))
    if probabilities is not None:
        probabilities = valid_probabilities(probabilities, labels)
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        result.update(ece=calibration_error(labels, probabilities), brier=float(np.mean((labels - probabilities) ** 2)), log_loss=float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))))
    return result

def risk_coverage(labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray | None = None, points: int = 20) -> list[dict]:
    labels = validate_labels(labels)
    probabilities = valid_probabilities(probabilities, labels)
    predictions = (probabilities >= 0.5).astype(int) if predictions is None else validate_labels(predictions)
    if predictions.shape != labels.shape or isinstance(points, bool) or not isinstance(points, int) or points < 1:
        raise ValueError("Risk coverage requires matching predictions and positive point counts")
    confidence = np.abs(probabilities - 0.5)
    order = np.argsort(-confidence, kind="stable")
    mistakes = predictions[order] != labels[order]
    sizes = np.unique(np.maximum(1, np.ceil(np.linspace(1 / points, 1, points) * len(labels)).astype(int)))
    return [{"coverage": float(size / len(labels)), "risk": float(mistakes[:size].mean()), "accepted": int(size), "confidence_cutoff": float(confidence[order[size - 1]])} for size in sizes]

def evaluate(rows: list[dict], score_key: str = "fusion_risk") -> dict:
    rows = score_rows(rows, score_key)
    if not rows or any("label" not in row or score_key not in row for row in rows):
        raise ValueError("Every evaluated sample requires a label and score")
    labels = validate_labels(np.asarray([row["label"] for row in rows]))
    scores = np.asarray([row[score_key] for row in rows], dtype=float)
    probabilities = np.asarray([row["probability"] for row in rows]) if all(row.get("probability") is not None for row in rows) else None
    predictions = np.asarray([row["prediction"] if row.get("prediction") is not None else int(scores[index] >= 0) for index, row in enumerate(rows)])
    report = binary(labels, scores, probabilities, predictions)
    probability_count = sum(row.get("probability") is not None for row in rows)
    report["probability_coverage"] = {"observed": probability_count, "total": len(rows), "fraction": probability_count / len(rows), "metrics_available": probabilities is not None, "policy": "Probability metrics require probabilities for every scored input; partial subsets are not substituted"}
    domains = defaultdict(list)
    platforms = defaultdict(list)
    for index, row in enumerate(rows):
        domains[str(row.get("domain", "unknown"))].append(index)
        platforms[str(row.get("platform", "unknown"))].append(index)
    for name, groups in (("domains", domains), ("platforms", platforms)):
        report[name] = {key: binary(labels[indices], scores[indices], probabilities[indices] if probabilities is not None else None, predictions[indices]) for key, indices in sorted(groups.items())}
    valid_auc = [value["auc"] for value in report["domains"].values() if value["auc"] is not None]
    report["macro_domain_auc"] = float(np.mean(valid_auc)) if valid_auc else None
    report["macro_domain_f1"] = float(np.mean([value["macro_f1"] for value in report["domains"].values()]))
    if probabilities is not None:
        report["risk_coverage"] = risk_coverage(labels, probabilities, predictions)
        report["probability_calibrated"] = all(row.get("probability_calibrated", False) for row in rows)
    if all(row.get("prediction_set") is not None for row in rows):
        sets = [row["prediction_set"] for row in rows]
        accepted = np.asarray([len(value) == 1 for value in sets])
        report["conformal"] = {"empirical_label_coverage": float(np.mean([label in value for label, value in zip(labels, sets)])), "singleton_fraction": float(accepted.mean()), "empty_fraction": float(np.mean([not value for value in sets])), "average_set_size": float(np.mean([len(value) for value in sets])), "singleton_risk": float(np.mean(predictions[accepted] != labels[accepted])) if accepted.any() else None}
    return report

def group_bootstrap(rows: list[dict], score_key: str = "fusion_risk", repeats: int = 500, seed: int = 42, group_key: str = "domain") -> dict:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    rows = score_rows(rows, score_key)
    if any(group_key not in row for row in rows):
        raise ValueError("Every bootstrap row requires observed grouping metadata")
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    keys = sorted(groups)
    if not keys:
        raise ValueError("No bootstrap samples")
    if len(keys) == 1:
        return {"unit": group_key, "independent_groups": 1, "repeats": repeats, "intervals": {}, "scope": "One group cannot provide a group confidence interval"}
    rng, values = np.random.default_rng(seed), defaultdict(list)
    for _ in range(repeats):
        sample = [row for key in rng.choice(keys, len(keys), replace=True) for row in groups[key]]
        result = evaluate(sample, score_key)
        for metric in ("auc", "macro_f1", "accuracy", "ece", "brier"):
            if result.get(metric) is not None:
                values[metric].append(result[metric])
    return {"unit": group_key, "independent_groups": len(keys), "repeats": repeats, "intervals": {metric: {"lower": float(np.quantile(samples, 0.025)), "upper": float(np.quantile(samples, 0.975)), "valid_repeats": len(samples)} for metric, samples in values.items()}}

def quadrants(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if "semantic_logit" not in row:
            raise ValueError("Quadrant analysis requires semantic logits")
        key = f"semantic_{'high' if row['semantic_logit'] >= 0 else 'low'}_physical_{'high' if row['physical_risk'] >= 0 else 'low'}"
        groups[key].append(row)
    return {key: evaluate(value) for key, value in sorted(groups.items())}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
