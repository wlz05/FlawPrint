# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import dataclass
import math
from numbers import Real
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from engine.results import prediction_rows, score_rows
from engine.lineage import collect, disjoint, ensure_disjoint, scope
from engine.storage import real_array

def validate_labels(labels: np.ndarray) -> np.ndarray:
    original = np.asarray(labels, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in original.flat):
        raise ValueError("Binary numeric labels cannot be Boolean metadata")
    labels = real_array(original)
    if labels.ndim != 1 or not len(labels) or not np.isfinite(labels).all() or np.any((labels != 0) & (labels != 1)):
        raise ValueError("Expected binary labels")
    return labels.astype(int)

def fuse(physical_risk: np.ndarray, semantic_logits: np.ndarray | None = None, semantic_threshold: float = 0.0) -> np.ndarray:
    physical = real_array(physical_risk)
    if physical.ndim != 1 or not np.isfinite(physical).all() or isinstance(semantic_threshold, bool) or not isinstance(semantic_threshold, Real) or not math.isfinite(semantic_threshold):
        raise ValueError("Fusion requires finite one-dimensional scores and a finite threshold")
    if semantic_logits is None:
        return physical
    semantic = real_array(semantic_logits)
    if semantic.shape != physical.shape or not np.isfinite(semantic).all():
        raise ValueError("Fusion score shapes must match")
    return np.maximum(physical, semantic - semantic_threshold)

@dataclass
class Platt:
    slope: float = 1.0
    intercept: float = 0.0
    calibrated: bool = False

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in (self.slope, self.intercept)) or self.slope < 0 or not isinstance(self.calibrated, bool):
            raise ValueError("Probability calibration parameters must be finite and orientation preserving")

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray) -> Platt:
        labels = validate_labels(labels)
        scores = real_array(scores)
        if scores.shape != labels.shape or not np.isfinite(scores).all() or len(np.unique(labels)) < 2:
            raise ValueError("Calibration requires finite scores and both classes")
        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            logits = parameters[0] * scores + parameters[1]
            loss = np.mean(np.logaddexp(0, logits) - labels * logits) + 1e-4 * parameters[0] ** 2
            residual = expit(logits) - labels
            gradient = np.array([np.mean(residual * scores) + 2e-4 * parameters[0], np.mean(residual)])
            return float(loss), gradient
        fitted = minimize(objective, [1.0, 0.0], jac=True, method="L-BFGS-B", bounds=[(0, None), (None, None)])
        if not fitted.success:
            raise RuntimeError("Probability calibration did not converge")
        return cls(float(fitted.x[0]), float(fitted.x[1]), True)

    def probability(self, scores: np.ndarray) -> np.ndarray:
        scores = real_array(scores)
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("Probability prediction requires finite one-dimensional scores")
        return expit(self.slope * scores + self.intercept)

@dataclass
class SplitConformal:
    alpha: float
    quantile: float
    sample_count: int
    identifiers: list[str]
    exchangeability_scope: str = "Coverage requires exchangeable labeled calibration and test examples"

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in (self.alpha, self.quantile)) or not 0 < self.alpha < 1 or not 0 <= self.quantile <= 1 or isinstance(self.sample_count, bool) or not isinstance(self.sample_count, (int, np.integer)) or self.sample_count < 1:
            raise ValueError("Invalid split conformal state")
        if not isinstance(self.identifiers, list) or len(self.identifiers) not in (0, self.sample_count) or any(not isinstance(identifier, str) or not identifier.strip() for identifier in self.identifiers) or len(set(self.identifiers)) != len(self.identifiers):
            raise ValueError("Conformal identifiers must be unique and match the sample count")

    @classmethod
    def fit(cls, probabilities: np.ndarray, labels: np.ndarray, alpha: float = 0.1, identifiers: list[str] | None = None) -> SplitConformal:
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("Alpha must be between zero and one")
        labels = validate_labels(labels)
        probabilities = real_array(probabilities)
        if probabilities.shape != labels.shape or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Invalid calibrated probabilities")
        scores = np.where(labels == 1, 1 - probabilities, probabilities)
        order = math.ceil((len(scores) + 1) * (1 - alpha))
        # Include the finite-sample conformal correction.
        quantile = float(np.sort(scores)[order - 1]) if order <= len(scores) else 1.0
        return cls(alpha, quantile, len(scores), list(identifiers or []))

    def prediction_sets(self, probabilities: np.ndarray) -> list[list[int]]:
        probabilities = real_array(probabilities)
        if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Prediction sets require finite probabilities in the unit interval")
        return [[label for label, nonconformity in ((0, probability), (1, 1 - probability)) if nonconformity <= self.quantile] for probability in probabilities]

def validate_partitions(*partitions: list[dict]) -> list[list[str]]:
    identifiers, seen = [], set()
    histories = []
    for rows in partitions:
        rows = prediction_rows(rows)
        current = [row.get("id") for row in rows]
        if any(not isinstance(identifier, str) or not identifier.strip() for identifier in current) or len(set(current)) != len(current) or seen & set(current):
            raise ValueError("Calibration partitions require unique disjoint sample identifiers")
        histories.append(collect(rows))
        identifiers.append(current)
        seen.update(current)
    disjoint(*histories)
    return identifiers

def calibrate_rows(validation: list[dict], calibration: list[dict], alpha: float = 0.1, score_key: str = "fusion_risk") -> dict:
    validation, calibration = score_rows(validation, score_key), score_rows(calibration, score_key)
    first, second = validate_partitions(validation, calibration)
    if any("label" not in row for row in validation + calibration):
        raise ValueError("Conformal calibration requires observed labels")
    if any(row.get("split") == "test" or row.get("profile_overlap") for row in validation + calibration):
        raise ValueError("Use independent validation and calibration partitions")
    platt = Platt.fit(np.array([row[score_key] for row in validation]), np.array([row["label"] for row in validation]))
    probabilities = platt.probability(np.array([row[score_key] for row in calibration]))
    conformal = SplitConformal.fit(probabilities, np.array([row["label"] for row in calibration]), alpha, [row["id"] for row in calibration])
    return {"version": "flawprint-calibration-v1", "score_key": score_key, "platt": vars(platt), "conformal": vars(conformal), "validation_identifiers": sorted(first), "evidence_history": collect(validation, calibration)}

def apply_calibration(rows: list[dict], state: dict | None = None) -> list[dict]:
    rows = prediction_rows(rows)
    validate_partitions(rows)
    if state is not None and (not isinstance(state, dict) or state.get("version") != "flawprint-calibration-v1"):
        raise ValueError("Unsupported calibration state version")
    try:
        platt = Platt(**state["platt"]) if state else Platt()
        conformal = SplitConformal(**state["conformal"]) if state else None
        key = state["score_key"] if state else "fusion_risk"
        history = state["validation_identifiers"] + state["conformal"]["identifiers"] if state else []
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed probability calibration state") from error
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Stored calibration score keys must be nonempty strings")
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in history) or len(set(history)) != len(history):
        raise ValueError("Stored calibration identifiers must be unique and disjoint")
    forbidden = set(history)
    if forbidden & {row["id"] for row in rows}:
        raise ValueError("Prediction data overlaps probability or conformal calibration")
    evidence_history = state.get("evidence_history") if state else None
    ensure_disjoint(rows, evidence_history)
    rows = score_rows(rows, key)
    probabilities = platt.probability(np.array([row[key] for row in rows]))
    sets = conformal.prediction_sets(probabilities) if conformal else [None] * len(rows)
    output = []
    for row, probability, prediction_set in zip(rows, probabilities, sets):
        result = dict(row, probability=float(probability), probability_calibrated=platt.calibrated, prediction_set=prediction_set, abstain=prediction_set is not None and len(prediction_set) != 1, calibration_overlap_audit=scope(evidence_history) if state else "not_calibrated")
        result["prediction"] = prediction_set[0] if prediction_set is not None and len(prediction_set) == 1 else int(probability >= 0.5)
        output.append(result)
    return output

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
