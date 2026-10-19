# -*- coding: utf-8 -*-
# Channel Complementarity: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import re
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from engine.results import score_rows
from engine.conformal import validate_labels, validate_partitions, SplitConformal
from engine.evidence import paired, transitions
from engine.metrics import evaluate
from engine.lineage import collect, ensure_disjoint
def compare(semantic: list[dict], physical: list[dict], deep: list[dict], provenance: dict, repeats: int = 200, seed: int = 42, group_key: str = "group", alpha: float = 0.1, regularization: float = 1e-4) -> dict:
    if not np.isfinite(regularization) or regularization < 0 or not 0 < alpha < 1:
        raise ValueError("Channel comparison requires valid regularization and conformal alpha")
    channels = [score_rows(rows, "fusion_risk") for rows in (semantic, physical, deep)]
    maps = [{row["id"]: row for row in rows} for rows in channels]
    if not maps[0] or any(len(mapping) != len(rows) or set(mapping) != set(maps[0]) for mapping, rows in zip(maps, channels)):
        raise ValueError("All channels must contain every identity exactly once")
    if not isinstance(provenance, dict) or set(provenance) != {"semantic", "physical", "deep"}:
        raise ValueError("Record provenance for all three frozen channels")
    identifiers = sorted(maps[0])
    rows = [maps[0][identifier] for identifier in identifiers]
    for name, mapping in zip(("semantic", "physical", "deep"), maps):
        contract = provenance[name]
        if not isinstance(contract, dict) or contract.get("frozen") is not True or contract.get("risk_direction") != "high_fake" or not re.fullmatch("[0-9a-f]{64}", str(contract.get("artifact_sha256", ""))) or contract.get("output_task") not in ("news_veracity", "image_forgery"):
            raise ValueError("Each frozen channel requires a hashed artifact, output task and risk direction")
        if contract.get("fit_history") is None or not isinstance(contract.get("fit_identifiers"), list):
            raise ValueError("Frozen channel fitting history must be recorded")
        if set(contract["fit_identifiers"]) & set(identifiers):
            raise ValueError("Channel fitting identities overlap the comparison partitions")
        ensure_disjoint(rows, contract["fit_history"])
        for original in rows:
            other = mapping[original["id"]]
            if original.get("profile_overlap") or other.get("profile_overlap") or other.get("abstain", False) or any(other.get(key) != original.get(key) for key in ("label", "split", "domain", "group", "platform", "image", "content_sha256", "physical_manipulated", "narrative_false")):
                raise ValueError("Channels require matching complete scores, labels and evaluation metadata")
    if provenance["semantic"]["output_task"] != "news_veracity":
        raise ValueError("The shared semantic channel must predict news veracity")
    parts = {split: [index for index, row in enumerate(rows) if row.get("split") == split] for split in ("validation", "calibration", "test")}
    if sum(len(part) for part in parts.values()) != len(rows) or any(not part for part in parts.values()):
        raise ValueError("Channels require independent validation, calibration and test partitions")
    validate_partitions(*[[rows[index] for index in parts[split]] for split in parts])
    labels = validate_labels(np.asarray([row.get("label", np.nan) for row in rows]))
    if len(np.unique(labels[parts["validation"]])) != 2:
        raise ValueError("Source validation must contain both news classes")
    if {rows[index]["domain"] for index in parts["validation"] + parts["calibration"]} & {rows[index]["domain"] for index in parts["test"]}:
        raise ValueError("Target test domains cannot supply the channel head or calibration labels")
    values = np.asarray([[mapping[identifier]["fusion_risk"] for mapping in maps] for identifier in identifiers], dtype=float)
    first = np.asarray(parts["validation"], dtype=int)
    center = values[first].mean(axis=0)
    scale = np.maximum(values[first].std(axis=0), 1e-6)
    standardized = (values - center) / scale
    reports, predictions, states = {}, {}, {}
    methods = {"semantic": [0], "semantic_physical": [0, 1], "semantic_deep": [0, 2], "semantic_physical_deep": [0, 1, 2]}
    for name, columns in methods.items():
        design, targets = standardized[first][:, columns], labels[first]
        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            logits = design @ parameters[:-1] + parameters[-1]
            residual = expit(logits) - targets
            loss = np.mean(np.logaddexp(0, logits) - targets * logits) + regularization * np.sum(parameters[:-1] ** 2)
            gradient = np.r_[design.T @ residual / len(targets) + 2 * regularization * parameters[:-1], np.mean(residual)]
            return float(loss), gradient
        fitted = minimize(objective, np.r_[np.ones(len(columns)), 0.0], jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(columns) + [(None, None)])
        if not fitted.success:
            raise ValueError("The source-only channel comparison head did not converge")
        risk = standardized[:, columns] @ fitted.x[:-1] + fitted.x[-1]
        probabilities = expit(risk)
        calibration = parts["calibration"]
        state = SplitConformal.fit(probabilities[calibration], labels[calibration], alpha, [rows[index]["id"] for index in calibration])
        test = parts["test"]
        sets = state.prediction_sets(probabilities[test])
        predicted = [dict(rows[index], fusion_risk=float(risk[index]), probability=float(probabilities[index]), probability_calibrated=True, prediction=prediction_set[0] if len(prediction_set) == 1 else int(risk[index] >= 0), prediction_set=prediction_set, abstain=len(prediction_set) != 1) for index, prediction_set in zip(test, sets)]
        predictions[name] = predicted
        reports[name] = evaluate(predicted)
        states[name] = {"columns": columns, "weights": fitted.x[:-1].tolist(), "intercept": float(fitted.x[-1]), "center": center[columns].tolist(), "scale": scale[columns].tolist(), "regularization": regularization, "validation_identifiers": [rows[index]["id"] for index in first], "calibration_identifiers": state.identifiers, "conformal": vars(state), "fit_history": collect([rows[index] for index in parts["validation"]], [rows[index] for index in calibration])}
    contrasts = {}
    for reference, candidate in (("semantic", "semantic_physical"), ("semantic", "semantic_deep"), ("semantic_deep", "semantic_physical_deep"), ("semantic_physical", "semantic_physical_deep")):
        contrasts[candidate + "_vs_" + reference] = {"paired": paired(predictions[reference], predictions[candidate], "fusion_risk", repeats, seed, group_key), "transitions": transitions(predictions[reference], predictions[candidate], "fusion_risk")}
    return {"version": "flawprint-channel-comparison-v1", "provenance": provenance, "methods": reports, "contrasts": contrasts, "heads": states, "predictions": predictions, "denominator": len(parts["test"]), "scope": "Auxiliary source-fitted nonnegative heads compare frozen channels on identical news labels and test identities; these are complementarity controls, not the deployed FlawPrint max rule; target conformal coverage is empirical"}
