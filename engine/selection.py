# -*- coding: utf-8 -*-
# Reference Selection: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
import re
import numpy as np
from engine.storage import validate_bundle, subset
from engine.results import prediction_rows
from engine.conformal import validate_labels, validate_partitions
from engine.lineage import ensure_disjoint, collect, merge
from engine.profile import DomainProfile, ProfileConfig
from engine.metrics import evaluate
def selector_contract(metadata: dict, rows: list[dict]) -> None:
    if not isinstance(metadata, dict) or metadata.get("version") != "flawprint-selector-v1" or metadata.get("frozen") is not True:
        raise ValueError("Reference selection requires a versioned frozen external selector")
    if metadata.get("task") != "news_veracity" or metadata.get("calibration_scope") != "source_validation" or metadata.get("score_origin") != "external_news_model":
        raise ValueError("Reference selection requires externally calibrated news probabilities")
    if not isinstance(metadata.get("model"), str) or not metadata["model"].strip() or not re.fullmatch("[0-9a-f]{64}", str(metadata.get("checkpoint_sha256", ""))):
        raise ValueError("Selector model and checkpoint hash must be recorded")
    identifiers = metadata.get("fit_identifiers")
    if not isinstance(identifiers, list) or not identifiers or any(not isinstance(value, str) or not value.strip() for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Selector fitting identifiers must be unique and nonempty")
    if set(identifiers) & {row["id"] for row in rows} or metadata.get("fit_history") is None:
        raise ValueError("Selector fitting must be independent and record evidence history")
    ensure_disjoint(rows, metadata["fit_history"])
def select(bundle: dict, predictions: list[dict], metadata: dict, threshold: float = 0.05) -> tuple[dict, dict]:
    validate_bundle(bundle)
    if not isinstance(threshold, (int, float, np.integer, np.floating)) or isinstance(threshold, (bool, np.bool_)) or not np.isfinite(threshold) or not 0 < threshold < 0.5:
        raise ValueError("The pseudo-genuine probability threshold must lie between zero and one half")
    rows, predictions = prediction_rows(bundle["rows"]), prediction_rows(predictions)
    if not rows or any(row.get("split") != "profile" for row in rows) or len({row["domain"] for row in rows}) != 1:
        raise ValueError("Reference selection requires one independent domain profile pool")
    selector_contract(metadata, rows)
    mapped = {row["id"]: row for row in predictions}
    if len(mapped) != len(predictions) or set(mapped) != {row["id"] for row in rows}:
        raise ValueError("Selector predictions must match the complete pool exactly once")
    probabilities = []
    for row in rows:
        prediction = mapped[row["id"]]
        value = prediction.get("fake_probability")
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Selector fake probabilities must be finite numeric values in the unit interval")
        if prediction.get("domain", row["domain"]) != row["domain"] or prediction.get("split", "profile") != "profile":
            raise ValueError("Selector predictions disagree with profile identities")
        probabilities.append(float(value))
    retained = np.flatnonzero(np.asarray(probabilities) <= threshold)
    selected = subset(bundle, retained)
    for row in selected["rows"]:
        for key in ("label", "physical_manipulated", "narrative_false"):
            row.pop(key, None)
    report = {"version": "flawprint-reference-selection-v1", "selector": copy.deepcopy(metadata), "threshold": float(threshold), "pool_samples": len(rows), "selected_samples": len(retained), "retained_fraction": len(retained) / len(rows), "selected_identifiers": [rows[index]["id"] for index in retained], "pool_identifiers": [row["id"] for row in rows], "pool_history": collect(rows), "selection_rule": "Externally calibrated fake_probability <= threshold", "scope": "Pseudo-genuine selection; target purity and class-conditional dependence are not certified by source confidence"}
    return selected, report
def fit(bundle: dict, predictions: list[dict], metadata: dict, domain: str, config: ProfileConfig, threshold: float = 0.05, columns: list[int] | None = None) -> DomainProfile:
    config.validate()
    selected, report = select(bundle, predictions, metadata, threshold)
    if len(selected["rows"]) < config.min_samples:
        raise ValueError("Too few pseudo-genuine references; keep the frozen profile or use the semantic route")
    profile = DomainProfile.fit(selected, domain, config, columns)
    profile.identifiers = report["pool_identifiers"]
    profile.history = merge(profile.history, report["pool_history"], metadata["fit_history"])
    profile.quality.update(reference_selection=report, selector_fit_identifiers=list(metadata["fit_identifiers"]), reference_scope="Externally selected pseudo-genuine reference; purity requires transferable selector precision")
    return profile
def retained_contamination(q: float, genuine_acceptance: float, fake_acceptance: float) -> float | None:
    values = (q, genuine_acceptance, fake_acceptance)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) for value in values) or not np.isfinite(values).all() or any(not 0 <= value <= 1 for value in values):
        raise ValueError("Composition and class acceptance rates must lie in the unit interval")
    denominator = (1 - q) * genuine_acceptance + q * fake_acceptance
    return float(q * fake_acceptance / denominator) if denominator > 0 else None
def selection_bias(coordinates: np.ndarray, labels: np.ndarray, accepted: np.ndarray) -> dict:
    coordinates = np.asarray(coordinates, dtype=float)
    labels = validate_labels(labels)
    accepted = np.asarray(accepted)
    if coordinates.ndim != 2 or len(coordinates) != len(labels) or not np.isfinite(coordinates).all() or accepted.dtype.kind != "b" or accepted.shape != labels.shape:
        raise ValueError("Selection bias requires fixed coordinates, offline labels and a Boolean admission mask")
    report = {}
    for label in (0, 1):
        original = coordinates[labels == label]
        retained = coordinates[(labels == label) & accepted]
        item = {"original_samples": len(original), "selected_samples": len(retained), "mean_shift_l2": None, "covariance_shift_frobenius": None}
        if len(original) >= 2 and len(retained) >= 2:
            item.update(mean_shift_l2=float(np.linalg.norm(original.mean(axis=0) - retained.mean(axis=0))), covariance_shift_frobenius=float(np.linalg.norm(np.atleast_2d(np.cov(original, rowvar=False)) - np.atleast_2d(np.cov(retained, rowvar=False)), ord="fro")))
        report[str(label)] = item
    return {"classes": report, "scope": "Offline class-conditional moment shifts on a common frozen unfiltered coordinate panel; finite-sample diagnostics, not a purity or invariance certificate"}
def filtering(pool: dict, test: dict, predictions: list[dict], metadata: dict, domain: str, config: ProfileConfig, fractions: list[float], size: int = 40, repeats: int = 5, seed: int = 42, threshold: float = 0.05) -> dict:
    validate_bundle(pool)
    validate_bundle(test)
    config.validate()
    validate_partitions(pool["rows"], test["rows"])
    selector_contract(metadata, pool["rows"] + test["rows"])
    if pool["names"].tolist() != test["names"].tolist() or any(row.get("domain") != domain or row.get("split") != "profile" for row in pool["rows"]) or any(row.get("domain") != domain or row.get("split") != "test" for row in test["rows"]):
        raise ValueError("Filtering controls require matching independent target profile and test partitions")
    labels = validate_labels(np.asarray([row.get("label", np.nan) for row in pool["rows"]]))
    validate_labels(np.asarray([row.get("label", np.nan) for row in test["rows"]]))
    if isinstance(size, bool) or not isinstance(size, int) or size < config.min_samples or isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1 or not fractions or not np.isfinite(fractions).all() or any(not 0 <= q <= 1 for q in fractions):
        raise ValueError("Filtering controls require valid fractions, pool size and repeat count")
    select(pool, predictions, metadata, threshold)
    mapped = {row["id"]: row for row in predictions}
    classes = [np.flatnonzero(labels == label) for label in (0, 1)]
    rng, results, skipped = np.random.default_rng(seed), [], []
    for q in sorted(set(fractions)):
        counts = [size - int(np.floor(size * q + 0.5)), int(np.floor(size * q + 0.5))]
        if any(count > len(candidates) for count, candidates in zip(counts, classes)):
            skipped.append({"fraction": q, "reason": "Insufficient independent pool items; replacement is disabled"})
            continue
        for repetition in range(repeats):
            chosen = np.concatenate([rng.choice(candidates, count, replace=False) for count, candidates in zip(counts, classes)])
            samples = subset(pool, chosen)
            offline_labels = {row["id"]: row["label"] for row in samples["rows"]}
            for row in samples["rows"]:
                row.pop("label", None)
            scores = [mapped[row["id"]] for row in samples["rows"]]
            retained, admission = select(samples, scores, metadata, threshold)
            selected_labels = [offline_labels[row["id"]] for row in retained["rows"]]
            item = {"requested_fake_fraction": q, "actual_fake_fraction": counts[1] / size, "repetition": repetition, "selection": admission, "retained_fake_fraction": float(np.mean(selected_labels)) if selected_labels else None, "class_acceptance": {str(label): sum(value == label for value in selected_labels) / counts[label] if counts[label] else None for label in (0, 1)}, "methods": {}}
            item["selection_bias"] = None
            for method in ("unfiltered", "filtered"):
                try:
                    profile = DomainProfile.fit(samples, domain, config) if method == "unfiltered" else fit(samples, scores, metadata, domain, config, threshold)
                    rows = [dict(row, fusion_risk=row["physical_risk"]) for row in profile.predict(test)]
                    if any(row["profile_overlap"] for row in rows):
                        raise ValueError("Filtering test rows overlap reference selection history")
                    gaps = {key: float(np.mean([row[key] for row in rows if row["label"] == 0]) - np.mean([row[key] for row in rows if row["label"] == 1])) if {row["label"] for row in rows} == {0, 1} else None for key in ("copula_score", "physical_evidence")}
                    item["methods"][method] = {"status": "scored", "metrics": evaluate(rows), "quality": profile.quality, "genuine_minus_fake_score_means": gaps}
                    if method == "unfiltered":
                        admitted = set(admission["selected_identifiers"])
                        item["selection_bias"] = selection_bias(profile.transform(samples), np.asarray([offline_labels[row["id"]] for row in samples["rows"]]), np.asarray([row["id"] in admitted for row in samples["rows"]]))
                except ValueError as error:
                    item["methods"][method] = {"status": "unavailable", "reason": str(error), "metrics": None}
            results.append(item)
    return {"results": results, "skipped": skipped, "seed": seed, "scope": "Offline label-controlled composition; selection and fitting receive no pool labels; frozen test identities; target labels only evaluate results; source confidence does not guarantee preserved class copulas"}
