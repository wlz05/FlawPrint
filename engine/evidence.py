# -*- coding: utf-8 -*-
# Increment and Role Audit: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.optimize import brentq
from engine.conformal import validate_labels, validate_partitions
from engine.experiment import indices
from engine.metrics import evaluate, binary
from engine.copula import Marginals, correlation, score, positive_matrix
from engine.profile import DomainProfile, ProfileConfig
from engine.results import prediction_rows, score_rows
from engine.dependence import availability
from engine.data import missing
from engine.storage import subset, validate_bundle
METRICS = ("auc", "macro_f1", "accuracy", "ece", "brier", "log_loss", "macro_domain_auc", "macro_domain_f1")
def orientation(genuine: np.ndarray, fake: np.ndarray, fractions: list[float] | None = None) -> dict:
    if any(np.asarray(reference).dtype.kind == "b" for reference in (genuine, fake)):
        raise ValueError("Class correlation matrices require numeric values")
    genuine, fake = positive_matrix(genuine, True), positive_matrix(fake, True)
    if genuine.shape != fake.shape or len(genuine) < 2:
        raise ValueError("Orientation analysis requires matching correlation matrices with at least two dimensions")
    requested = [0, 0.25, 0.5, 0.75, 1] if fractions is None else fractions
    if any(isinstance(value, (bool, np.bool_)) for value in np.asarray(requested, dtype=object).flat):
        raise ValueError("Mixture fractions cannot be Boolean metadata")
    values = np.asarray(requested, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Mixture fractions must be a nonempty finite vector in the unit interval")
    difference, points = genuine - fake, []
    def gap(fraction: float) -> float:
        reference = (1 - fraction) * genuine + fraction * fake
        return -float(np.trace(np.linalg.solve(reference, difference)))
    for fraction in sorted(set(values.tolist())):
        reference = (1 - fraction) * genuine + fraction * fake
        operator = np.eye(len(reference)) - np.linalg.solve(reference, np.eye(len(reference)))
        offset = -float(np.linalg.slogdet(reference)[1])
        first, second = offset + float(np.trace(operator @ genuine)), offset + float(np.trace(operator @ fake))
        factor = np.linalg.cholesky(reference)
        whitened = np.linalg.solve(factor, np.linalg.solve(factor, difference).T).T
        direction = gap(fraction)
        points.append({"fake_fraction": fraction, "reference": reference.tolist(), "mean_genuine": first, "mean_fake": second, "mean_gap": direction, "identity_error": abs(first - second - direction), "gap_derivative": -float(np.sum(whitened ** 2)), "log_determinant": -offset, "model_power_screen": availability(reference)["power_screen"], "genuine_variance": max(0.0, 2 * float(np.trace((operator @ genuine) @ (operator @ genuine)))), "fake_variance": max(0.0, 2 * float(np.trace((operator @ fake) @ (operator @ fake))))})
    endpoints, crossing, status = [gap(0), gap(1)], None, "no_sign_crossing"
    if np.linalg.norm(difference, ord="fro") <= 1e-10:
        status = "class_references_indistinguishable_at_tolerance"
    elif endpoints[0] > 0 and endpoints[1] < 0:
        crossing, status = float(brentq(gap, 0, 1)), "unique_interior_crossing"
    elif endpoints[0] == 0 or endpoints[1] == 0:
        crossing, status = float(endpoints[1] == 0), "endpoint_crossing"
    maximum = crossing if crossing is not None else None if status == "class_references_indistinguishable_at_tolerance" else 0.0 if endpoints[0] < 0 else 1.0
    return {"genuine_reference": genuine.tolist(), "fake_reference": fake.tolist(), "points": points, "endpoint_gaps": endpoints, "crossing_fraction": crossing, "crossing_status": status, "gaussian_entropy_maximizing_fraction": maximum, "reference_difference_tolerance": 1e-10, "strict_decrease_condition": "Distinct positive definite class references under the stated common-marginal population model", "scope": "Offline class-reference mechanism analysis with zero means, common standard normal marginals and unregularized linear population correlation mixing; the Gaussian projection entropy is not the actual mixture entropy; mean direction is not an AUC guarantee; class references and test labels never choose deployment orientation"}
def mechanism(rho: float = 0.8, samples: int = 5000, pool_size: int = 1000, repeats: int = 3, fractions: list[float] | None = None, shrinkage: float = 0.1, seed: int = 42) -> dict:
    if not np.isfinite(rho) or not 0 < rho < 1 or not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("Mechanism correlation must be positive below one and shrinkage must lie in the unit interval")
    for value, minimum in ((samples, 100), (pool_size, 12), (repeats, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("Mechanism sample sizes and repetitions must satisfy their integer minima")
    fractions = np.asarray([0, 0.25, 0.5, 0.75, 1] if fractions is None else fractions, dtype=float)
    if fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("Mechanism fractions must be a nonempty finite vector in the unit interval")
    rng = np.random.default_rng(seed)
    genuine_reference = np.array([[1.0, rho], [rho, 1.0]])
    fake_reference = np.array([[1.0, -rho], [-rho, 1.0]])
    genuine = rng.multivariate_normal(np.zeros(2), genuine_reference, samples)
    fake = rng.multivariate_normal(np.zeros(2), fake_reference, samples)
    queries = np.vstack((genuine, fake))
    labels = np.r_[np.zeros(samples, dtype=int), np.ones(samples, dtype=int)]
    population, estimated = [], []
    for fraction in sorted(set(fractions.tolist())):
        direction = (1 - 2 * fraction) * rho
        reference = np.array([[1.0, direction], [direction, 1.0]])
        operator = np.eye(2) - np.linalg.solve(reference, np.eye(2))
        offset = -float(np.linalg.slogdet(reference)[1])
        expected_genuine = offset + float(np.trace(operator @ genuine_reference))
        expected_fake = offset + float(np.trace(operator @ fake_reference))
        closed_gap = 4 * rho * direction / (1 - direction ** 2)
        scores = score(queries, reference)
        observed_gap = float(scores[:samples].mean() - scores[samples:].mean())
        variance_genuine = 2 * float(np.trace((operator @ genuine_reference) @ (operator @ genuine_reference)))
        variance_fake = 2 * float(np.trace((operator @ fake_reference) @ (operator @ fake_reference)))
        gap_error = float(np.sqrt(max(0, variance_genuine + variance_fake) / samples))
        screening = availability(reference)
        population.append({"fake_fraction": fraction, "reference_correlation": direction, "expected_genuine_score": expected_genuine, "expected_fake_score": expected_fake, "expected_gap": expected_genuine - expected_fake, "closed_gap": closed_gap, "identity_error": abs(expected_genuine - expected_fake - closed_gap), "observed_gap": observed_gap, "gap_standard_error": gap_error, "gaussian_power_screen": screening["power_screen"], "gap_to_reference_sd": screening["gap_to_real_sd"], "metrics": binary(labels, -scores), "mixture_fourth_moment_excess": 2 * rho ** 2, "gaussian_summary_fourth_moment_excess": 2 * direction ** 2})
        fake_count = int(np.floor(pool_size * fraction + 0.5))
        for repetition in range(repeats):
            pool = np.vstack((rng.multivariate_normal(np.zeros(2), genuine_reference, pool_size - fake_count), rng.multivariate_normal(np.zeros(2), fake_reference, fake_count)))
            marginals = Marginals.fit(pool)
            fitted = correlation(marginals.transform(pool), shrinkage)
            values = score(marginals.transform(queries), fitted)
            screening = availability(fitted)
            estimated.append({"requested_fake_fraction": fraction, "actual_fake_fraction": fake_count / pool_size, "repetition": repetition, "profile_samples": pool_size, "estimated_correlation": float(fitted[0, 1]), "observed_gap": float(values[:samples].mean() - values[samples:].mean()), "gaussian_power_screen": screening["power_screen"], "pool_fourth_moment_excess": float(np.mean(pool[:, 0] ** 2 * pool[:, 1] ** 2) - 1), "metrics": binary(labels, -values)})
    return {"settings": {"rho": rho, "samples_per_class": samples, "profile_samples": pool_size, "repeats": repeats, "fractions": fractions.tolist(), "shrinkage": shrinkage, "seed": seed}, "population": population, "estimated": estimated, "marginal_controls": {"population_single_coordinate_auc": 0.5, "empirical_single_coordinate_auc": [binary(labels, queries[:, column])["auc"] for column in range(2)]}, "oracle_class_likelihood_ratio_metrics": binary(labels, (score(queries, fake_reference) - score(queries, genuine_reference)) / 2), "class_self_reference_mean": -float(np.log1p(-rho ** 2)), "genuine_covariance": genuine_reference.tolist(), "fake_covariance": fake_reference.tolist(), "test_set_policy": "One frozen independently generated test set shared across all profile fractions and repetitions", "population_reference_policy": "Known standard-normal marginals and exact Gaussian correlation summary of a component mixture", "estimated_reference_policy": "Fresh independent profile draws, empirical PIT and configured correlation shrinkage; no labels passed to marginal fitting or correlation estimation", "scope": "Synthetic component labels are not verified news veracity; equal marginals can conceal dependence; zero correlation does not imply independence; model availability does not establish label orientation or real-world benefit"}
def differences(reference: dict, candidate: dict) -> dict:
    return {key: candidate[key] - reference[key] if candidate.get(key) is not None and reference.get(key) is not None else None for key in METRICS}
def paired(reference: list[dict], candidate: list[dict], score_key: str = "fusion_risk", repeats: int = 1000, seed: int = 42, group_key: str = "domain") -> dict:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 20 or group_key not in ("domain", "group"):
        raise ValueError("Paired evaluation requires twenty repeats and a domain or event grouping key")
    reference, candidate = score_rows(reference, score_key), score_rows(candidate, score_key)
    validate_partitions(reference)
    validate_partitions(candidate)
    first, second = {row["id"]: row for row in reference}, {row["id"]: row for row in candidate}
    if not first or set(first) != set(second):
        raise ValueError("Paired methods must contain exactly the same test identifiers")
    identifiers = sorted(first)
    reference, candidate = [first[key] for key in identifiers], [second[key] for key in identifiers]
    groups = defaultdict(list)
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left.get("split") != "test" or right.get("split") != "test" or left.get("profile_overlap") or right.get("profile_overlap"):
            raise ValueError("Paired claims require held-out test predictions without profile overlap")
        if "label" not in left or left["label"] != right.get("label") or any(left.get(key) != right.get(key) for key in ("domain", "group", "platform", "image", "content_sha256", "physical_manipulated", "narrative_false")):
            raise ValueError("Paired methods disagree on labels or evaluation metadata")
        if group_key not in left or not str(left[group_key]).strip():
            raise ValueError("Every paired row requires the selected grouping metadata")
        if left.get(score_key) is None or right.get(score_key) is None:
            raise ValueError("Full discrimination comparison requires scores for every input; use deployment reports for no-evidence routes")
        groups[str(left[group_key])].append(index)
    left, right = evaluate(reference, score_key), evaluate(candidate, score_key)
    deltas = defaultdict(list)
    rng, keys = np.random.default_rng(seed), sorted(groups)
    if len(keys) >= 2:
        for _ in range(repeats):
            sampled = rng.choice(keys, len(keys), replace=True)
            first_rows, second_rows = [], []
            for draw, key in enumerate(sampled):
                for index in groups[key]:
                    first_row, second_row = dict(reference[index]), dict(candidate[index])
                    if group_key == "domain":
                        first_row["domain"] = second_row["domain"] = f"draw{draw}:{key}"
                    first_rows.append(first_row)
                    second_rows.append(second_row)
            for metric, value in differences(evaluate(first_rows, score_key), evaluate(second_rows, score_key)).items():
                if value is not None:
                    deltas[metric].append(value)
    intervals = {metric: {"lower": float(np.quantile(values, 0.025)), "upper": float(np.quantile(values, 0.975)), "valid_repeats": len(values)} if values else None for metric in METRICS for values in [deltas[metric]]}
    domains = {}
    for domain in sorted({row.get("domain", "unknown") for row in reference}):
        selected = [index for index, row in enumerate(reference) if row.get("domain", "unknown") == domain]
        first_report, second_report = evaluate([reference[index] for index in selected], score_key), evaluate([candidate[index] for index in selected], score_key)
        domains[domain] = {"samples": len(selected), "reference": first_report, "candidate": second_report, "delta": differences(first_report, second_report)}
    return {"samples": len(reference), "reference": left, "candidate": right, "delta": differences(left, right), "paired_intervals": intervals, "unit": group_key, "independent_groups": len(keys), "repeats": repeats, "seed": seed, "domains": domains, "decision_coverage": {"reference": float(np.mean([not row.get("abstain", False) for row in reference])), "candidate": float(np.mean([not row.get("abstain", False) for row in candidate]))}, "scope": "Paired descriptive cluster bootstrap; full-score classification and selective decision coverage are separate; one group cannot provide a group confidence interval; this does not establish official SOTA provenance"}
def transitions(reference: list[dict], candidate: list[dict], score_key: str) -> dict:
    reference, candidate = score_rows(reference, score_key), score_rows(candidate, score_key)
    validate_partitions(reference)
    validate_partitions(candidate)
    if len(reference) != len(candidate) or any(left["id"] != right["id"] or "label" not in left or left["label"] != right.get("label") for left, right in zip(reference, candidate)):
        raise ValueError("Error transitions require aligned identifiers and labels")
    rescued, harmed, unchanged_correct, unchanged_wrong = 0, 0, 0, 0
    for left, right in zip(reference, candidate):
        if left.get("abstain", False) or right.get("abstain", False):
            continue
        first = left["prediction"] if left.get("prediction") is not None else int(left[score_key] >= 0)
        second = right["prediction"] if right.get("prediction") is not None else int(right[score_key] >= 0)
        correct_first, correct_second = first == left["label"], second == left["label"]
        rescued += int(not correct_first and correct_second)
        harmed += int(correct_first and not correct_second)
        unchanged_correct += int(correct_first and correct_second)
        unchanged_wrong += int(not correct_first and not correct_second)
    decided = rescued + harmed + unchanged_correct + unchanged_wrong
    wrong_reference, correct_reference = rescued + unchanged_wrong, harmed + unchanged_correct
    return {"samples": len(reference), "joint_decided": decided, "joint_decision_coverage": decided / len(reference) if reference else None, "rescued": rescued, "harmed": harmed, "unchanged_correct": unchanged_correct, "unchanged_wrong": unchanged_wrong, "reference_wrong": wrong_reference, "reference_correct": correct_reference, "rescue_rate_given_reference_error": rescued / wrong_reference if wrong_reference else None, "harm_rate_given_reference_correct": harmed / correct_reference if correct_reference else None, "rescue_fraction": rescued / decided if decided else None, "harm_fraction": harmed / decided if decided else None, "matched_accuracy_delta": (rescued - harmed) / decided if decided else None, "scope": "Error transitions use only rows decided by both methods; full-score metrics retain every input"}
def stratified(reference: list[dict], candidate: list[dict], annotations: list[dict], score_key: str = "fusion_risk", repeats: int = 1000, seed: int = 42, group_key: str = "domain") -> dict:
    reference, candidate, annotations = prediction_rows(reference), prediction_rows(candidate), prediction_rows(annotations)
    overall = paired(reference, candidate, score_key, repeats, seed, group_key)
    first, second = {row["id"]: row for row in reference}, {row["id"]: row for row in candidate}
    if any(not row.get("id") for row in annotations) or len({row["id"] for row in annotations}) != len(annotations):
        raise ValueError("Attribute annotations require unique nonempty identifiers")
    attributes = {row["id"]: row for row in annotations}
    if set(attributes) != set(first):
        raise ValueError("Attribute annotations must cover exactly every paired test identifier, including unknown status")
    statuses, sources = {}, set()
    for key, annotation in attributes.items():
        if "label" in annotation and annotation["label"] != first[key]["label"]:
            raise ValueError("Attribute annotations disagree with news veracity labels")
        if "physical_manipulated" in annotation:
            source = annotation.get("annotation_source")
            if not isinstance(source, str) or missing(source) or not source.strip():
                raise ValueError("Known physical status requires a documented annotation source")
            if "physical_manipulated" in first[key] and first[key]["physical_manipulated"] != annotation["physical_manipulated"]:
                raise ValueError("Attribute annotations disagree with recorded physical status")
            sources.add(source.strip())
            statuses[key] = "altered" if annotation["physical_manipulated"] else "pristine"
        else:
            statuses[key] = "unknown"
    identifiers = sorted(first)
    reference, candidate = [first[key] for key in identifiers], [second[key] for key in identifiers]
    status_reports, cells = {}, []
    for status in ("pristine", "altered", "unknown"):
        chosen = [key for key in identifiers if statuses[key] == status]
        left, right = [first[key] for key in chosen], [second[key] for key in chosen]
        status_reports[status] = {"samples": len(chosen), "population_fraction": len(chosen) / len(identifiers), "comparison": paired(left, right, score_key, repeats, seed, group_key) if chosen else None, "transitions": transitions(left, right, score_key)}
        for label in (0, 1):
            selected = [key for key in chosen if first[key]["label"] == label]
            left, right = [first[key] for key in selected], [second[key] for key in selected]
            first_report, second_report = (evaluate(left, score_key), evaluate(right, score_key)) if selected else (None, None)
            comparison = {"reference": first_report, "candidate": second_report, "delta": differences(first_report, second_report), "scope": "Single-label descriptive metrics without bootstrap interaction inference"} if selected else None
            cells.append({"physical_status": status, "news_label": label, "samples": len(selected), "population_fraction": len(selected) / len(identifiers), "comparison": comparison, "transitions": transitions(left, right, score_key)})
    changes = transitions(reference, candidate, score_key)
    known = sum(status != "unknown" for status in statuses.values())
    return {"overall": overall, "transitions": changes, "physical_status": status_reports, "cells": cells, "annotation_coverage": {"known": known, "unknown": len(identifiers) - known, "total": len(identifiers), "fraction": known / len(identifiers), "sources": sorted(sources)}, "accuracy_accounting": {"cell_rescued_total": sum(cell["transitions"]["rescued"] for cell in cells), "cell_harmed_total": sum(cell["transitions"]["harmed"] for cell in cells), "cell_joint_decided_total": sum(cell["transitions"]["joint_decided"] for cell in cells), "matched_accuracy_delta": changes["matched_accuracy_delta"], "full_population_identity_applicable": changes["joint_decided"] == len(identifiers)}, "annotation_policy": "Offline external physical attributes; missing status remains unknown; annotations never change predictions, reference fitting, thresholds or deployment gates", "scope": "Descriptive physical-status and news-label heterogeneity; single-class AUC and absent cells are undefined; subgroup intervals do not test interactions or establish causal effects; aggregate improvements can conceal subgroup harm"}
def mixture(pool: dict, test: dict, domain: str, config: ProfileConfig, fractions: list[float], size: int = 40, repeats: int = 5, seed: int = 42, image_only: bool = False) -> dict:
    validate_bundle(pool)
    validate_bundle(test)
    config.validate()
    if pool["names"].tolist() != test["names"].tolist() or any(row.get("domain") != domain or row.get("split") == "test" for row in pool["rows"]) or any(row.get("domain") != domain or row.get("split") != "test" for row in test["rows"]):
        raise ValueError("Mixture sensitivity requires a labeled independent pool and a matching target test schema")
    validate_partitions(pool["rows"], test["rows"])
    labels = validate_labels(np.asarray([row.get("label", np.nan) for row in pool["rows"]]))
    validate_labels(np.asarray([row.get("label", np.nan) for row in test["rows"]]))
    if isinstance(size, bool) or not isinstance(size, int) or size < config.min_samples or isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Mixture pool size must satisfy the profile minimum and repeats must be positive")
    if not fractions or not np.isfinite(fractions).all() or any(not 0 <= value <= 1 for value in fractions):
        raise ValueError("Mixture fractions must lie in the unit interval")
    columns = [index for index, name in enumerate(pool["names"]) if str(name) not in ("text_length", "character_diversity", "hashtag_density", "symbol_density", "exclamation_density", "uppercase_ratio")] if image_only else None
    genuine, fake = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    rng, rows, skipped = np.random.default_rng(seed), [], []
    for fraction in sorted(set(fractions)):
        fake_count = int(np.floor(size * fraction + 0.5))
        real_count = size - fake_count
        if real_count > len(genuine) or fake_count > len(fake):
            skipped.append({"fake_fraction": fraction, "reason": "Insufficient independent examples; replacement is disabled"})
            continue
        for repetition in range(repeats):
            selected = np.r_[rng.choice(genuine, real_count, replace=False), rng.choice(fake, fake_count, replace=False)]
            rng.shuffle(selected)
            samples = subset(pool, selected)
            for row in samples["rows"]:
                row.pop("label", None)
                row["split"] = "profile"
            profile = DomainProfile.fit(samples, domain, config, columns)
            predicted = profile.predict(test)
            scored = [dict(row, fusion_risk=row["physical_risk"]) for row in predicted]
            report = evaluate(scored)
            screening = availability(profile.reference)
            rows.append({"requested_fake_fraction": fraction, "actual_fake_fraction": fake_count / size, "real_samples": real_count, "fake_samples": fake_count, "repetition": repetition, "samples": size, "profile_ids": profile.identifiers, "retained_features": profile.names, "condition_number": profile.quality["condition_number"], "gap_to_real_sd": screening["gap_to_real_sd"], "model_power_screen": screening["power_screen"], "metrics": report})
    return {"results": rows, "skipped": skipped, "seed": seed, "image_only": image_only, "repeats": repeats, "scope": "Offline label-assisted mixture construction; all labels removed before fitting; independent frozen test set; repeated draws share a finite pool and are not independent domains; model power screening is not a genuine-label guarantee"}
