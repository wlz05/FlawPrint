# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
from scipy.special import expit
from engine.storage import subset, write_json, write_rows, write_csv, relative_path, validate_bundle, native
from engine.data import audit_splits
from engine.traces import FAMILIES, NAMES
from engine.profile import DomainProfile, ProfileConfig
from engine.baselines import Baseline, METHODS, needs_semantic
from engine.conformal import Platt, SplitConformal, fuse, validate_partitions
from engine.metrics import evaluate, group_bootstrap, quadrants, binary
from engine.provenance import source_digest
from doctor import environment
from engine.lineage import collect
from engine.results import prediction_rows

def indices(bundle: dict, split: str, domain: str | None = None) -> list[int]:
    return [index for index, row in enumerate(bundle["rows"]) if row["split"] == split and (domain is None or row["domain"] == domain)]

def bundle_digest(bundle: dict) -> str:
    checksum = hashlib.sha256(json.dumps(native(bundle["rows"]), sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8"))
    for key in ("values", "observed", "patches", "names"):
        array = np.ascontiguousarray(bundle[key])
        checksum.update(key.encode("ascii") + str(array.shape).encode("ascii") + array.dtype.str.encode("ascii"))
        checksum.update(array.tobytes())
    return checksum.hexdigest()

def named_risk(bundle: dict, risks: np.ndarray) -> list[dict]:
    result = []
    for row, risk in zip(bundle["rows"], risks):
        item = {key: row[key] for key in ("id", "domain", "platform", "group", "split", "label", "semantic_logit", "image", "content_sha256", "scenario", "physical_manipulated", "narrative_false") if key in row}
        item.update(fusion_risk=float(risk), prediction=int(risk >= 0))
        result.append(item)
    return result

def fit_profiles(bundle: dict, config: ProfileConfig, columns: list[int] | None = None) -> dict[str, DomainProfile]:
    domains = sorted({row["domain"] for row in bundle["rows"]})
    profiles = {}
    for domain in domains:
        target = indices(bundle, "profile", domain)
        source = indices(bundle, "train", domain)
        if target and source:
            raise ValueError("A target profile domain cannot also have source training rows")
        selected = target or source
        if not selected:
            raise ValueError(f"No independent profile data for domain {domain}")
        profiles[domain] = DomainProfile.fit(subset(bundle, selected), domain, config, columns)
    return profiles

def physical_rows(bundle: dict, profiles: dict[str, DomainProfile], split: str, mode: str = "physical", fallback: DomainProfile | None = None) -> list[dict]:
    output = []
    for domain in sorted({row["domain"] for row in bundle["rows"]}):
        selected = indices(bundle, split, domain)
        if not selected:
            continue
        profile = fallback or profiles[domain]
        predictions = profile.predict(subset(bundle, selected), explain=split == "test", signs=mode != "copula")
        for row in predictions:
            if row["profile_overlap"]:
                raise ValueError("Scored rows overlap the fitted profile")
            physical = row["copula_risk"] if mode == "copula" else row["sign_conflict_ratio"] - 0.5 if mode == "sign" else row["physical_risk"]
            row["fusion_risk"] = float(fuse(np.array([physical]), np.array([row["semantic_logit"]]) if mode == "fusion" and "semantic_logit" in row else None)[0])
            output.append(row)
    return sorted(output, key=lambda row: row["id"])

def probability_rows(validation: list[dict], calibration: list[dict], test: list[dict], alpha: float = 0.1, conformal: bool = True) -> tuple[list[dict], dict]:
    validation, calibration, test = prediction_rows(validation), prediction_rows(calibration), prediction_rows(test)
    validate_partitions(validation, calibration, test)
    if any(row.get("split") == "test" or row.get("profile_overlap") for row in validation + calibration):
        raise ValueError("Benchmark calibration must exclude test and profile overlap")
    if any("label" not in row for row in validation + test):
        raise ValueError("Validation and evaluation require labels")
    platt = Platt.fit(np.asarray([row["fusion_risk"] for row in validation]), np.asarray([row["label"] for row in validation]))
    predictor = None
    if conformal:
        if not calibration or any("label" not in row for row in calibration):
            raise ValueError("Conformal evaluation requires independent labeled calibration")
        predictor = SplitConformal.fit(platt.probability(np.asarray([row["fusion_risk"] for row in calibration])), np.asarray([row["label"] for row in calibration]), alpha, [row["id"] for row in calibration])
    probabilities = platt.probability(np.asarray([row["fusion_risk"] for row in test]))
    sets = predictor.prediction_sets(probabilities) if predictor else [None] * len(test)
    result = []
    for row, probability, prediction_set in zip(test, probabilities, sets):
        prediction = prediction_set[0] if prediction_set is not None and len(prediction_set) == 1 else row["prediction"]
        result.append(dict(row, prediction=prediction, probability=float(probability), probability_calibrated=True, prediction_set=prediction_set, abstain=prediction_set is not None and len(prediction_set) != 1, calibration_overlap_audit="identifiers_and_recorded_evidence"))
    state = {"version": "flawprint-calibration-v1" if conformal else "flawprint-probability-v1", "score_key": "fusion_risk", "platt": vars(platt), "conformal": vars(predictor) if predictor else None, "validation_identifiers": [row["id"] for row in validation], "evidence_history": collect(validation, calibration if conformal else []), "scope": "Source probability calibration; target coverage is empirical under domain shift"}
    return result, state

def benchmark(bundle: dict, output: str, config: ProfileConfig, alpha: float = 0.1, bootstraps: int = 200, ablations: bool = False) -> dict:
    config.validate()
    validate_bundle(bundle)
    audit = audit_splits(bundle["rows"])
    train = subset(bundle, indices(bundle, "train"))
    validation = subset(bundle, indices(bundle, "validation"))
    calibration = subset(bundle, indices(bundle, "calibration"))
    test = subset(bundle, indices(bundle, "test"))
    if min(len(train["rows"]), len(validation["rows"]), len(test["rows"])) < 1:
        raise ValueError("Benchmark requires train, validation, and test splits")
    output_path = relative_path(output)
    start = time.perf_counter()
    profiles = fit_profiles(bundle, config)
    profile_seconds = time.perf_counter() - start
    for domain, profile in profiles.items():
        safe_name = hashlib.sha256(domain.encode("utf-8")).hexdigest()[:12]
        profile.save((output_path / "profiles" / f"{safe_name}.json").as_posix())
    reports, states, predictions, timing, skipped = {}, {}, {}, {}, {}
    def record(name: str, first: list[dict], second: list[dict], third: list[dict], seconds: float) -> None:
        rows, state = probability_rows(first, second, third, alpha, conformal=bool(second))
        result = evaluate(rows)
        result["bootstrap"] = group_bootstrap(rows, repeats=bootstraps, seed=config.seed)
        if all("semantic_logit" in row and "physical_risk" in row for row in rows):
            result["quadrants"] = quadrants(rows)
        reports[name], states[name], predictions[name], timing[name] = result, state, rows, seconds
        write_rows((output_path / "predictions" / f"{name}.jsonl").as_posix(), rows)
    for method in METHODS:
        if needs_semantic(method) and any("semantic_logit" not in row for row in validation["rows"] + calibration["rows"] + test["rows"]):
            skipped[method] = "No complete external semantic predictions were supplied"
            continue
        if method.startswith("image_rbf") and sum(name in NAMES[:20] for name in bundle["names"]) < 2:
            skipped[method] = "Fewer than two canonical image traces were supplied"
            continue
        start = time.perf_counter()
        baseline = Baseline.fit(method, train, validation, config.seed)
        first, second, third = [named_risk(part, baseline.predict(part)) if part["rows"] else [] for part in (validation, calibration, test)]
        record(method, first, second, third, time.perf_counter() - start)
        states[method]["method"] = method
        if baseline.fit_state is not None:
            states[method]["head_fit"] = baseline.fit_state
        if baseline.column is not None:
            states[method]["selected_trace"] = str(bundle["names"][baseline.column])
    modes = {"copula_only": "copula", "sign_only": "sign", "physical_full": "physical"}
    if all("semantic_logit" in row for row in validation["rows"] + calibration["rows"] + test["rows"]):
        modes["flawprint_fusion"] = "fusion"
    else:
        skipped["flawprint_fusion"] = "No complete external semantic predictions were supplied"
    for name, mode in modes.items():
        start = time.perf_counter()
        first, second, third = [physical_rows(bundle, profiles, split, mode) for split in ("validation", "calibration", "test")]
        record(name, first, second, third, time.perf_counter() - start)
    if ablations:
        for family, excluded in FAMILIES.items():
            start = time.perf_counter()
            columns = [column for column in range(len(bundle["names"])) if column not in excluded]
            reduced = fit_profiles(bundle, config, columns)
            first, second, third = [physical_rows(bundle, reduced, split) for split in ("validation", "calibration", "test")]
            record(f"without_{family}", first, second, third, time.perf_counter() - start)
            if family == "narrative" and "flawprint_fusion" in modes:
                start = time.perf_counter()
                first, second, third = [physical_rows(bundle, reduced, split, "fusion") for split in ("validation", "calibration", "test")]
                record("image_only_fusion", first, second, third, time.perf_counter() - start)
        start = time.perf_counter()
        random_profiles = copy.deepcopy(profiles)
        rng = np.random.default_rng(config.seed)
        for profile in random_profiles.values():
            random = rng.choice([-1, 1], profile.directions.shape)
            profile.directions = np.triu(random, 1) + np.triu(random, 1).T + np.eye(len(random))
        first, second, third = [physical_rows(bundle, random_profiles, split) for split in ("validation", "calibration", "test")]
        record("random_sign_prior", first, second, third, time.perf_counter() - start)
        start = time.perf_counter()
        shuffled = copy.deepcopy(profiles)
        for profile in shuffled.values():
            upper = np.triu_indices(len(profile.weights), 1)
            profile.weights[upper] = rng.permutation(profile.weights[upper])
            profile.weights[(upper[1], upper[0])] = profile.weights[upper]
        first, second, third = [physical_rows(bundle, shuffled, split) for split in ("validation", "calibration", "test")]
        record("shuffled_sign_weights", first, second, third, time.perf_counter() - start)
        start = time.perf_counter()
        source_domains = sorted({row["domain"] for row in train["rows"]})
        fallback = profiles[source_domains[0]]
        first, second, third = [physical_rows(bundle, profiles, split, fallback=fallback) for split in ("validation", "calibration", "test")]
        record("source_profile_transfer", first, second, third, time.perf_counter() - start)
    summary = {"version": "flawprint-benchmark-v1", "audit": audit, "config": asdict(config), "profile_seconds": profile_seconds, "timing_seconds": timing, "methods": reports, "skipped": skipped, "profile_quality": {domain: profile.quality for domain, profile in profiles.items()}, "result_scope": "Computed results for supplied data; synthetic fixtures are not real-data evidence"}
    write_json((output_path / "report.json").as_posix(), summary)
    write_json((output_path / "calibration.json").as_posix(), states)
    write_json((output_path / "provenance.json").as_posix(), {"source_sha256": source_digest(), "environment": environment(), "configuration": asdict(config), "bootstrap_repetitions": bootstraps, "alpha": alpha, "ablations": ablations, "bundle_sha256": bundle_digest(bundle), "sample_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in bundle["rows"]).encode("utf-8")).hexdigest(), "features": bundle["names"].tolist(), "scope": summary["result_scope"]})
    write_csv((output_path / "summary.csv").as_posix(), [{"method": name, **{key: result.get(key) for key in ("accuracy", "macro_f1", "fake_precision", "fake_recall", "fake_f1", "real_precision", "real_recall", "real_f1", "auc", "ece", "brier", "macro_domain_auc", "macro_domain_f1")}, "seconds": timing[name]} for name, result in reports.items()])
    return summary

def domain_matrix(bundle: dict, method: str = "trace_logistic", seed: int = 42) -> dict:
    domains = sorted({row["domain"] for row in bundle["rows"]})
    result, unavailable = [], []
    for source in domains:
        train_indices = indices(bundle, "train", source)
        validation_indices = indices(bundle, "validation", source)
        if not train_indices or not validation_indices:
            unavailable.append(source)
            continue
        baseline = Baseline.fit(method, subset(bundle, train_indices), subset(bundle, validation_indices), seed)
        for target in domains:
            selected = indices(bundle, "test", target)
            if not selected:
                continue
            test = subset(bundle, selected)
            metrics = binary(np.asarray([row["label"] for row in test["rows"]]), baseline.predict(test))
            result.append({"source": source, "target": target, "method": method, **metrics})
    complements = []
    if method == "semantic_external":
        aucs = {row["target"]: row["auc"] for row in result if row["auc"] is not None}
        for first in domains:
            for second in domains:
                if first < second and first in aucs and second in aucs:
                    complements.append({"first": first, "second": second, "auc_sum": aucs[first] + aucs[second], "distance_from_one": abs(aucs[first] + aucs[second] - 1), "scope": "Same supplied detector; complementarity alone does not establish rule inversion"})
    for row in result:
        own = next((item for item in result if item["source"] == row["source"] and item["target"] == row["source"]), None)
        row["auc_drop"] = own["auc"] - row["auc"] if own and own["auc"] is not None and row["auc"] is not None else None
        row["f1_drop"] = own["macro_f1"] - row["macro_f1"] if own else None
    return {"pairs": result, "unavailable_source_domains": unavailable, "fixed_detector_complements": complements}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
