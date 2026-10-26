# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import argparse
import time
import tracemalloc
import platform
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from engine.storage import load_bundle, subset, save_bundle, write_json, write_csv, relative_path
from engine.profile import DomainProfile, ProfileConfig
from engine.experiment import indices, fit_profiles
from engine.copula import moments, separation
from engine.stream import pool, adapt
from engine.traces import extract, TraceConfig, NAMES
from engine.robustness import corrupt
from engine.metrics import binary
from engine.dependence import availability
from engine.conformal import validate_partitions
from engine.provenance import source_digest
from engine.storage import digest
from doctor import environment
from engine.package import configuration, configuration_record, is_builtin

def metrics(bundle: dict, rows: list[dict], threshold: float) -> dict:
    if [row["id"] for row in bundle["rows"]] != [row["id"] for row in rows] or any(row.get("profile_overlap") for row in rows) or not np.isfinite(threshold):
        raise ValueError("Study evaluation requires aligned held-out predictions and a finite threshold")
    labels = np.asarray([row["label"] for row in bundle["rows"]])
    evidence = np.asarray([row["physical_evidence"] for row in rows])
    risks = threshold - evidence
    result = binary(labels, risks, expit(risks))
    genuine, fake = labels == 0, labels == 1
    result["fpr"] = float(np.mean(risks[genuine] >= 0)) if genuine.any() else None
    result["power"] = float(np.mean(risks[fake] >= 0)) if fake.any() else None
    result["threshold"] = threshold
    result["probability_scope"] = "Sigmoid diagnostic; probabilities are not independently calibrated"
    return result

def cold_start(args: argparse.Namespace) -> dict:
    bundle = load_bundle(args.features)
    histories = [DomainProfile.load(path) for path in args.history]
    prior = pool(histories, args.domain)
    candidates = subset(bundle, indices(bundle, "profile", args.domain))
    test = subset(bundle, indices(bundle, "test", args.domain))
    if not test["rows"]:
        raise ValueError("Cold-start evaluation requires target test examples")
    rows, skipped = [], []
    for count in args.sizes:
        if count < 0:
            raise ValueError("Cold-start sample sizes cannot be negative")
        if count > len(candidates["rows"]):
            skipped.append({"samples": count, "reason": "Insufficient independent profile samples"})
            continue
        target = subset(candidates, np.arange(count)) if count else None
        automatic = adapt(prior, target, args.domain, args.n0)
        weights = [automatic.quality.get("shrinkage_weight", 0.0)] if count < 4 else sorted(set([automatic.quality["shrinkage_weight"], *args.weights]))
        for weight in weights:
            if not 0 <= weight <= 1:
                raise ValueError("Cold-start weights must lie in the unit interval")
            profile = adapt(prior, target, args.domain, args.n0)
            if count >= 4:
                from engine.copula import correlation
                empirical = correlation(profile.transform(target), profile.config.shrinkage)
                profile.reference = (1 - weight) * prior.reference + weight * empirical
                from engine.copula import partial_correlation
                profile.directions = np.sign(partial_correlation(profile.reference))
                profile.weights = np.maximum(0, 2 * profile.stability - 1) * np.abs(partial_correlation(profile.reference))
                np.fill_diagonal(profile.weights, 0)
                profile.quality = separation(profile.reference, profile.config.kappa)
            predicted = profile.predict(test)
            result = metrics(test, predicted, profile.quality["threshold"])
            rows.append({"samples": count, "weight": weight, "automatic_weight": automatic.quality.get("shrinkage_weight", 0.0), "condition_number": float(np.linalg.cond(profile.reference)), **result})
    write_csv(relative_path(args.output) / "cold_start.csv", rows)
    return {"results": rows, "skipped": skipped, "n0": args.n0, "scope": "Frozen historical prior and independent target profile/test partitions"}

def thresholds(args: argparse.Namespace) -> dict:
    if not 0 < args.alpha < 0.5:
        raise ValueError("Threshold study alpha must lie between zero and one half")
    bundle = load_bundle(args.features)
    profile = DomainProfile.load(args.profile)
    validation = subset(bundle, indices(bundle, "validation"))
    test = subset(bundle, indices(bundle, "test", args.domain))
    if not validation["rows"] or not test["rows"]:
        raise ValueError("Threshold comparison requires source validation and target test data")
    validate_partitions(validation["rows"], test["rows"])
    first = profile.predict(validation)
    third = profile.predict(test)
    genuine = np.asarray([row["physical_evidence"] for row in first if row["label"] == 0])
    if len(genuine) < 4:
        raise ValueError("Threshold comparison requires four genuine validation samples")
    closed = moments(profile.reference)
    candidates = {"independence_zero": 0.0, "source_genuine_target_scale": float(np.quantile(genuine, args.alpha, method="lower")), "gaussian_moment_screen": float(closed["real_mean"] - norm.ppf(1 - args.alpha) * np.sqrt(closed["real_variance"]))}
    if args.target_calibration:
        target = load_bundle(args.target_calibration)
        if any(row.get("domain") != args.domain or row.get("split") == "test" or "label" not in row for row in target["rows"]):
            raise ValueError("Use a separate labeled target calibration bundle")
        validate_partitions(validation["rows"], target["rows"], test["rows"])
        second = profile.predict(target)
        if any(row["profile_overlap"] for row in second):
            raise ValueError("Target oracle calibration overlaps the fitted profile")
        values = [row["physical_evidence"] for row in second if row["label"] == 0]
        if len(values) < 4:
            raise ValueError("Target oracle calibration needs four genuine examples")
        candidates["target_labeled_oracle"] = float(np.quantile(values, args.alpha, method="lower"))
    results = [{"method": name, **metrics(test, third, threshold)} for name, threshold in candidates.items()]
    write_csv(relative_path(args.output) / "thresholds.csv", results)
    return {"results": results, "availability": availability(profile.reference, args.alpha), "scope": "Moment threshold is a diagnostic comparison; target oracle is explicitly label-assisted"}

def propagation(args: argparse.Namespace) -> dict:
    validation_path = getattr(args, "validation", None)
    if args.generations < 0 or not 0 < args.resize <= 1 or not 1 <= args.quality <= 95 or not 0 < args.alpha < 0.5:
        raise ValueError("Invalid propagation generations, resize factor, JPEG quality, or alpha")
    bundle = load_bundle(args.features)
    config = ProfileConfig(**load_configuration(args.config))
    config.validate()
    domain_rows = [row for row in bundle["rows"] if row["domain"] == args.domain and row["split"] in ("profile", "test")]
    if not domain_rows:
        raise ValueError("Propagation study requires profile and test images")
    original = subset(bundle, [index for index, row in enumerate(bundle["rows"]) if row in domain_rows])
    if validation_path:
        validation = load_bundle(validation_path)
        if validation["names"].tolist() != original["names"].tolist() or any(row.get("split") != "validation" for row in validation["rows"]):
            raise ValueError("Propagation calibration requires a separate validation bundle with the same schema")
        validate_partitions(original["rows"], validation["rows"])
        for key in ("values", "observed", "patches"):
            original[key] = np.concatenate([original[key], validation[key]], axis=0)
        original["rows"].extend(validation["rows"])
    current, results = original, []
    output = relative_path(args.output)
    for generation in range(args.generations + 1):
        if generation:
            resized = corrupt(current["rows"], (output / f"generation_{generation}" / "resize").as_posix(), "resize", args.resize, args.seed + generation)
            recompressed = corrupt(resized, (output / f"generation_{generation}" / "jpeg").as_posix(), "jpeg", args.quality, args.seed + generation)
            values, observed, patches, metadata = extract(recompressed, TraceConfig(), args.workers)
            current = {"values": values, "observed": observed, "patches": patches, "names": np.asarray(NAMES), "rows": metadata}
            save_bundle(output / f"generation_{generation}" / "features.npz", values, observed, patches, metadata, NAMES)
        profile_data = subset(current, indices(current, "profile", args.domain))
        test = subset(current, indices(current, "test", args.domain))
        profile = DomainProfile.fit(profile_data, args.domain, config)
        predicted = profile.predict(test)
        result = metrics(test, predicted, 0.0)
        screening = availability(profile.reference, args.alpha)
        calibrated = None
        if validation_path:
            validation = subset(current, indices(current, "validation"))
            profile.calibrate_genuine(validation, args.alpha)
            calibrated = metrics(test, profile.predict(test), profile.quality["threshold"])
        results.append({"generation": generation, "condition_number": float(np.linalg.cond(profile.reference)), "gap_to_real_sd": screening["gap_to_real_sd"], "power_screen": screening["power_screen"], "calibration_available": bool(validation_path), "calibrated_fpr": calibrated["fpr"] if calibrated else None, "calibrated_power": calibrated["power"] if calibrated else None, "calibration_threshold": calibrated["threshold"] if calibrated else None, "nominal_fpr": args.alpha if calibrated else None, **result})
        profile.save((output / f"generation_{generation}" / "profile.json").as_posix())
    write_csv(output / "propagation.csv", results)
    return {"results": results, "scope": "Paired local resize/JPEG pipeline; the same corruption is applied to profile, test, and optional source validation; source-label quantiles require transfer and compatible PIT for target error control; uncalibrated zero-threshold FPR is a diagnostic; no universal monotonicity or actual platform encoder claim"}

def load_configuration(path: str) -> dict:
    return configuration(path)

def latency(args: argparse.Namespace) -> dict:
    if args.repeats < 2 or args.batch_size < 1 or args.workers < 1:
        raise ValueError("Latency study needs two repeats and positive batch settings")
    started = time.perf_counter()
    bundle = load_bundle(args.features)
    feature_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    profile = DomainProfile.load(args.profile)
    profile_load_seconds = time.perf_counter() - started
    selected = indices(bundle, "test", args.domain)
    if not selected:
        raise ValueError("Latency study requires independent test examples")
    data = subset(bundle, selected)
    if any(row["id"] in profile.identifiers for row in data["rows"]):
        raise ValueError("Latency test overlaps the fitted profile")
    from engine.decision import deploy, GatePolicy
    records = []
    for repetition in range(args.repeats):
        for begin in range(0, len(data["rows"]), args.batch_size):
            part = subset(data, np.arange(begin, min(begin + args.batch_size, len(data["rows"]))))
            if args.pixels:
                tracemalloc.start()
                started = time.perf_counter()
                values, observed, patches, rows = extract(part["rows"], TraceConfig(), args.workers)
                elapsed = time.perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                records.append({"phase": "image_io_and_traces", "repetition": repetition, "samples": len(rows), "seconds": elapsed, "python_traced_peak_bytes": peak})
                part = {"values": values, "observed": observed, "patches": patches, "rows": rows, "names": np.asarray(NAMES)}
            for phase, action in (("copula_only", lambda: profile.predict(part, signs=False)), ("copula_and_local_signs", lambda: profile.predict(part)), ("deployment_routing", lambda: deploy(part, profile, GatePolicy()))):
                tracemalloc.start()
                started = time.perf_counter()
                action()
                elapsed = time.perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                records.append({"phase": phase, "repetition": repetition, "samples": len(part["rows"]), "seconds": elapsed, "python_traced_peak_bytes": peak})
    summaries = {}
    for phase in sorted({row["phase"] for row in records}):
        all_rows = [row for row in records if row["phase"] == phase]
        warm = [row for row in all_rows if row["repetition"] > 0]
        durations = np.asarray([row["seconds"] for row in warm])
        summaries[phase] = {"first_pass_seconds": sum(row["seconds"] for row in all_rows if row["repetition"] == 0), "warm_batch_p50_seconds": float(np.quantile(durations, 0.5)), "warm_batch_p95_seconds": float(np.quantile(durations, 0.95)), "warm_samples_per_second": sum(row["samples"] for row in warm) / max(durations.sum(), 1e-12), "python_traced_peak_bytes": max(row["python_traced_peak_bytes"] for row in all_rows)}
    write_csv(relative_path(args.output) / "latency.csv", records)
    return {"results": summaries, "feature_load_seconds": feature_load_seconds, "profile_load_seconds": profile_load_seconds, "samples": len(data["rows"]), "repeats": args.repeats, "batch_size": args.batch_size, "workers": args.workers, "pixels_recomputed": args.pixels, "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(), "numpy": np.__version__}, "scope": "First pass is not an OS cold-cache benchmark; traced Python allocation is not total process memory; external semantic inference and offline fitting are excluded; phase measurements are separate and not additive"}

def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted FlawPrint threshold and deployment studies.")
    commands = parser.add_subparsers(dest="study", required=True)
    cold = commands.add_parser("cold-start")
    cold.add_argument("features")
    cold.add_argument("--history", action="append", required=True)
    cold.add_argument("--domain", required=True)
    cold.add_argument("--sizes", type=int, nargs="+", default=[0, 4, 12, 25, 50, 100, 200, 500, 1000, 4000])
    cold.add_argument("--weights", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    cold.add_argument("--n0", type=float, default=100.0)
    cold.add_argument("--output", default="./outputs/studies/cold_start")
    cold.set_defaults(handler=cold_start)
    threshold = commands.add_parser("thresholds")
    threshold.add_argument("features")
    threshold.add_argument("--profile", required=True)
    threshold.add_argument("--domain", required=True)
    threshold.add_argument("--target-calibration")
    threshold.add_argument("--alpha", type=float, default=0.05)
    threshold.add_argument("--output", default="./outputs/studies/thresholds")
    threshold.set_defaults(handler=thresholds)
    pipeline = commands.add_parser("propagation")
    pipeline.add_argument("features")
    pipeline.add_argument("--domain", required=True)
    pipeline.add_argument("--config", default="builtin:default")
    pipeline.add_argument("--validation", help="Use a separate labeled source validation bundle at each generation")
    pipeline.add_argument("--generations", type=int, default=5)
    pipeline.add_argument("--resize", type=float, default=0.85)
    pipeline.add_argument("--quality", type=int, default=70)
    pipeline.add_argument("--workers", type=int, default=1)
    pipeline.add_argument("--seed", type=int, default=42)
    pipeline.add_argument("--alpha", type=float, default=0.05)
    pipeline.add_argument("--output", default="./outputs/studies/propagation")
    pipeline.set_defaults(handler=propagation)
    cost = commands.add_parser("latency")
    cost.add_argument("features")
    cost.add_argument("--profile", required=True)
    cost.add_argument("--domain", required=True)
    cost.add_argument("--batch-size", type=int, default=32)
    cost.add_argument("--repeats", type=int, default=5)
    cost.add_argument("--workers", type=int, default=1)
    cost.add_argument("--pixels", action="store_true")
    cost.add_argument("--output", default="./outputs/studies/latency")
    cost.set_defaults(handler=latency)
    args = parser.parse_args()
    result = args.handler(args)
    result["provenance"] = {"source_sha256": source_digest(), "environment": environment(), "inputs": [{"path": getattr(args, key), "sha256": digest(getattr(args, key))} for key in ("features", "profile", "config", "validation", "target_calibration") if getattr(args, key, None) and not (key == "config" and is_builtin(getattr(args, key)))], **({"configuration": configuration_record(args.config)} if getattr(args, "config", None) else {})}
    write_json(relative_path(args.output) / "report.json", result)
    print(f"Saved the {args.study} study to {args.output}")

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
