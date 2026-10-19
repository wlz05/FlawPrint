# -*- coding: utf-8 -*-
# Command Interface: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from engine.storage import read_json, write_json, read_rows, write_rows, write_csv, load_bundle, save_bundle, subset, relative_path, resolved, same_path
from engine.data import normalize, partition, audit_splits, join_predictions
from engine.traces import TraceConfig, NAMES, extract
from engine.profile import ProfileConfig, DomainProfile
from engine.conformal import calibrate_rows, apply_calibration, fuse
from engine.metrics import evaluate, group_bootstrap
from engine.experiment import benchmark, domain_matrix, indices
from engine.baselines import METHODS
from engine.attacks import quantile_match, marginal_report, independent_shuffle, joint_replay
from engine.dependence import tail_report, permutation_calibration, student_selection, availability, platform_alignment, attenuation
from engine.stream import pool, adapt
from engine.robustness import corrupt
from engine.provenance import manuscript_status, conventions, source_digest, document_path, snapshot_path
from engine.validation import preflight
from doctor import environment
from engine.results import prediction_rows
from engine.synthetic import write_fixture, image_demo
from engine.theory import verify
from engine.invariance import verify as verify_invariance
from engine.decision import GatePolicy, deploy, calibrate_routes, deployment_report
from engine.package import VERSION
def profile_config(path: str | None) -> ProfileConfig:
    config = ProfileConfig(**read_json(path)) if path else ProfileConfig()
    config.validate()
    return config
def choose(bundle: dict, split: str | None = None, domain: str | None = None) -> dict:
    selected = [index for index, row in enumerate(bundle["rows"]) if (split is None or row["split"] == split) and (domain is None or row["domain"] == domain)]
    if not selected:
        raise ValueError("The requested sample selection is empty")
    return subset(bundle, selected)
def command_prepare(args: argparse.Namespace) -> dict:
    rows = normalize(read_rows(args.input, args.encoding, args.delimiter, args.records_key), args.image_root, args.labels, args.domain_key)
    audit = audit_splits(rows)
    write_rows(args.output, rows)
    return {"samples": len(rows), "output": args.output, "label_convention": "zero-real-one-fake", "audit": audit}
def command_validate(args: argparse.Namespace) -> dict:
    config = TraceConfig(args.max_side, args.tile_grid, args.min_side)
    result = preflight(read_rows(args.input), args.check_images, args.benchmark, args.min_samples, config)
    write_json(args.output, result)
    if not result["passed"]:
        raise ValueError("Data preflight failed; inspect " + args.output)
    return {key: result[key] for key in ("passed", "samples", "counts", "warnings")}
def command_split(args: argparse.Namespace) -> dict:
    rows = normalize(read_rows(args.input))
    result = partition(rows, args.target_domain, args.seed, args.profile_fraction)
    audit = audit_splits(result, args.check_files)
    write_rows(args.output, result)
    return audit
def command_join(args: argparse.Namespace) -> dict:
    rows = join_predictions(normalize(read_rows(args.input)), read_rows(args.predictions), args.score_key, args.real_positive)
    write_rows(args.output, rows)
    return {"samples": len(rows), "output": args.output}
def command_extract(args: argparse.Namespace) -> dict:
    from engine.storage import digest
    original_hash = digest(args.input)
    rows = normalize(read_rows(args.input))
    audit = audit_splits(rows, True)
    config = TraceConfig(args.max_side, args.tile_grid, args.min_side)
    values, observed, patches, metadata = extract(rows, config, args.workers)
    if digest(args.input) != original_hash:
        raise ValueError("The manifest changed during extraction; rerun with a stable input")
    save_bundle(args.output, values, observed, patches, metadata, NAMES)
    write_json(str(relative_path(args.output).with_suffix(".provenance.json")), {"version": VERSION, "manifest": {"path": args.input, "sha256": original_hash}, "features": {"path": args.output, "sha256": digest(args.output), "names": NAMES}, "trace_config": asdict(config), "workers": args.workers, "source_sha256": source_digest(), "environment": environment(), "audit": audit})
    return {"samples": len(rows), "features": len(NAMES), "tiles": patches.shape[1], "output": args.output, "config": asdict(config), "audit": audit}
def command_profile(args: argparse.Namespace) -> dict:
    from engine.storage import digest
    predictions, metadata = getattr(args, "selector_predictions", None), getattr(args, "selector_metadata", None)
    if bool(predictions) != bool(metadata):
        raise ValueError("Reference selection requires both external predictions and selector metadata")
    inputs = [args.features] + [value for value in (args.config, predictions, metadata, args.genuine_validation) if value]
    fingerprints = [{"path": path, "sha256": digest(path)} for path in inputs]
    bundle = choose(load_bundle(args.features), args.split, args.domain)
    if predictions:
        from engine.selection import fit
        profile = fit(bundle, read_rows(predictions), read_json(metadata), args.domain, profile_config(args.config), args.selector_threshold)
        profile.quality["reference_selection"]["inputs"] = fingerprints
    else:
        profile = DomainProfile.fit(bundle, args.domain, profile_config(args.config))
    if args.genuine_validation:
        profile.calibrate_genuine(load_bundle(args.genuine_validation), args.alpha)
    if any(digest(item["path"]) != item["sha256"] for item in fingerprints):
        raise ValueError("A profile input changed during fitting")
    profile.save(args.output)
    return profile.quality
def command_predict(args: argparse.Namespace) -> dict:
    bundle = choose(load_bundle(args.features), args.split, args.domain)
    profile = DomainProfile.load(args.profile)
    rows = profile.predict(bundle, args.explain, not args.no_signs)
    if any(row["profile_overlap"] for row in rows):
        raise ValueError("Prediction examples overlap the fitted profile")
    if args.fusion and any("semantic_logit" not in row for row in rows):
        raise ValueError("Fusion requires an external semantic score for every sample")
    for row in rows:
        row["fusion_risk"] = float(fuse(np.array([row["physical_risk"]]), np.array([row["semantic_logit"]]) if args.fusion else None, args.semantic_threshold)[0])
        row["prediction"] = int(row["fusion_risk"] >= 0)
    rows = apply_calibration(rows, read_json(args.calibration) if args.calibration else None)
    write_rows(args.output, rows)
    return {"samples": len(rows), "output": args.output, "calibrated": bool(args.calibration), "profile_domain": profile.domain}
def command_calibrate(args: argparse.Namespace) -> dict:
    validation, calibration = prediction_rows(read_rows(args.validation)), prediction_rows(read_rows(args.calibration))
    state = calibrate_routes(validation, calibration, args.alpha) if args.routed else calibrate_rows(validation, calibration, args.alpha, args.score_key)
    write_json(args.output, state)
    return {"output": args.output, "alpha": args.alpha, "calibration_samples": len(calibration), "routed": args.routed}
def command_deploy(args: argparse.Namespace) -> dict:
    bundle = choose(load_bundle(args.features), args.split, args.domain)
    profile = DomainProfile.load(args.profile) if args.profile else None
    policy = GatePolicy(args.min_observed, args.min_image_features, args.gate_alpha, args.semantic_threshold)
    rows = deploy(bundle, profile, policy, args.explain, not args.no_signs, read_json(args.calibration) if args.calibration else None)
    write_rows(args.output, rows)
    report = deployment_report(rows)
    write_json(args.report, report)
    return {key: report[key] for key in ("samples", "accepted_samples", "decision_coverage", "routes", "gate_reasons")}
def command_evaluate(args: argparse.Namespace) -> dict:
    rows = prediction_rows(read_rows(args.predictions))
    if rows and all("decision_route" in row for row in rows):
        result = deployment_report(rows)
        write_json(args.output, result)
        return {key: result[key] for key in ("samples", "accepted_samples", "decision_coverage", "accepted_risk")}
    result = evaluate(rows, args.score_key)
    if args.bootstrap > 0:
        result["bootstrap"] = group_bootstrap(rows, args.score_key, args.bootstrap, args.seed, args.group_key)
    write_json(args.output, result)
    return {key: result.get(key) for key in ("samples", "auc", "macro_f1", "ece", "brier", "macro_domain_auc")}
def command_benchmark(args: argparse.Namespace) -> dict:
    from engine.storage import digest
    original_hash = digest(args.features)
    bundle = load_bundle(args.features)
    if digest(args.features) != original_hash:
        raise ValueError("The feature file changed during loading")
    result = benchmark(bundle, args.output, profile_config(args.config), args.alpha, args.bootstrap, args.ablations)
    provenance_path = relative_path(args.output) / "provenance.json"
    provenance = read_json(provenance_path)
    provenance["input"] = {"path": args.features, "sha256": original_hash}
    if args.config:
        provenance["configuration_file"] = {"path": args.config, "sha256": digest(args.config)}
    write_json(provenance_path, provenance)
    return {"methods": list(result["methods"]), "skipped": result["skipped"], "output": args.output}
def command_theory(args: argparse.Namespace) -> dict:
    result = verify(args.seed, args.samples, args.repetitions)
    write_json(args.output, result)
    return {"maximum_identity_error": max(result["identity_errors"].values()), "output": args.output, "scope": result["scope"]}
def command_invariance(args: argparse.Namespace) -> dict:
    result = verify_invariance(args.seed, args.samples, args.dimensions, args.repetitions)
    write_json(args.output, result)
    return {"output": args.output, "pairing_identities_verified": result["fixed_reference_pairing"]["summary"]["all_identities_verified"], "shared_map_invariance": all(item["invariance_verified"] for item in result["shared_coordinate_maps"]), "marginal_attack": result["marginal_attack"], "scope": "Synthetic F25 diagnostics with explicit reference and tie boundaries"}
def command_mma(args: argparse.Namespace) -> dict:
    bundle = choose(load_bundle(args.features), args.split, args.domain)
    genuine = [index for index, row in enumerate(bundle["rows"]) if row.get("label") == 0]
    generated = [index for index, row in enumerate(bundle["rows"]) if row.get("label") == 1]
    if min(len(genuine), len(generated)) < 2:
        raise ValueError("MMA evaluation requires two examples of each class")
    result = subset(bundle, np.arange(len(bundle["rows"])))
    before = marginal_report(bundle["values"][generated], bundle["values"][genuine], bundle["names"].tolist())
    if args.joint:
        result = joint_replay(bundle, genuine, generated)
    else:
        result["values"][generated] = quantile_match(bundle["values"][generated], bundle["values"][genuine])
        if args.shuffle:
            result["values"][generated] = independent_shuffle(result["values"][generated], args.seed)
        for index in generated:
            result["rows"][index]["attack"] = "feature_mma"
            result["rows"][index]["patch_signs_valid_after_attack"] = False
    after = marginal_report(result["values"][generated], result["values"][genuine], bundle["names"].tolist())
    save_bundle(args.output, result["values"], result["observed"], result["patches"], result["rows"], result["names"].tolist())
    report = {"before": before, "after": after, "scope": "Oracle joint feature and patch replay; semantic predictions remain unchanged" if args.joint else "Feature-space attack; local patch signs are invalid and must be disabled for this attacked bundle", "joint_replay": args.joint}
    write_json(args.report, report)
    return {"output": args.output, "report": args.report, "largest_ks_after": after["largest_ks"]}
def command_dependence(args: argparse.Namespace) -> dict:
    bundle = choose(load_bundle(args.features), args.split, args.domain)
    profile = DomainProfile.load(args.profile)
    if any(row["id"] in profile.identifiers for row in bundle["rows"]):
        raise ValueError("Dependence testing requires held-out examples")
    z = profile.transform(bundle)
    uniform = ndtr(z)
    result = {"tail": tail_report(uniform, profile.names, args.tail_threshold), "permutation": permutation_calibration(z, profile.reference, args.repeats, args.alpha, args.seed, len(profile.identifiers)), "availability": availability(profile.reference, args.alpha, args.batch_size), "student": student_selection(uniform) if args.student else None}
    write_json(args.output, result)
    return {"output": args.output, "dependence_p_value": result["permutation"]["dependence_p_value"], "power_screen": result["availability"]["power_screen"]}
def command_stream(args: argparse.Namespace) -> dict:
    profiles = [DomainProfile.load(path) for path in args.history]
    prior = pool(profiles, args.domain)
    bundle = choose(load_bundle(args.features), args.split, args.domain) if args.features else None
    profile = adapt(prior, bundle, args.domain, args.n0)
    profile.save(args.output)
    return profile.quality
def command_platform(args: argparse.Namespace) -> dict:
    profiles = [DomainProfile.load(path) for path in args.profiles]
    common = sorted(set.intersection(*(set(profile.columns) for profile in profiles)))
    if len(common) < 2:
        raise ValueError("Platform diagnostics need two common features")
    references = {}
    for profile in profiles:
        selected = [profile.columns.index(column) for column in common]
        references[profile.domain] = profile.reference[np.ix_(selected, selected)]
    result = platform_alignment(references)
    result["attenuation"] = [{"domain": domain, "coefficient": coefficient, "condition_number": float(np.linalg.cond(attenuation(reference, np.full(len(reference), coefficient)))), "mean_gap": availability(attenuation(reference, np.full(len(reference), coefficient)))["mean_gap"]} for domain, reference in references.items() for coefficient in (1.0, 0.8, 0.6, 0.4, 0.2)]
    write_json(args.output, result)
    return {"output": args.output, "domains": list(references), "common_features": len(common)}
def command_audit(args: argparse.Namespace) -> dict:
    document, snapshot = document_path(args.manuscript), snapshot_path(args.manuscript, args.snapshot)
    if any(same_path(args.output, path) for path in (document, snapshot, snapshot.with_suffix(".json"))):
        raise ValueError("The audit output cannot overwrite a document or snapshot artifact")
    code = conventions(args.root, args.manuscript)
    result = {"conventions": code, "manuscript": manuscript_status(args.manuscript, args.snapshot, args.accept and code["passed"])}
    if args.manifest:
        result["data"] = audit_splits(normalize(read_rows(args.manifest)), args.check_files)
    write_json(args.output, result)
    if not result["conventions"]["passed"]:
        raise ValueError("Code conventions failed; inspect the audit report")
    return {"code": result["conventions"], "manuscript_changed": result["manuscript"]["changed"], "formulas": result["manuscript"]["formulas"], "output": args.output}
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FlawPrint forensic research workflow.")
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Normalize a JSONL, JSON, CSV, or TSV manifest")
    prepare.add_argument("input")
    prepare.add_argument("--output", default="./data/manifest.jsonl")
    prepare.add_argument("--image-root", default=".")
    prepare.add_argument("--labels", choices=("fake-one", "real-one"), default="fake-one")
    prepare.add_argument("--domain-key", default="domain")
    prepare.add_argument("--encoding", default="utf-8-sig")
    prepare.add_argument("--delimiter")
    prepare.add_argument("--records-key")
    prepare.set_defaults(handler=command_prepare)
    validation = commands.add_parser("validate", help="Inspect canonical data, split coverage, and image readability before extraction")
    validation.add_argument("input")
    validation.add_argument("--check-images", action="store_true")
    validation.add_argument("--benchmark", action="store_true")
    validation.add_argument("--min-samples", type=int, default=20)
    validation.add_argument("--max-side", type=int, default=768)
    validation.add_argument("--min-side", type=int, default=32)
    validation.add_argument("--tile-grid", type=int, default=4)
    validation.add_argument("--output", default="./outputs/preflight.json")
    validation.set_defaults(handler=command_validate)
    split = commands.add_parser("split", help="Build group-disjoint source and target partitions")
    split.add_argument("input")
    split.add_argument("--target-domain", action="append", required=True)
    split.add_argument("--output", default="./data/split.jsonl")
    split.add_argument("--profile-fraction", type=float, default=0.35)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--check-files", action="store_true")
    split.set_defaults(handler=command_split)
    join = commands.add_parser("join", help="Join independently produced semantic logits by sample ID")
    join.add_argument("input")
    join.add_argument("predictions")
    join.add_argument("--score-key", default="logit")
    join.add_argument("--real-positive", action="store_true")
    join.add_argument("--output", default="./data/semantic_manifest.jsonl")
    join.set_defaults(handler=command_join)
    extraction = commands.add_parser("extract", help="Compute deterministic image and narrative traces")
    extraction.add_argument("input")
    extraction.add_argument("--output", default="./data/features.npz")
    extraction.add_argument("--max-side", type=int, default=768)
    extraction.add_argument("--min-side", type=int, default=32)
    extraction.add_argument("--tile-grid", type=int, default=4)
    extraction.add_argument("--workers", type=int, default=1)
    extraction.set_defaults(handler=command_extract)
    profile = commands.add_parser("profile", help="Fit a domain copula from an independent sample pool")
    profile.add_argument("features")
    profile.add_argument("--domain", required=True)
    profile.add_argument("--split", choices=("train", "profile"), default="profile")
    profile.add_argument("--config", default="./configs/default.json")
    profile.add_argument("--genuine-validation")
    profile.add_argument("--selector-predictions")
    profile.add_argument("--selector-metadata")
    profile.add_argument("--selector-threshold", type=float, default=0.05)
    profile.add_argument("--alpha", type=float, default=0.05)
    profile.add_argument("--output", default="./models/profile.json")
    profile.set_defaults(handler=command_profile)
    predict = commands.add_parser("predict", help="Score samples using a frozen domain profile")
    predict.add_argument("features")
    predict.add_argument("--profile", required=True)
    predict.add_argument("--domain")
    predict.add_argument("--split", default="test")
    predict.add_argument("--calibration")
    predict.add_argument("--fusion", action="store_true")
    predict.add_argument("--semantic-threshold", type=float, default=0.0)
    predict.add_argument("--explain", action="store_true")
    predict.add_argument("--no-signs", action="store_true")
    predict.add_argument("--output", default="./outputs/predictions.jsonl")
    predict.set_defaults(handler=command_predict)
    deployment = commands.add_parser("deploy", help="Gate physical evidence and route available channels")
    deployment.add_argument("features")
    deployment.add_argument("--profile")
    deployment.add_argument("--domain")
    deployment.add_argument("--split", default="test")
    deployment.add_argument("--calibration")
    deployment.add_argument("--semantic-threshold", type=float, default=0.0)
    deployment.add_argument("--min-observed", type=float, default=0.8)
    deployment.add_argument("--min-image-features", type=int, default=2)
    deployment.add_argument("--gate-alpha", type=float, default=0.05)
    deployment.add_argument("--explain", action="store_true")
    deployment.add_argument("--no-signs", action="store_true")
    deployment.add_argument("--output", default="./outputs/deployment/predictions.jsonl")
    deployment.add_argument("--report", default="./outputs/deployment/report.json")
    deployment.set_defaults(handler=command_deploy)
    calibration = commands.add_parser("calibrate", help="Fit source probabilities and labeled split conformal")
    calibration.add_argument("--validation", required=True)
    calibration.add_argument("--calibration", required=True)
    calibration.add_argument("--score-key", default="fusion_risk")
    calibration.add_argument("--routed", action="store_true")
    calibration.add_argument("--alpha", type=float, default=0.1)
    calibration.add_argument("--output", default="./models/calibration.json")
    calibration.set_defaults(handler=command_calibrate)
    evaluation = commands.add_parser("evaluate", help="Report discrimination, calibration, and selective risk")
    evaluation.add_argument("predictions")
    evaluation.add_argument("--score-key", default="fusion_risk")
    evaluation.add_argument("--bootstrap", type=int, default=200)
    evaluation.add_argument("--group-key", default="domain")
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--output", default="./outputs/evaluation.json")
    evaluation.set_defaults(handler=command_evaluate)
    comparison = commands.add_parser("benchmark", help="Run controls, FlawPrint, and optional ablations")
    comparison.add_argument("features")
    comparison.add_argument("--config", default="./configs/default.json")
    comparison.add_argument("--output", default="./outputs/benchmark")
    comparison.add_argument("--bootstrap", type=int, default=200)
    comparison.add_argument("--alpha", type=float, default=0.1)
    comparison.add_argument("--ablations", action="store_true")
    comparison.set_defaults(handler=command_benchmark)
    theory = commands.add_parser("theory", help="Check identities and synthetic sampling predictions")
    theory.add_argument("--samples", type=int, default=100000)
    theory.add_argument("--repetitions", type=int, default=40)
    theory.add_argument("--seed", type=int, default=42)
    theory.add_argument("--output", default="./outputs/theory.json")
    theory.set_defaults(handler=command_theory)
    invariance = commands.add_parser("invariance", help="Check F25 ranks, ties, and frozen-reference pairing identities")
    invariance.add_argument("--samples", type=int, default=4000)
    invariance.add_argument("--dimensions", type=int, default=4)
    invariance.add_argument("--seed", type=int, default=42)
    invariance.add_argument("--repetitions", type=int, default=32)
    invariance.add_argument("--output", default="./outputs/invariance.json")
    invariance.set_defaults(handler=command_invariance)
    mma = commands.add_parser("mma", help="Construct an explicitly marked feature-space adversary")
    mma.add_argument("features")
    mma.add_argument("--domain")
    mma.add_argument("--split", default="test")
    attack_mode = mma.add_mutually_exclusive_group()
    attack_mode.add_argument("--shuffle", action="store_true")
    attack_mode.add_argument("--joint", action="store_true")
    mma.add_argument("--seed", type=int, default=42)
    mma.add_argument("--output", default="./outputs/mma/features.npz")
    mma.add_argument("--report", default="./outputs/mma/report.json")
    mma.set_defaults(handler=command_mma)
    dependence = commands.add_parser("dependence", help="Evaluate tail asymmetry, permutations, and availability")
    dependence.add_argument("features")
    dependence.add_argument("--profile", required=True)
    dependence.add_argument("--domain")
    dependence.add_argument("--split", default="test")
    dependence.add_argument("--tail-threshold", type=float, default=0.1)
    dependence.add_argument("--repeats", type=int, default=500)
    dependence.add_argument("--alpha", type=float, default=0.05)
    dependence.add_argument("--batch-size", type=int, default=1)
    dependence.add_argument("--seed", type=int, default=42)
    dependence.add_argument("--student", action="store_true")
    dependence.add_argument("--output", default="./outputs/dependence.json")
    dependence.set_defaults(handler=command_dependence)
    streaming = commands.add_parser("stream", help="Pool historical copulas and shrink a cold-start estimate")
    streaming.add_argument("--history", action="append", required=True)
    streaming.add_argument("--features")
    streaming.add_argument("--domain", required=True)
    streaming.add_argument("--split", default="profile")
    streaming.add_argument("--n0", type=float, default=100.0)
    streaming.add_argument("--output", default="./models/stream.json")
    streaming.set_defaults(handler=command_stream)
    platform = commands.add_parser("platform", help="Compare dominant copula directions and noise attenuation")
    platform.add_argument("--profiles", action="append", required=True)
    platform.add_argument("--output", default="./outputs/platform.json")
    platform.set_defaults(handler=command_platform)
    domains = commands.add_parser("domains", help="Evaluate source-to-target degradation with a fixed control")
    domains.add_argument("features")
    domains.add_argument("--method", default="trace_logistic", choices=METHODS)
    domains.add_argument("--seed", type=int, default=42)
    domains.add_argument("--output", default="./outputs/domain_matrix.json")
    domains.set_defaults(handler=lambda args: save_result(args.output, domain_matrix(load_bundle(args.features), args.method, args.seed)))
    corruption = commands.add_parser("corrupt", help="Export reproducible propagation-pipeline perturbations")
    corruption.add_argument("input")
    corruption.add_argument("--operation", choices=("jpeg", "resize", "blur", "noise", "crop", "strip_metadata"), required=True)
    corruption.add_argument("--severity", type=float, required=True)
    corruption.add_argument("--output", default="./data/corrupted")
    corruption.add_argument("--seed", type=int, default=42)
    corruption.set_defaults(handler=lambda args: {"samples": len(corrupt(normalize(read_rows(args.input)), args.output, args.operation, args.severity, args.seed)), "output": args.output})
    audit = commands.add_parser("audit", help="Check code rules, data leakage, and manuscript changes")
    audit.add_argument("--root", default=".")
    audit.add_argument("--manifest")
    audit.add_argument("--check-files", action="store_true")
    audit.add_argument("--manuscript", help="Audit a relative document; default to the local manuscript or public README")
    audit.add_argument("--snapshot", help="Use a relative snapshot; public clones default to outputs/audit_snapshot.md")
    audit.add_argument("--accept", action="store_true")
    audit.add_argument("--output", default="./outputs/audit.json")
    audit.set_defaults(handler=command_audit)
    fixture = commands.add_parser("fixture", help="Create a deterministic synthetic feature fixture")
    fixture.add_argument("--output", default="./data/fixture.npz")
    fixture.add_argument("--seed", type=int, default=42)
    fixture.add_argument("--per-split", type=int, default=24)
    fixture.set_defaults(handler=lambda args: write_fixture(args.output, args.seed, args.per_split))
    demo = commands.add_parser("demo", help="Create artificial images for extraction integration checks")
    demo.add_argument("--output", default="./data/demo")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--per-split", type=int, default=16)
    demo.set_defaults(handler=lambda args: image_demo(args.output, args.seed, args.per_split))
    return parser.parse_args()
def save_result(path: str, result: dict) -> dict:
    write_json(path, result)
    return {"output": path}
def main() -> None:
    args = arguments()
    try:
        if hasattr(args, "output"):
            targets = [args.output] + ([args.report] if getattr(args, "report", None) else [])
            if len(targets) == 2 and same_path(*targets):
                raise ValueError("Output and report files must be distinct")
            for name in ("input", "predictions", "manifest", "features", "profile", "calibration", "validation", "genuine_validation", "selector_predictions", "selector_metadata", "config", "manuscript", "snapshot"):
                value = getattr(args, name, None)
                if isinstance(value, str) and any(same_path(value, target) for target in targets):
                    raise ValueError("The output path cannot overwrite an input file")
        result = args.handler(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"Error: {error}") from error
    import json
    from engine.storage import native
    print(json.dumps(native(result), ensure_ascii=True, indent=2, allow_nan=False))
if __name__ == "__main__":
    main()
