# -*- coding: utf-8 -*-
# Research Evaluation: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import argparse
from dataclasses import asdict
from engine.evidence import paired, mixture, mechanism, stratified, orientation
from engine.quotient import audit as quotient
from engine.selection import filtering
from engine.channels import compare as channel_comparison
from engine.experiment import indices
from engine.profile import ProfileConfig
from engine.provenance import source_digest
from engine.results import prediction_rows
from engine.storage import digest, load_bundle, read_json, read_rows, relative_path, same_path, subset, write_json
from doctor import environment
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired increments, evidence roles, profile sensitivity, and copula mechanisms.")
    commands = parser.add_subparsers(dest="command", required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("reference")
    comparison.add_argument("candidate")
    comparison.add_argument("--score-key", default="fusion_risk")
    comparison.add_argument("--group-key", choices=("domain", "group"), default="domain")
    comparison.add_argument("--repeats", type=int, default=1000)
    comparison.add_argument("--output", default="./outputs/research/comparison.json")
    strata = commands.add_parser("strata")
    strata.add_argument("reference")
    strata.add_argument("candidate")
    strata.add_argument("--annotations", required=True)
    strata.add_argument("--score-key", default="fusion_risk")
    strata.add_argument("--group-key", choices=("domain", "group"), default="domain")
    strata.add_argument("--repeats", type=int, default=1000)
    strata.add_argument("--output", default="./outputs/research/strata.json")
    sensitivity = commands.add_parser("mixture")
    sensitivity.add_argument("pool")
    sensitivity.add_argument("features")
    sensitivity.add_argument("--domain", required=True)
    sensitivity.add_argument("--config", default="./configs/default.json")
    sensitivity.add_argument("--fractions", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 0.75, 1])
    sensitivity.add_argument("--size", type=int, default=40)
    sensitivity.add_argument("--repeats", type=int, default=5)
    sensitivity.add_argument("--image-only", action="store_true")
    sensitivity.add_argument("--output", default="./outputs/research/mixture.json")
    demonstration = commands.add_parser("mechanism")
    demonstration.add_argument("--rho", type=float, default=0.8)
    demonstration.add_argument("--samples", type=int, default=5000)
    demonstration.add_argument("--pool-size", type=int, default=1000)
    demonstration.add_argument("--repeats", type=int, default=3)
    demonstration.add_argument("--fractions", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    demonstration.add_argument("--shrinkage", type=float, default=0.1)
    demonstration.add_argument("--output", default="./outputs/research/mechanism.json")
    direction = commands.add_parser("orientation")
    direction.add_argument("matrices")
    direction.add_argument("--fractions", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    direction.add_argument("--output", default="./outputs/research/orientation.json")
    geometry = commands.add_parser("quotient")
    geometry.add_argument("matrices")
    geometry.add_argument("--output", default="./outputs/research/quotient.json")
    selection = commands.add_parser("filtering")
    selection.add_argument("pool")
    selection.add_argument("features")
    selection.add_argument("--selector-predictions", required=True)
    selection.add_argument("--selector-metadata", required=True)
    selection.add_argument("--selector-threshold", type=float, default=0.05)
    selection.add_argument("--domain", required=True)
    selection.add_argument("--config", default="./configs/default.json")
    selection.add_argument("--fractions", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 0.9, 1])
    selection.add_argument("--size", type=int, default=40)
    selection.add_argument("--repeats", type=int, default=5)
    selection.add_argument("--output", default="./outputs/research/filtering.json")
    channels = commands.add_parser("channels")
    channels.add_argument("semantic")
    channels.add_argument("physical")
    channels.add_argument("deep")
    channels.add_argument("--provenance", required=True)
    channels.add_argument("--group-key", choices=("domain", "group"), default="group")
    channels.add_argument("--repeats", type=int, default=200)
    channels.add_argument("--alpha", type=float, default=0.1)
    channels.add_argument("--regularization", type=float, default=1e-4)
    channels.add_argument("--output", default="./outputs/research/channels.json")
    for command in (comparison, strata, sensitivity, demonstration, direction, geometry, selection, channels):
        command.add_argument("--scope", choices=("synthetic",) if command in (demonstration, direction) else ("synthetic", "real"), required=True)
        command.add_argument("--dataset", required=True)
        command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        inputs = [getattr(args, field) for field in ("reference", "candidate", "annotations", "pool", "features", "config", "matrices", "selector_predictions", "selector_metadata", "semantic", "physical", "deep", "provenance") if hasattr(args, field)]
        if any(same_path(path, args.output) for path in inputs):
            raise ValueError("Research output cannot overwrite an input")
        fingerprints = [{"path": path, "sha256": digest(path)} for path in inputs]
        if args.command == "compare":
            result = paired(prediction_rows(read_rows(args.reference)), prediction_rows(read_rows(args.candidate)), args.score_key, args.repeats, args.seed, args.group_key)
        elif args.command == "strata":
            result = stratified(read_rows(args.reference), read_rows(args.candidate), read_rows(args.annotations), args.score_key, args.repeats, args.seed, args.group_key)
        elif args.command == "mixture":
            bundle, pool = load_bundle(args.features), load_bundle(args.pool)
            test = subset(bundle, indices(bundle, "test", args.domain))
            config = ProfileConfig(**read_json(args.config))
            result = mixture(pool, test, args.domain, config, args.fractions, args.size, args.repeats, args.seed, args.image_only)
            result["configuration"] = asdict(config)
        elif args.command == "orientation":
            matrices = read_json(args.matrices)
            result = orientation(matrices["genuine"], matrices["fake"], args.fractions)
        elif args.command == "quotient":
            result = quotient(read_json(args.matrices))
        elif args.command == "filtering":
            pool, bundle = load_bundle(args.pool), load_bundle(args.features)
            test = subset(bundle, indices(bundle, "test", args.domain))
            config = ProfileConfig(**read_json(args.config))
            result = filtering(pool, test, read_rows(args.selector_predictions), read_json(args.selector_metadata), args.domain, config, args.fractions, args.size, args.repeats, args.seed, args.selector_threshold)
            result["configuration"] = asdict(config)
        elif args.command == "channels":
            result = channel_comparison(read_rows(args.semantic), read_rows(args.physical), read_rows(args.deep), read_json(args.provenance), args.repeats, args.seed, args.group_key, args.alpha, args.regularization)
        else:
            result = mechanism(args.rho, args.samples, args.pool_size, args.repeats, args.fractions, args.shrinkage, args.seed)
        if any(digest(item["path"]) != item["sha256"] for item in fingerprints):
            raise ValueError("An input changed during the research run")
        result.update(dataset=args.dataset, evidence_scope=args.scope, inputs=fingerprints, source_sha256=source_digest(), environment=environment())
        write_json(args.output, result)
        print(f"Saved the {args.command} evidence to {args.output}")
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
if __name__ == "__main__":
    main()
