# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import argparse
from dataclasses import replace
import json
import math
import os
import statistics
import subprocess
import sys
from PIL.Image import DecompressionBombError
from engine.storage import relative_path, resolved, same_path, read_json, write_json, read_rows, write_rows, load_bundle, digest, protect_outputs, staged_outputs
from engine.data import normalize, missing
from engine.provenance import source_digest
from engine.package import configuration, configuration_record, is_builtin

SEEDS = (42, 52, 3047, 3407)

def seed_summary(records, seeds=SEEDS):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("Seeds must be distinct unsigned 32-bit integers")
    if any(not isinstance(row, dict) or type(row.get("seed")) is not int or "metrics" not in row or not isinstance(row.get("protocol"), dict) for row in records):
        raise ValueError("Every seed result needs metrics and an experiment protocol")
    if len(records) != len(seeds) or {row["seed"] for row in records} != set(seeds):
        raise ValueError("Missing or duplicate seed results; partial averages are forbidden")
    if any(row["protocol"] != records[0]["protocol"] for row in records):
        raise ValueError("All seeds must use the same inputs and experiment settings")
    def flatten(value, prefix=""):
        result = {}
        if isinstance(value, dict):
            for key, item in value.items():
                result.update(flatten(item, prefix + ("." if prefix else "") + str(key)))
        elif value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
            if value is not None and not math.isfinite(value):
                raise ValueError("Seed metrics must be finite or explicitly undefined")
            result[prefix] = value
        return result
    byseed = {str(row["seed"]): row["metrics"] for row in records}
    flat = {seed: flatten(byseed[str(seed)]) for seed in seeds}
    metrics = {}
    for key in sorted(set().union(*(value.keys() for value in flat.values()))):
        values = [flat[seed].get(key) for seed in seeds]
        undefined = [seed for seed, value in zip(seeds, values) if value is None]
        metrics[key] = {"mean": statistics.mean(values) if not undefined else None, "sample_std": statistics.stdev(values) if not undefined and len(values) > 1 else None, "undefined_seeds": undefined}
    return {"seeds": seeds, "seed_count": len(seeds), "byseed": byseed, "metrics": metrics, "protocol": records[0]["protocol"], "aggregation": "Compute each seed's test metrics first, then take their equally weighted arithmetic mean."}

def experiment_commands(args):
    seeds = list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("Seeds must be distinct unsigned 32-bit integers")
    inputs = [relative_path(args.features), relative_path(args.embeddings)]
    inputs.extend(relative_path(value) for value in (args.config, args.profile_config) if not is_builtin(value))
    root = relative_path(args.output)
    outputs = [root / "summary.json"]
    commands = []
    for seed in seeds:
        directory = root / ("seed" + str(seed))
        model, prediction = directory / "model", directory / "predictions.jsonl"
        if any(resolved(path).is_relative_to(resolved(model)) for path in inputs):
            raise ValueError("Seed model directories cannot contain input files")
        if model.exists() and (not model.is_dir() or any(model.iterdir())):
            raise ValueError("Use fresh or empty model directories for a new seed experiment")
        report = prediction.with_name(prediction.stem + ".report.json")
        outputs.extend((prediction, report, directory / "result.json"))
        train = [sys.executable, "-B", "deep.py", "train", str(inputs[0]), str(inputs[1]), "--config", args.config, "--profile-config", args.profile_config, "--seed", str(seed), "--dataset", args.dataset, "--device", args.device, "--output", model.as_posix()]
        if args.mode is not None:
            train.extend(("--mode", args.mode))
        predict = [sys.executable, "-B", "deep.py", "predict", str(inputs[0]), str(inputs[1]), "--model", model.as_posix(), "--split", "test", "--device", args.device, "--output", prediction.as_posix()]
        commands.append((seed, train, predict))
    protect_outputs(outputs, inputs)
    return commands, inputs

def run_experiment(args):
    commands, inputs = experiment_commands(args)
    if args.dry_run:
        return {"seeds": list(args.seeds), "commands": [{"seed": seed, "train": train[1:], "predict": predict[1:]} for seed, train, predict in commands]}
    fingerprints = {path.as_posix(): digest(path) for path in inputs}
    expected = load_bundle(args.features)
    expected = sorted((row["id"], row["label"], row["domain"]) for row in expected["rows"] if row["split"] == "test")
    if not expected:
        raise ValueError("A seed experiment requires a complete labeled test partition")
    training, profile = configuration(args.config), configuration(args.profile_config)
    training.pop("seed", None)
    profile.pop("seed", None)
    if args.mode is not None:
        training["mode"] = args.mode
    protocol = {"inputs": fingerprints, "training": training, "profile": profile, "dataset": args.dataset, "test_identities": expected}
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    records = []
    for seed, train, predict in commands:
        if any(digest(path) != checksum for path, checksum in fingerprints.items()):
            raise ValueError("An experiment input changed between seeds")
        subprocess.run(train, check=True, env=environment)
        directory = relative_path(args.output) / ("seed" + str(seed))
        metadata = read_json(directory / "model" / "model.json")
        if type(metadata.get("config", {}).get("seed")) is not int or metadata["config"]["seed"] != seed:
            raise ValueError("The selected checkpoint has a different training seed")
        subprocess.run(predict, check=True, env=environment)
        rows = read_rows(directory / "predictions.jsonl")
        if sorted((row.get("id"), row.get("label"), row.get("domain")) for row in rows) != expected:
            raise ValueError("All seeds must evaluate every frozen test identity exactly once")
        if any(digest(path) != checksum for path, checksum in fingerprints.items()):
            raise ValueError("An experiment input changed during a seed run")
        record = {"seed": seed, "protocol": protocol, "metrics": read_json(directory / "predictions.report.json")}
        seed_summary([record], [seed])
        write_json(directory / "result.json", record)
        records.append(record)
    result = seed_summary(records, args.seeds)
    write_json(relative_path(args.output) / "summary.json", result)
    return result

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pretrained representations and train reference-coupled attention.")
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract")
    extraction.add_argument("manifest")
    for name in ("bert", "mae", "clip"):
        extraction.add_argument("--" + name, required=True)
    extraction.add_argument("--forensic", help="Optional local forensic CLIP checkpoint in Hugging Face format")
    extraction.add_argument("--batch-size", type=int, default=64)
    extraction.add_argument("--max-length", type=int, default=128)
    extraction.add_argument("--output", default="./data/deep.npz")
    training = commands.add_parser("train")
    training.add_argument("features")
    training.add_argument("embeddings")
    training.add_argument("--config", default="builtin:neural")
    training.add_argument("--profile-config", default="builtin:default")
    training.add_argument("--seed", type=int)
    training.add_argument("--mode", type=str.lower, choices=("coupled", "unbiased", "independent", "deep", "concat", "uniform", "broadcast", "diagonal", "shuffled", "summary"))
    training.add_argument("--dataset", type=str.lower, choices=("weibo", "weibo21", "finefake"))
    training.add_argument("--output", default="./models/neural")
    prediction = commands.add_parser("predict")
    prediction.add_argument("features")
    prediction.add_argument("embeddings")
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--split", default="test")
    prediction.add_argument("--reference-domain")
    prediction.add_argument("--explain", action="store_true", help="Export attention contrasts and fixed-checkpoint potential-off, uniform, and summary responses")
    prediction.add_argument("--output", default="./outputs/neural/predictions.jsonl")
    experiment = commands.add_parser("experiment", help="Train and test four seeds, then average seed metrics")
    experiment.add_argument("features")
    experiment.add_argument("embeddings")
    experiment.add_argument("--config", default="builtin:neural")
    experiment.add_argument("--profile-config", default="builtin:default")
    experiment.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    experiment.add_argument("--mode", type=str.lower, choices=("coupled", "unbiased", "independent", "deep", "concat", "uniform", "broadcast", "diagonal", "shuffled", "summary"))
    experiment.add_argument("--dataset", type=str.lower, choices=("weibo", "weibo21", "finefake"), required=True)
    experiment.add_argument("--output", default="./outputs/experiment")
    experiment.add_argument("--dry-run", action="store_true")
    for command in (extraction, training, prediction, experiment):
        command.add_argument("--device", default="cuda:0")
    return parser.parse_args()

def main() -> None:
    args = arguments()
    try:
        if args.command == "experiment":
            print(json.dumps(run_experiment(args), indent=2, allow_nan=False))
            return
        from engine import embeddings
        for name in ("manifest", "features", "embeddings", "output", "model", "bert", "mae", "clip", "forensic", "config", "profile_config"):
            value = getattr(args, name, None)
            if value is not None and not (name in ("config", "profile_config") and is_builtin(value)):
                setattr(args, name, relative_path(value).as_posix())
        protected = [getattr(args, name) for name in ("manifest", "features", "embeddings", "config", "profile_config") if getattr(args, name, None) and not (name in ("config", "profile_config") and is_builtin(getattr(args, name)))]
        if any(same_path(args.output, path) for path in protected):
            raise ValueError("The output cannot overwrite an input")
        fingerprints = [{"path": path, "sha256": digest(path)} for path in protected]
        if args.command == "extract":
            directories = {name: getattr(args, name) for name in ("bert", "mae", "clip", "forensic") if getattr(args, name, None)}
            if any(resolved(args.output).is_relative_to(resolved(path)) for path in directories.values()):
                raise ValueError("Embedding output must remain outside encoder directories")
            rows = normalize(read_rows(args.manifest))
            protect_outputs([args.output], [row["image"] for row in rows if row["image"]])
            archive = embeddings.extract(rows, directories, args.batch_size, args.device, args.max_length)
            if any(digest(item["path"]) != item["sha256"] for item in fingerprints):
                raise ValueError("An input changed during feature extraction")
            archive["provenance"].update(inputs=fingerprints, source_sha256=source_digest())
            embeddings.save(args.output, archive)
            result = {"samples": len(archive["rows"]), "streams": {name: list(values.shape) for name, values in archive["features"].items()}, "output": args.output}
        elif args.command == "train":
            from engine.learning import fit
            from engine.neural import NeuralConfig
            from engine.profile import ProfileConfig
            config = NeuralConfig(**configuration(args.config))
            if args.seed is not None:
                config = replace(config, seed=args.seed)
            if args.mode is not None:
                config = replace(config, mode=args.mode)
            profile = replace(ProfileConfig(**configuration(args.profile_config)), seed=config.seed)
            provenance = {"inputs": fingerprints, "configurations": {"training": configuration_record(args.config), "profile": configuration_record(args.profile_config)}, "source_sha256": source_digest(), "dataset": args.dataset, "inputs_changed_during_training": False}
            result = fit(load_bundle(args.features), embeddings.load(args.embeddings), args.output, config, profile, args.device, provenance)
        else:
            from engine.learning import predict
            from engine.metrics import evaluate
            if resolved(args.output).is_relative_to(resolved(args.model)):
                raise ValueError("Predictions must remain outside the model directory")
            predictions = predict(load_bundle(args.features), embeddings.load(args.embeddings), args.model, args.split, args.device, args.reference_domain, args.explain)
            if any(digest(item["path"]) != item["sha256"] for item in fingerprints):
                raise ValueError("An input changed during prediction")
            report = relative_path(args.output).with_name(relative_path(args.output).stem + ".report.json")
            if any(same_path(report, path) for path in protected) or resolved(report).is_relative_to(resolved(args.model)):
                raise ValueError("The prediction report cannot overwrite an input or model artifact")
            protect_outputs([args.output, report], [row["image"] for row in predictions if row.get("image")])
            # Validate report inputs before replacing existing predictions.
            metrics = evaluate(predictions) if all(not missing(row.get("label")) for row in predictions) else None
            if metrics is None and report.exists():
                raise ValueError("Choose a fresh output path for unlabeled predictions when a report already exists")
            with staged_outputs([args.output] + ([report] if metrics is not None else [])) as stages:
                write_rows(stages[0], predictions)
                if metrics is not None:
                    write_json(stages[1], metrics)
            result = {"samples": len(predictions), "output": args.output}
            if metrics is not None:
                result["report"] = report.as_posix()
        print(json.dumps(result, indent=2, allow_nan=False))
    except ImportError as error:
        raise SystemExit('Install optional deep dependencies with python -m pip install "torch>=2.2" "transformers>=4.40,<6"') from error
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, DecompressionBombError) as error:
        raise SystemExit("Error: " + str(error)) from error

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
