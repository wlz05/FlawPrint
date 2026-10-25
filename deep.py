# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import argparse
from dataclasses import replace
import json
from engine.storage import relative_path, resolved, same_path, read_json, write_json, read_rows, write_rows, load_bundle, digest
from engine.provenance import source_digest

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pretrained representations and train reference-coupled attention.")
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract")
    extraction.add_argument("manifest")
    for name in ("bert", "mae", "clip"):
        extraction.add_argument("--" + name, required=True)
    extraction.add_argument("--forensic", help="Optional local forensic CLIP checkpoint in Hugging Face format")
    extraction.add_argument("--batch-size", type=int, default=64)
    extraction.add_argument("--max-length", type=int, default=197)
    extraction.add_argument("--output", default="./data/deep.npz")
    training = commands.add_parser("train")
    training.add_argument("features")
    training.add_argument("embeddings")
    training.add_argument("--config", default="./configs/neural.json")
    training.add_argument("--profile-config", default="./configs/default.json")
    training.add_argument("--seed", type=int)
    training.add_argument("--mode", type=str.lower, choices=("coupled", "unbiased", "independent", "deep", "concat", "broadcast", "diagonal", "shuffled"))
    training.add_argument("--dataset", type=str.lower, choices=("weibo", "weibo21", "finefake"))
    training.add_argument("--output", default="./models/neural")
    prediction = commands.add_parser("predict")
    prediction.add_argument("features")
    prediction.add_argument("embeddings")
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--split", default="test")
    prediction.add_argument("--reference-domain")
    prediction.add_argument("--explain", action="store_true", help="Export the fixed-checkpoint response to disabling the attention potential")
    prediction.add_argument("--output", default="./outputs/neural/predictions.jsonl")
    for command in (extraction, training, prediction):
        command.add_argument("--device", default="cuda:0")
    return parser.parse_args()

def main() -> None:
    args = arguments()
    try:
        from engine import embeddings
        protected = [getattr(args, name) for name in ("manifest", "features", "embeddings", "config", "profile_config") if getattr(args, name, None)]
        if any(same_path(args.output, path) for path in protected):
            raise ValueError("The output cannot overwrite an input")
        fingerprints = [{"path": path, "sha256": digest(path)} for path in protected]
        if args.command == "extract":
            directories = {name: getattr(args, name) for name in ("bert", "mae", "clip", "forensic") if getattr(args, name, None)}
            if any(resolved(args.output).is_relative_to(resolved(path)) for path in directories.values()):
                raise ValueError("Embedding output must remain outside encoder directories")
            archive = embeddings.extract(read_rows(args.manifest), directories, args.batch_size, args.device, args.max_length)
            if any(digest(item["path"]) != item["sha256"] for item in fingerprints):
                raise ValueError("An input changed during feature extraction")
            archive["provenance"].update(inputs=fingerprints, source_sha256=source_digest())
            embeddings.save(args.output, archive)
            result = {"samples": len(archive["rows"]), "streams": {name: list(values.shape) for name, values in archive["features"].items()}, "output": args.output}
        elif args.command == "train":
            from engine.learning import fit
            from engine.neural import NeuralConfig
            from engine.profile import ProfileConfig
            config = NeuralConfig(**read_json(args.config))
            if args.seed is not None:
                config = replace(config, seed=args.seed)
            if args.mode is not None:
                config = replace(config, mode=args.mode)
            profile = replace(ProfileConfig(**read_json(args.profile_config)), seed=config.seed)
            result = fit(load_bundle(args.features), embeddings.load(args.embeddings), args.output, config, profile, args.device)
            changed = any(digest(item["path"]) != item["sha256"] for item in fingerprints)
            write_json(relative_path(args.output) / "provenance.json", {"inputs": fingerprints, "source_sha256": source_digest(), "dataset": args.dataset, "inputs_changed_during_training": changed})
            if changed:
                raise ValueError("An input changed during training; its original hashes are preserved")
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
            write_rows(args.output, predictions)
            result = {"samples": len(predictions), "output": args.output}
            if all("label" in row for row in predictions):
                write_json(report, evaluate(predictions))
                result["report"] = report.as_posix()
        print(json.dumps(result, indent=2, allow_nan=False))
    except ImportError as error:
        raise SystemExit("Install optional deep dependencies with python -m pip install -r ./requirements-optional.txt") from error
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        raise SystemExit("Error: " + str(error)) from error

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
