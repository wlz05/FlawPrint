# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
from dataclasses import asdict, replace
import os
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import numpy as np
from engine import embeddings
from engine.copula import score
from engine.experiment import indices
from engine.profile import DomainProfile, ProfileConfig
from engine.storage import subset, write_rows, read_rows, read_json, save_bundle, write_json, digest
from engine.synthetic import fixture
try:
    import torch
    from engine.neural import NeuralConfig, ResidualAttention, ResidualBlock, potential, scramble, contrast, summarize, transport, sensitivity
    from engine.learning import prepare, relation, fit, predict, standardize, logits, scored
except ImportError:
    torch = None

@unittest.skipIf(torch is None, "Optional PyTorch dependency is unavailable")
class NeuralTests(unittest.TestCase):
    def setUp(self) -> None:
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        Path("./tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir="./tmp")
        self.root = Path(self.temporary.name).relative_to(Path.cwd())
        self.bundle = fixture(seed=42, per_split=12)
        self.bundle["observed"][:] = True
        for row in self.bundle["rows"]:
            row["image"] = "./images/" + row["id"] + ".jpg"
            row["has_image"] = True
        rng = np.random.default_rng(52)
        features = {name: rng.normal(size=(len(self.bundle["rows"]), size)).astype(np.float32) for name, size in zip(embeddings.STREAMS, (8, 6, 4, 4))}
        features["bert"][:, 0] += np.asarray([3 * (2 * row.get("label", 0) - 1) for row in self.bundle["rows"]])
        self.archive = {"features": features, "observed": np.ones((len(self.bundle["rows"]), 4), dtype=bool), "rows": copy.deepcopy(self.bundle["rows"]), "provenance": {"encoders": {name: {"sha256": "a" * 64, "frozen": True} for name in ("bert", "mae", "clip")}}}
        self.profile = ProfileConfig(bootstrap=2, min_samples=4, seed=42)
        self.config = NeuralConfig(dimension=16, heads=2, layers=1, dropout=0.0, epochs=12, batch_size=12, learning_rate=0.003, patience=12, seed=42)

    def tearDown(self) -> None:
        self.temporary.cleanup()
        torch.set_num_threads(self.threads)

    def source_data(self, roles: tuple = ("train", "validation", "calibration", "test")) -> tuple[dict, dict]:
        selected = [index for index, row in enumerate(self.bundle["rows"]) if row["domain"].startswith("source") and row["split"] in roles]
        bundle = subset(self.bundle, selected)
        archive = copy.deepcopy(self.archive)
        archive["rows"] = [archive["rows"][index] for index in selected]
        archive["features"] = {name: values[selected] for name, values in archive["features"].items()}
        archive["observed"] = archive["observed"][selected]
        return bundle, archive

    def test_three_partition_training_preserves_roles_and_uses_uncalibrated_outputs(self) -> None:
        bundle, archive = self.source_data(("train", "validation", "test"))
        before = copy.deepcopy(bundle["rows"])
        output = self.root / "official"
        with patch("torch.optim.AdamW", wraps=torch.optim.AdamW) as optimizer:
            result = fit(bundle, archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        self.assertEqual(optimizer.call_args.kwargs["weight_decay"], 0.01)
        metadata = read_json(output / "model.json")
        self.assertEqual(metadata["config"]["weight_decay"], 0.01)
        self.assertEqual(bundle["rows"], before)
        self.assertIsNone(metadata["calibration"])
        self.assertEqual(metadata["calibration_protocol"], "uncalibrated")
        self.assertEqual(result["calibration_samples"], 0)
        self.assertEqual(metadata["partition_counts"], {role: sum(row["split"] == role for row in before) for role in ("train", "validation", "calibration", "profile", "test")})
        rows = predict(bundle, archive, output.as_posix(), device="cpu")
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in before if row["split"] == "test"])
        for row in rows:
            self.assertFalse(row["probability_calibrated"])
            self.assertIsNone(row["prediction_set"])
            self.assertFalse(row["abstain"])
            self.assertEqual(row["prediction"], int(row["fusion_risk"] >= 0))

    def test_experiment_runs_and_summarizes_all_four_seed_tests(self) -> None:
        import deep
        from engine.metrics import evaluate
        bundle, archive = self.source_data(("train", "validation", "test"))
        physical, vectors = self.root / "physical.npz", self.root / "vectors.npz"
        save_bundle(physical, bundle["values"], bundle["observed"], bundle["patches"], bundle["rows"], bundle["names"])
        embeddings.save(vectors, archive)
        args = SimpleNamespace(features=physical.as_posix(), embeddings=vectors.as_posix(), config="builtin:neural", profile_config="builtin:default", seeds=list(deep.SEEDS), dataset="weibo", mode="coupled", device="cpu", output=(self.root / "experiment").as_posix(), dry_run=False)
        expected = [row for row in bundle["rows"] if row["split"] == "test"]
        def launch(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            if command[3] == "train":
                seed = int(command[command.index("--seed") + 1])
                output.mkdir(parents=True)
                write_json(output / "model.json", {"config": {"seed": seed}})
            else:
                model = command[command.index("--model") + 1]
                seed = read_json(Path(model) / "model.json")["config"]["seed"]
                rows = []
                for index, row in enumerate(expected):
                    prediction = 1 - row["label"] if index < deep.SEEDS.index(seed) else row["label"]
                    rows.append(dict(row, prediction=prediction, fusion_risk=float(2 * prediction - 1)))
                write_rows(output, rows)
                write_json(output.with_name(output.stem + ".report.json"), evaluate(rows))
        with patch("deep.subprocess.run", side_effect=launch) as process:
            result = deep.run_experiment(args)
        self.assertEqual(process.call_count, 8)
        self.assertEqual(result["seeds"], list(deep.SEEDS))
        means = [result["byseed"][str(seed)]["accuracy"] for seed in deep.SEEDS]
        self.assertAlmostEqual(result["metrics"]["accuracy"]["mean"], sum(means) / 4)
        self.assertTrue((Path(args.output) / "summary.json").is_file())
        with self.assertRaisesRegex(ValueError, "fresh"):
            deep.experiment_commands(args)

    def test_four_training_seeds_keep_the_same_frozen_partitions(self) -> None:
        bundle, archive = self.source_data(("train", "validation", "test"))
        before = copy.deepcopy(bundle["rows"])
        identities = [row["id"] for role in ("train", "validation") for row in before if row["split"] == role]
        for seed in (42, 52, 3047, 3407):
            with self.subTest(seed=seed):
                output = self.root / ("seed" + str(seed))
                fit(bundle, archive, output.as_posix(), replace(self.config, seed=seed, epochs=1), self.profile, "cpu")
                metadata = read_json(output / "model.json")
                self.assertEqual(metadata["config"]["seed"], seed)
                self.assertEqual(metadata["fit_identifiers"], identities)
                self.assertEqual(bundle["rows"], before)

    def test_optional_calibration_does_not_change_the_trained_checkpoint(self) -> None:
        plain, plain_archive = self.source_data(("train", "validation", "test"))
        calibrated, calibrated_archive = self.source_data()
        outputs = self.root / "plain", self.root / "calibrated"
        for bundle, archive, output in ((plain, plain_archive, outputs[0]), (calibrated, calibrated_archive, outputs[1])):
            fit(bundle, archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        first, second = [torch.load(output / "parameters.pt", map_location="cpu", weights_only=True) for output in outputs]
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        one = predict(plain, plain_archive, outputs[0].as_posix(), device="cpu")
        two = predict(calibrated, calibrated_archive, outputs[1].as_posix(), device="cpu")
        np.testing.assert_array_equal([row["fusion_risk"] for row in one], [row["fusion_risk"] for row in two])
        self.assertTrue(all(row["probability_calibrated"] and row["prediction_set"] is not None for row in two))

    def test_training_still_requires_source_train_and_validation_roles(self) -> None:
        for roles in (("train", "test"), ("validation", "test")):
            bundle, archive = self.source_data(roles)
            with self.subTest(roles=roles):
                with self.assertRaisesRegex(ValueError, "requires train and validation"):
                    fit(bundle, archive, (self.root / "missing").as_posix(), self.config, self.profile, "cpu")

    def test_builtin_neural_defaults_match_the_runtime_dataclass(self) -> None:
        from inspect import signature
        from engine.package import configuration
        self.assertEqual(configuration("builtin:neural"), asdict(NeuralConfig()))
        self.assertEqual(configuration("builtin:experiment")["seeds"], [42, 52, 3047, 3407])
        self.assertEqual(NeuralConfig().weight_decay, 0.01)
        self.assertEqual(signature(embeddings.extract).parameters["max_length"].default, 128)
        self.assertEqual(embeddings.context(128, 512, 77), 77)

    def test_cache_alignment_requires_matching_well_formed_image_fingerprints(self) -> None:
        for physical, deep in ((None, "a" * 64), ("a" * 64, None), ("a" * 64, "b" * 64), ("invalid", "invalid"), (False, False)):
            bundle, archive = copy.deepcopy(self.bundle), copy.deepcopy(self.archive)
            if physical is not None:
                bundle["rows"][0]["content_sha256"] = physical
            if deep is not None:
                archive["rows"][0]["content_sha256"] = deep
            with self.subTest(physical=physical, deep=deep):
                with self.assertRaisesRegex(ValueError, "different image bytes"):
                    embeddings.align(bundle, archive)
        self.bundle["rows"][0]["content_sha256"] = "a" * 64
        self.archive["rows"][0]["content_sha256"] = "a" * 64
        embeddings.align(self.bundle, self.archive)

    def test_selected_epoch_restores_the_first_best_validation_checkpoint(self) -> None:
        bundle, archive = self.source_data(("train", "validation", "test"))
        states = []
        def observed(model, *arguments, **options):
            states.append({name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
            return logits(model, *arguments, **options)
        output = self.root / "selected"
        with patch("engine.learning.logits", side_effect=observed), patch("engine.learning.binary", side_effect=[{"macro_f1": value} for value in (.8, .3, .8)]):
            result = fit(bundle, archive, output.as_posix(), replace(self.config, epochs=3, patience=2), self.profile, "cpu")
        metadata = read_json(output / "model.json")
        self.assertEqual(result["selected_epoch"], 1)
        self.assertEqual(metadata["selected_epoch"], 1)
        self.assertEqual(result["epochs"], 3)
        self.assertTrue(any(not torch.equal(states[0][name], states[2][name]) for name in states[0]))
        restored = torch.load(output / "parameters.pt", map_location="cpu", weights_only=True)
        for name in restored:
            torch.testing.assert_close(restored[name], states[0][name], rtol=0, atol=0)
            torch.testing.assert_close(states[3][name], states[0][name], rtol=0, atol=0)

    def test_invalid_prediction_report_path_preserves_existing_predictions(self) -> None:
        import deep
        physical, vectors, output = self.root / "physical.npz", self.root / "vectors.npz", self.root / "predictions.jsonl"
        for path in (physical, vectors):
            path.write_bytes(b"fixture")
        write_rows(output, [{"id": "retained"}])
        report = self.root / "predictions.report.json"
        report.mkdir()
        before = output.read_bytes()
        args = SimpleNamespace(command="predict", features=physical.as_posix(), embeddings=vectors.as_posix(), output=output.as_posix(), model=(self.root / "model").as_posix(), split="test", device="cpu", reference_domain=None, explain=False)
        with patch("deep.arguments", return_value=args), patch("deep.load_bundle", return_value=self.bundle), patch("engine.embeddings.load", return_value=self.archive), patch("engine.learning.predict", return_value=[{"id": "news", "label": 0}]), patch("engine.metrics.evaluate", side_effect=AssertionError("Invalid output paths must fail before evaluation")):
            with self.assertRaisesRegex(SystemExit, "regular files"):
                deep.main()
        self.assertEqual(output.read_bytes(), before)
        self.assertTrue(report.is_dir())

    def test_embedding_roundtrip_and_identity_alignment(self) -> None:
        path = self.root / "features.npz"
        embeddings.save(path.as_posix(), self.archive)
        restored = embeddings.load(path.as_posix())
        order = np.arange(len(restored["rows"]))[::-1]
        restored["rows"] = [restored["rows"][index] for index in order]
        restored["features"] = {name: value[order] for name, value in restored["features"].items()}
        restored["observed"] = restored["observed"][order]
        aligned, mask = embeddings.align(self.bundle, restored)
        np.testing.assert_array_equal(aligned["bert"], self.archive["features"]["bert"])
        np.testing.assert_array_equal(mask, self.archive["observed"])
        restored["rows"][0]["text"] = "A different news story."
        with self.assertRaisesRegex(ValueError, "metadata"):
            embeddings.align(self.bundle, restored)

    def test_invalid_embedding_values_and_masks_are_rejected(self) -> None:
        for damage in ("nan", "mask", "missing"):
            archive = copy.deepcopy(self.archive)
            if damage == "nan":
                archive["features"]["mae"][0, 0] = np.nan
            elif damage == "mask":
                archive["observed"] = archive["observed"].astype(float)
            else:
                archive["observed"][0, 0] = False
            with self.assertRaises(ValueError):
                embeddings.validate(archive)

    def test_failed_cache_replacement_preserves_the_existing_archive(self) -> None:
        path = self.root / "cache.npz"
        embeddings.save(path.as_posix(), self.archive)
        original = path.read_bytes()
        with patch("engine.embeddings.np.savez_compressed", side_effect=OSError("Interrupted cache write")):
            with self.assertRaisesRegex(OSError, "Interrupted"):
                embeddings.save(path.as_posix(), self.archive)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(self.root.glob(".flawprint-*.tmp")))
        embeddings.validate(embeddings.load(path.as_posix()))

    def test_invalid_npz_archives_fail_with_an_explicit_format_error(self) -> None:
        valid = self.root / "valid.npz"
        embeddings.save(valid.as_posix(), self.archive)
        for damage in ("truncated", "array", "duplicate", "metadata", "missing"):
            with self.subTest(damage=damage):
                path = self.root / (damage + ".npz")
                if damage == "truncated":
                    path.write_bytes(valid.read_bytes()[:40])
                elif damage == "array":
                    with path.open("wb") as stream:
                        np.save(stream, np.zeros((2, 2)))
                else:
                    metadata = '{"version":"bad","version":"flawprint-embeddings-v1"}' if damage == "duplicate" else "[]"
                    if damage == "missing":
                        metadata = json.dumps({"version": "flawprint-embeddings-v1", "streams": list(embeddings.STREAMS)})
                    np.savez_compressed(path, metadata=np.asarray(metadata))
                original = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "Invalid embedding archive"):
                    embeddings.load(path.as_posix())
                self.assertEqual(path.read_bytes(), original)

    def test_utf8_byte_metadata_loads_with_the_recorded_stream_order(self) -> None:
        path = self.root / "encoded.npz"
        embeddings.save(path.as_posix(), self.archive)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["metadata"] = np.asarray(arrays["metadata"].item().encode("utf-8"))
        np.savez_compressed(path, **arrays)
        restored = embeddings.load(path.as_posix())
        self.assertEqual(list(restored["features"]), list(embeddings.STREAMS))
        np.testing.assert_array_equal(restored["features"]["bert"], self.archive["features"]["bert"])

    def test_embedding_metadata_rejects_overflow_in_unknown_fields(self) -> None:
        path = self.root / "overflow.npz"
        embeddings.save(path.as_posix(), self.archive)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["metadata"] = np.asarray('{"extra":1e999,' + arrays["metadata"].item()[1:])
        np.savez_compressed(path, **arrays)
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "finite"):
            embeddings.load(path.as_posix())
        self.assertEqual(path.read_bytes(), original)

    def test_empty_extraction_fails_before_optional_encoder_imports(self) -> None:
        with patch.dict(sys.modules, {"transformers": None}):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                embeddings.extract([], {name: "./missing" for name in ("bert", "mae", "clip")}, device="cpu")

    def test_blank_samples_fail_before_loading_pretrained_encoders(self) -> None:
        with patch.dict(sys.modules, {"transformers": None}):
            with self.assertRaisesRegex(ValueError, "every sample"):
                embeddings.extract([{"id": "empty", "text": "   "}], {name: "./missing" for name in ("bert", "mae", "clip")}, device="cpu")

    def test_deep_extraction_protects_original_images_before_model_loading(self) -> None:
        import deep
        from PIL import Image
        image = self.root / "image.png"
        Image.new("RGB", (64, 64), (30, 80, 100)).save(image)
        manifest = self.root / "manifest.jsonl"
        write_rows(manifest, [{"id": "news", "text": "Report.", "image": image.as_posix()}])
        before = image.read_bytes()
        args = SimpleNamespace(command="extract", manifest=manifest.as_posix(), bert="./missing", mae="./missing", clip="./missing", forensic=None, batch_size=1, device="cpu", max_length=8, output=image.as_posix())
        with patch("deep.arguments", return_value=args), patch("engine.embeddings.extract", side_effect=AssertionError("Model loading must not start")):
            with self.assertRaisesRegex(SystemExit, "overwrite"):
                deep.main()
        self.assertEqual(image.read_bytes(), before)
        args.output = (self.root / "features.npz").as_posix()
        with patch("deep.arguments", return_value=args), patch("engine.embeddings.extract", side_effect=Image.DecompressionBombError("Image exceeds the pixel limit")):
            with self.assertRaisesRegex(SystemExit, "pixel limit"):
                deep.main()
        self.assertFalse(Path(args.output).exists())

    def test_cuda_errors_precede_training_and_explain_the_device_choice(self) -> None:
        output = self.root / "cuda"
        with patch("torch.cuda.is_available", return_value=False), patch("engine.learning.prepare", side_effect=AssertionError("Reference fitting must not start")):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                fit(self.bundle, self.archive, output.as_posix(), self.config, self.profile, "cuda:0")
        self.assertFalse(output.exists())
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            with self.assertRaisesRegex(ValueError, "device index"):
                embeddings.runtime("cuda:1")
        self.assertEqual(embeddings.runtime("cpu"), "cpu")

    def test_malformed_cache_metadata_and_float_overflow_are_rejected(self) -> None:
        for damage in ("identifier", "context", "ratio", "readout", "overflow", "container"):
            with self.subTest(damage=damage):
                archive = copy.deepcopy(self.archive)
                if damage == "identifier":
                    archive["rows"][0]["id"] = " "
                elif damage == "context":
                    archive["provenance"].update(max_text_length=8, text_context={"bert": 12, "clip": 8})
                elif damage == "ratio":
                    archive["provenance"]["mae_mask_ratio"] = False
                elif damage == "readout":
                    archive["provenance"]["readout"] = {"bert": "class_token"}
                elif damage == "overflow":
                    archive["features"]["bert"] = archive["features"]["bert"].astype(np.float64)
                    archive["features"]["bert"][0, 0] = 1e300
                else:
                    archive["features"]["bert"] = archive["features"]["bert"].tolist()
                with self.assertRaises(ValueError):
                    embeddings.validate(archive)

    def test_training_rejects_boolean_and_nonfinite_numeric_settings(self) -> None:
        for name in ("dropout", "bias_limit", "learning_rate", "weight_decay", "auxiliary_weight"):
            for value in (False, True, float("nan"), float("inf"), "0.1"):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        replace(self.config, **{name: value}).validate()
        with self.assertRaises(ValueError):
            replace(self.config, seed=2 ** 32).validate()

    def test_standardization_rejects_observed_overflow_and_ignores_missing_streams(self) -> None:
        values = np.array([[0.], [0.], [3e38]], dtype=np.float32)
        packet = {"features": {"bert": values}, "observed": np.ones((3, 1), dtype=bool)}
        with self.assertRaisesRegex(ValueError, "float32 range"):
            standardize(packet, [0, 1])
        packet["features"]["bert"] = np.array([[3e38], [3e38], [0.]], dtype=np.float32)
        packet["observed"][2, 0] = False
        result, state = standardize(packet, [0, 1])
        self.assertTrue(np.isfinite(result["features"]["bert"]).all())
        self.assertEqual(result["features"]["bert"][2, 0], 0)

    def test_checkpoint_rejects_changed_feature_protocol_and_loads_legacy_metadata(self) -> None:
        archive = copy.deepcopy(self.archive)
        archive["provenance"].update(max_text_length=128, mae_mask_ratio=0.0, text_context={"bert": 128, "clip": 77}, readout={name: "class_token" if name in ("bert", "mae") else "projected_vector" for name in embeddings.STREAMS})
        directory = self.root / "protocol"
        fit(self.bundle, archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        before = predict(self.bundle, archive, directory.as_posix(), device="cpu")
        self.assertTrue(all(row["embedding_protocol_verified"] for row in before))
        changed = copy.deepcopy(archive)
        changed["provenance"].update(max_text_length=64, text_context={"bert": 64, "clip": 64})
        with self.assertRaisesRegex(ValueError, "preprocessing protocol"):
            predict(self.bundle, changed, directory.as_posix(), device="cpu")
        metadata = read_json(directory / "model.json")
        metadata.pop("embedding_protocol")
        write_json(directory / "model.json", metadata)
        legacy = predict(self.bundle, archive, directory.as_posix(), device="cpu")
        self.assertFalse(any(row["embedding_protocol_verified"] for row in legacy))
        np.testing.assert_array_equal([row["fusion_risk"] for row in before], [row["fusion_risk"] for row in legacy])

    def test_checkpoint_rejects_malformed_metadata_before_inference(self) -> None:
        directory = self.root / "metadata"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        original = read_json(directory / "model.json")
        cases = [None, [], 1, {}, {key: value for key, value in original.items() if key != "config"}]
        for key, value in (("config", []), ("config", {"unexpected": 1}), ("profiles", []), ("profile_sha256", {}), ("fit_history", None), ("fit_identifiers", [["invalid"]]), ("fit_identifiers", ["a", "a"]), ("feature_order", {}), ("trace_names", ["x", "x"]), ("normalizers", []), ("embedding_protocol", []), ("calibration", {})):
            damaged = copy.deepcopy(original)
            damaged[key] = value
            cases.append(damaged)
        for metadata in cases:
            with self.subTest(metadata=metadata):
                write_json(directory / "model.json", metadata)
                with patch.object(ResidualAttention, "forward", side_effect=AssertionError("Inference must not start")):
                    with self.assertRaises(ValueError):
                        predict(self.bundle, self.archive, directory.as_posix(), device="cpu")
        write_json(directory / "model.json", original)
        self.assertTrue(predict(self.bundle, self.archive, directory.as_posix(), device="cpu"))

    def test_checkpoint_rejects_corrupt_or_nonfinite_weights_before_inference(self) -> None:
        directory = self.root / "weights"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        before = predict(self.bundle, self.archive, directory.as_posix(), device="cpu")
        path = directory / "parameters.pt"
        original = path.read_bytes()
        metadata = read_json(directory / "model.json")
        state = torch.load(path, map_location="cpu", weights_only=True)
        name = next(iter(state))
        for damage in ("empty", "invalid", "truncated", "container", "missing", "extra", "shape", "dtype", "nontensor", "nan", "inf"):
            with self.subTest(damage=damage):
                damaged = copy.deepcopy(state)
                if damage in ("empty", "invalid", "truncated"):
                    path.write_bytes({"empty": b"", "invalid": b"invalid checkpoint", "truncated": original[:80]}[damage])
                else:
                    if damage == "container":
                        damaged = []
                    elif damage == "missing":
                        damaged.pop(name)
                    elif damage == "extra":
                        damaged["unexpected"] = torch.tensor(1.)
                    elif damage == "shape":
                        damaged[name] = damaged[name].flatten()[:1]
                    elif damage == "dtype":
                        damaged[name] = damaged[name].double()
                    elif damage == "nontensor":
                        damaged[name] = 0
                    else:
                        damaged[name].reshape(-1)[0] = float(damage)
                    torch.save(damaged, path)
                metadata["weights_sha256"] = digest(path)
                write_json(directory / "model.json", metadata)
                with patch.object(ResidualAttention, "forward", side_effect=AssertionError("Inference must not start")):
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        predict(self.bundle, self.archive, directory.as_posix(), device="cpu")
        path.write_bytes(original)
        metadata["weights_sha256"] = digest(path)
        write_json(directory / "model.json", metadata)
        after = predict(self.bundle, self.archive, directory.as_posix(), device="cpu")
        np.testing.assert_array_equal([row["fusion_risk"] for row in before], [row["fusion_risk"] for row in after])

    def test_checkpoint_detects_metadata_changed_during_loading(self) -> None:
        directory = self.root / "metadatachange"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        def changing(path, *arguments):
            result = read_json(path, *arguments)
            Path(path).write_bytes(Path(path).read_bytes() + b" ")
            return result
        with patch("engine.learning.read_json", side_effect=changing), patch.object(ResidualAttention, "forward", side_effect=AssertionError("Inference must not start")):
            with self.assertRaisesRegex(ValueError, "changed"):
                predict(self.bundle, self.archive, directory.as_posix(), device="cpu")

    def test_metadata_replacement_and_restoration_cannot_change_the_loaded_state(self) -> None:
        directory = self.root / "metadatasnapshot"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        def changing(path, *arguments):
            original = Path(path).read_bytes()
            document = read_json(path)
            document["normalizers"]["bert"]["center"][0] += 1
            write_json(path, document)
            try:
                return read_json(path, *arguments)
            finally:
                Path(path).write_bytes(original)
        with patch("engine.learning.read_json", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "changed|fingerprint"):
                predict(self.bundle, self.archive, directory.as_posix(), device="cpu")

    def test_reference_replacement_and_restoration_cannot_change_the_loaded_state(self) -> None:
        directory = self.root / "referencesnapshot"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        original = DomainProfile.load
        def changing(path, *arguments):
            content = Path(path).read_bytes()
            document = read_json(path)
            document["sorted_values"] = (np.asarray(document["sorted_values"]) + 1).tolist()
            write_json(path, document)
            try:
                return original(path, *arguments)
            finally:
                Path(path).write_bytes(content)
        with patch.object(DomainProfile, "load", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "changed|fingerprint"):
                predict(self.bundle, self.archive, directory.as_posix(), device="cpu")

    def test_weight_decoding_uses_the_verified_bytes(self) -> None:
        import io
        from engine.learning import parameters
        model = ResidualAttention({name: 4 for name in embeddings.STREAMS}, 3, self.config)
        expected = {name: value.clone() for name, value in model.state_dict().items()}
        path = self.root / "parameters.pt"
        torch.save(expected, path)
        content, checksum = path.read_bytes(), digest(path)
        alternate = copy.deepcopy(expected)
        name = next(name for name, value in alternate.items() if value.dtype.is_floating_point)
        alternate[name] += 1
        stream = io.BytesIO()
        torch.save(alternate, stream)
        original = torch.load
        def changing(source, **options):
            if not isinstance(source, (str, Path)):
                return original(source, **options)
            path.write_bytes(stream.getvalue())
            try:
                return original(source, **options)
            finally:
                path.write_bytes(content)
        with patch("engine.learning.torch.load", side_effect=changing):
            parameters(model, path, checksum)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_checkpoint_requires_all_saved_files_before_inference(self) -> None:
        directory = self.root / "completecopy"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        metadata = read_json(directory / "model.json")
        backup = self.root / "retained"
        files = [directory / "model.json", directory / "parameters.pt"] + [directory / name for name in metadata["profiles"].values()]
        for path in files:
            path.rename(backup)
            with self.subTest(file=path.name), patch.object(ResidualAttention, "forward", side_effect=AssertionError("Inference must not start")):
                try:
                    with self.assertRaisesRegex(ValueError, "missing|incomplete"):
                        predict(self.bundle, self.archive, directory.as_posix(), device="cpu", reference_domain="source_alpha")
                finally:
                    backup.rename(path)
        self.assertTrue(predict(self.bundle, self.archive, directory.as_posix(), device="cpu"))

    def test_checkpoint_rejects_external_duplicate_and_mislabeled_references(self) -> None:
        directory = self.root / "referencepaths"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        document = directory / "model.json"
        metadata = read_json(document)
        domains = list(metadata["profiles"])
        path = directory / metadata["profiles"][domains[0]]
        external = self.root / "external.json"
        shutil.copyfile(path, external)
        for filename in ("../external.json", external.resolve().as_posix(), metadata["profiles"][domains[1]]):
            damaged = copy.deepcopy(metadata)
            damaged["profiles"][domains[0]] = filename
            write_json(document, damaged)
            with self.subTest(path=filename), patch.object(ResidualAttention, "forward", side_effect=AssertionError("Inference must not start")):
                with self.assertRaises(ValueError):
                    predict(self.bundle, self.archive, directory.as_posix(), device="cpu")
        profile = read_json(path)
        profile["domain"] = "mislabeled"
        write_json(path, profile)
        metadata["profile_sha256"][domains[0]] = digest(path)
        write_json(document, metadata)
        with patch.object(ResidualAttention, "forward", side_effect=AssertionError("Inference must not start")):
            with self.assertRaisesRegex(ValueError, "domain does not match"):
                predict(self.bundle, self.archive, directory.as_posix(), device="cpu")

    def test_checkpoint_detects_references_changed_during_loading(self) -> None:
        directory = self.root / "referencechange"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        original = DomainProfile.load
        def changing(path, *arguments):
            document = read_json(path)
            document["sorted_values"] = (np.asarray(document["sorted_values"]) + 1).tolist()
            write_json(path, document)
            return original(path, *arguments)
        with patch.object(DomainProfile, "load", side_effect=changing), patch.object(ResidualAttention, "forward", side_effect=AssertionError("Inference must not start")):
            with self.assertRaisesRegex(ValueError, "changed"):
                predict(self.bundle, self.archive, directory.as_posix(), device="cpu")

    def test_model_changes_during_inference_preserve_previous_exports(self) -> None:
        import deep
        directory = self.root / "inferencechange"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        metadata = read_json(directory / "model.json")
        physical, vectors, output = self.root / "physical.npz", self.root / "vectors.npz", self.root / "predictions.jsonl"
        save_bundle(physical, self.bundle["values"], self.bundle["observed"], self.bundle["patches"], self.bundle["rows"], self.bundle["names"].tolist())
        embeddings.save(vectors.as_posix(), self.archive)
        report = output.with_name("predictions.report.json")
        write_rows(output, [{"id": "retained"}])
        write_json(report, {"retained": True})
        before = output.read_bytes(), report.read_bytes()
        args = SimpleNamespace(command="predict", features=physical.as_posix(), embeddings=vectors.as_posix(), output=output.as_posix(), model=directory.as_posix(), split="test", device="cpu", reference_domain=None, explain=False)
        files = [directory / "model.json", directory / "parameters.pt", directory / next(iter(metadata["profiles"].values()))]
        for path in files:
            original = path.read_bytes()
            def changing(*arguments, **options):
                values = logits(*arguments, **options)
                path.write_bytes(original + b" ")
                return values
            with self.subTest(file=path.name), patch("deep.arguments", return_value=args), patch("engine.learning.logits", side_effect=changing):
                try:
                    with self.assertRaisesRegex(SystemExit, "changed"):
                        deep.main()
                    self.assertEqual((output.read_bytes(), report.read_bytes()), before)
                finally:
                    path.write_bytes(original)
        self.assertFalse(list(self.root.glob(".flawprint-*")))

    def test_checkpoint_detects_weights_changed_during_loading(self) -> None:
        directory = self.root / "changing"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        original = torch.load
        path = directory / "parameters.pt"
        def changing(source, **options):
            state = original(source, **options)
            with path.open("ab") as stream:
                stream.write(b"changed")
            return state
        with patch("engine.learning.torch.load", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "changed while loading"):
                predict(self.bundle, self.archive, directory.as_posix(), device="cpu")

    def test_explanation_differences_promote_finite_float32_risks(self) -> None:
        directory = self.root / "differences"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        count = sum(row["split"] == "test" for row in self.bundle["rows"])
        maximum = np.float32(3e38)
        risks = [np.full(count, value, dtype=np.float32) for value in (maximum, -maximum, -maximum, -maximum)]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", RuntimeWarning)
            with patch("engine.learning.logits", side_effect=risks):
                rows = predict(self.bundle, self.archive, directory.as_posix(), device="cpu", explain=True)
        self.assertFalse(recorded)
        for row in rows:
            self.assertEqual(row["potential_risk_delta"], 2 * float(maximum))
            self.assertEqual(row["query_readout_delta"], 2 * float(maximum))
            self.assertEqual(row["pair_readout_delta"], 2 * float(maximum))

    def test_inference_rejects_nonfinite_or_misaligned_batch_outputs(self) -> None:
        model = ResidualAttention({name: 4 for name in embeddings.STREAMS}, 3, self.config)
        packet = {"features": {name: np.zeros((2, 4), dtype=np.float32) for name in embeddings.STREAMS}, "observed": np.ones((2, 4), dtype=bool), "ranks": np.zeros((2, 3), dtype=np.float32), "graph": np.zeros((2, 3, 3), dtype=np.float32), "trace_observed": np.zeros((2, 3), dtype=bool)}
        for risk in (None, [0., 1.], torch.tensor(0.), torch.zeros(2, 1), torch.zeros(1), torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.long), torch.tensor([0., float("nan")]), torch.tensor([0., float("inf")])):
            with self.subTest(risk=risk):
                with patch.object(model, "forward", return_value={"logit": risk}):
                    with self.assertRaisesRegex(ValueError, "inference"):
                        logits(model, packet, [0, 1], "cpu", 2)
        for size in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "batch size"):
                logits(model, packet, [0, 1], "cpu", size)
        np.testing.assert_array_equal(logits(model, packet, [], "cpu", 2), np.empty(0))

    def test_inference_diagnostics_reject_missing_or_invalid_batch_values(self) -> None:
        model = ResidualAttention({name: 4 for name in embeddings.STREAMS}, 3, self.config)
        packet = {"features": {name: np.zeros((2, 4), dtype=np.float32) for name in embeddings.STREAMS}, "observed": np.ones((2, 4), dtype=bool), "ranks": np.zeros((2, 3), dtype=np.float32), "graph": np.zeros((2, 3, 3), dtype=np.float32), "trace_observed": np.zeros((2, 3), dtype=bool)}
        for name, value in (("attention_shift_tv", torch.tensor([0., 1.1])), ("attention_contrast_rms", torch.tensor([-1., 0.])), ("attention_shift_tv", torch.tensor([float("nan"), 0.])), ("attention_shift_tv", torch.zeros(1)), ("attention_shift_tv", [0., 0.]), ("pair_relation_fraction", torch.tensor([0., 1.1])), ("pair_contrast_rms", torch.tensor([-1., 0.])), ("pair_attention_shift_tv", torch.tensor([float("nan"), 0.]))):
            diagnostics = {"retained": "value"}
            with patch.object(model, "forward", return_value={"logit": torch.zeros(2), name: value}):
                with self.assertRaisesRegex(ValueError, "diagnostics"):
                    logits(model, packet, [0, 1], "cpu", 2, diagnostics)
            self.assertEqual(diagnostics, {"retained": "value"})
        outputs = [{"logit": torch.zeros(1), "attention_shift_tv": torch.zeros(1)}, {"logit": torch.zeros(1)}]
        with patch.object(model, "forward", side_effect=outputs):
            with self.assertRaisesRegex(ValueError, "every selected sample"):
                logits(model, packet, [0, 1], "cpu", 1, {})

    def test_scoring_cannot_silently_drop_samples(self) -> None:
        for risks in (np.zeros(1), np.zeros(3), np.zeros((2, 1)), np.array([0., np.nan]), np.array([0., np.inf]), np.ones(2, dtype=bool), np.array([0j, 1j])):
            with self.subTest(risks=risks):
                with self.assertRaisesRegex(ValueError, "one finite risk"):
                    scored(self.bundle["rows"], [0, 1], risks)
        rows = scored(self.bundle["rows"], [1, 0], np.array([0.1, -0.1]))
        self.assertEqual([row["id"] for row in rows], [self.bundle["rows"][1]["id"], self.bundle["rows"][0]["id"]])

    def test_standardizer_rejects_malformed_stored_vectors(self) -> None:
        packet = {"features": {"bert": np.ones((2, 2), dtype=np.float32)}, "observed": np.ones((2, 1), dtype=bool)}
        for state in ([], {}, {"bert": []}, {"bert": {"center": [0., 0.]}}, {"bert": {"center": ["0", "0"], "scale": [1., 1.]}}, {"bert": {"center": [False, False], "scale": [1., 1.]}}, {"bert": {"center": [0., 0.], "scale": [1., 0.]}}):
            with self.subTest(state=state):
                with self.assertRaisesRegex(ValueError, "standardizer"):
                    standardize(packet, [], state)

    def test_potential_rejects_nonsquare_matrices_and_numeric_masks(self) -> None:
        values, mask, graph = torch.zeros(2, 2, 7, 7), torch.ones(2, 7, dtype=torch.bool), torch.zeros(2, 3, 3)
        for mode in ("coupled", "uniform", "broadcast"):
            for logits_value, mask_value, graph_value in ((values[..., :6], mask, graph), (values[:, 0], mask, graph), (values, mask.float(), graph), (values.long(), mask, graph), (values, mask, graph.double()), (values, mask, graph[:, :, :2]), (values, mask, torch.zeros(2, 0, 0))):
                with self.subTest(mode=mode):
                    with self.assertRaises(ValueError):
                        potential(logits_value, mask_value, graph_value, mode)
        for dimensions, traces in (({"bert": True}, 3), ({"bert": 4}, True), ({"bert": 4}, 3.5)):
            with self.assertRaises(ValueError):
                ResidualAttention(dimensions, traces, self.config)

    def test_relation_graph_reconstructs_the_copula_quadratic_form(self) -> None:
        training = subset(self.bundle, indices(self.bundle, "train", "source_alpha"))
        query = subset(self.bundle, indices(self.bundle, "validation", "source_alpha"))
        profile = DomainProfile.fit(training, "source_alpha", self.profile)
        ranks, graph, mask = relation(profile, query)
        values = profile.transform(query)
        expected = score(values, profile.reference) + np.linalg.slogdet(profile.reference)[1]
        np.testing.assert_allclose(graph.sum((1, 2)), expected, rtol=1e-5, atol=1e-4)
        self.assertTrue(np.isfinite(ranks).all() and mask.any())
        with self.assertRaisesRegex(ValueError, "overlaps"):
            relation(profile, training)

    def test_potential_changes_attention_and_receives_training_gradients(self) -> None:
        torch.manual_seed(42)
        model = ResidualAttention({name: 4 for name in embeddings.STREAMS}, 3, self.config)
        features = {name: torch.randn(5, 4) for name in embeddings.STREAMS}
        observed = torch.ones(5, 4, dtype=torch.bool)
        ranks, graph, trace_mask = torch.randn(5, 3), torch.randn(5, 3, 3), torch.ones(5, 3, dtype=torch.bool)
        graph = graph + graph.transpose(-1, -2)
        coupled = model(features, observed, ranks, graph, trace_mask)["logit"]
        model.config = replace(self.config, mode="unbiased")
        unbiased = model(features, observed, ranks, graph, trace_mask)["logit"]
        self.assertGreater(float((coupled - unbiased).abs().max().detach()), 1e-6)
        coupled.square().mean().backward()
        self.assertGreater(float(model.blocks[0].strength.grad.abs()), 0)

    def test_missing_streams_and_trace_values_cannot_change_predictions(self) -> None:
        torch.manual_seed(42)
        model = ResidualAttention({name: 4 for name in embeddings.STREAMS}, 3, self.config).eval()
        features = {name: torch.randn(5, 4) for name in embeddings.STREAMS}
        observed = torch.ones(5, 4, dtype=torch.bool)
        observed[:, 0] = False
        ranks, graph, mask = torch.randn(5, 3), torch.randn(5, 3, 3), torch.ones(5, 3, dtype=torch.bool)
        mask[:, 0] = False
        first = model(features, observed, ranks, graph, mask)["logit"]
        features["bert"] = torch.randn(5, 4) * 100
        ranks[:, 0], graph[:, 0, :], graph[:, :, 0] = 1000, 1000, 1000
        second = model(features, observed, ranks, graph, mask)["logit"]
        torch.testing.assert_close(first, second)

    def test_masked_nonfinite_inputs_are_neutral_in_every_neural_mode(self) -> None:
        features = {name: torch.randn(2, 4) for name in embeddings.STREAMS}
        observed = torch.tensor([[False, True, True, True], [True, False, True, False]])
        ranks, graph = torch.randn(2, 4), torch.randn(2, 4, 4)
        graph = graph + graph.transpose(-1, -2)
        mask = torch.tensor([[False, True, True, True], [False, False, False, False]])
        pairs = mask[:, :, None] & mask[:, None, :]
        for mode in ("coupled", "unbiased", "independent", "deep", "concat", "uniform", "broadcast", "diagonal", "shuffled", "summary"):
            model = ResidualAttention({name: 4 for name in features}, 4, replace(self.config, mode=mode)).eval()
            expected = model(features, observed, ranks, graph, mask, explain=True)
            for value in (float("nan"), float("inf"), -float("inf")):
                changed = {name: values.masked_fill(~observed[:, column, None], value) for column, (name, values) in enumerate(features.items())}
                damaged_ranks = torch.full_like(ranks, value) if mode == "deep" else ranks.masked_fill(~mask, value)
                damaged_graph = torch.full_like(graph, value) if mode == "deep" else graph.masked_fill(~pairs, value)
                actual = model(changed, observed, damaged_ranks, damaged_graph, mask, explain=True)
                for name in expected:
                    with self.subTest(mode=mode, value=value, output=name):
                        self.assertTrue(torch.isfinite(actual[name]).all())
                        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_masked_nonfinite_features_keep_training_gradients_finite(self) -> None:
        model = ResidualAttention({name: 4 for name in embeddings.STREAMS}, 4, self.config)
        observed = torch.tensor([[False, True, True, True], [False, True, True, True]])
        features = {name: torch.randn(2, 4).requires_grad_() for name in embeddings.STREAMS}
        features["bert"] = torch.full((2, 4), float("nan"), requires_grad=True)
        mask = torch.ones(2, 4, dtype=torch.bool)
        graph = torch.eye(4)[None].expand(2, -1, -1)
        model(features, observed, torch.zeros(2, 4), graph, mask)["logit"].square().mean().backward()
        self.assertTrue(all(value.grad is None or torch.isfinite(value.grad).all() for value in model.parameters()))
        torch.testing.assert_close(features["bert"].grad, torch.zeros_like(features["bert"]))

    def test_attention_helpers_ignore_masked_nonfinite_queries_and_relations(self) -> None:
        values = torch.randn(1, 2, 7, 7)
        observed = torch.tensor([[True, False, True, False, True, True, True]])
        graph = torch.randn(1, 4, 4)
        graph = graph + graph.transpose(-1, -2)
        damaged = values.masked_fill(~(observed[:, None, :, None] & observed[:, None, None, :]), float("nan"))
        trace_mask = observed[:, 3:]
        changed = graph.masked_fill(~(trace_mask[:, :, None] & trace_mask[:, None, :]), float("nan"))
        torch.testing.assert_close(contrast(damaged, observed), contrast(values, observed))
        for mode in ("coupled", "uniform", "broadcast", "diagonal", "shuffled", "summary"):
            torch.testing.assert_close(potential(damaged, observed, changed, mode), potential(values, observed, graph, mode))
        for arguments in ((None, observed, graph), (values, None, graph), (values, observed, None)):
            with self.assertRaises(ValueError):
                potential(*arguments, "coupled")
        for tensor, mask in ((None, observed), (values, None), (values.long(), observed)):
            with self.assertRaises(ValueError):
                contrast(tensor, mask)

    def test_content_queries_select_different_reference_interactions(self) -> None:
        logits = torch.zeros(1, 2, 7, 7, requires_grad=True)
        with torch.no_grad():
            logits[:, :, 0, 3] = 4
            logits[:, :, 1, 4] = 4
        relation = torch.tensor([[[0.1, 0.4, -0.2, 0.0], [0.4, -0.1, 0.0, -0.2], [-0.2, 0.0, 0.1, 0.3], [0.0, -0.2, 0.3, -0.1]]])
        observed = torch.ones(1, 7, dtype=torch.bool)
        result = potential(logits, observed, relation, "coupled")
        self.assertGreater(float((result[:, :, 0, 3:] - result[:, :, 1, 3:]).abs().max().detach()), 0.1)
        self.assertGreater(float(result[:, :, :3, :3].abs().max().detach()), 0.1)
        result[:, :, 0, 1].sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_reference_transport_is_bounded_and_missing_anchors_are_neutral(self) -> None:
        torch.manual_seed(52)
        graph = torch.randn(3, 4, 4)
        graph = (graph + graph.transpose(-1, -2)) / 2
        graph = graph / (1 + torch.linalg.matrix_norm(graph, keepdim=True))
        observed = torch.ones(3, 7, dtype=torch.bool)
        observed[0, 3:] = False
        observed[1, 4] = False
        values = torch.randn(3, 2, 7, 7)
        for mode in ("coupled", "diagonal", "shuffled"):
            bias = potential(values, observed, graph, mode)
            self.assertTrue(torch.isfinite(bias).all())
            self.assertLess(float(bias.abs().max()), 1)
            self.assertTrue(torch.all(bias[0] == 0))
            damaged = graph.clone()
            damaged[1, 1, :], damaged[1, :, 1] = 999, 999
            torch.testing.assert_close(bias, potential(values, observed, damaged, mode))

    def test_pair_structure_survives_equal_scalar_and_row_summaries(self) -> None:
        cycle = torch.tensor([[0., 1., -1., 0.], [1., 0., 0., -1.], [-1., 0., 0., 1.], [0., -1., 1., 0.]], dtype=torch.float64)
        graphs = []
        for sign in (-1, 1):
            covariance = torch.linalg.inv(torch.eye(4, dtype=torch.float64) + sign * 0.2 * cycle)
            scale = covariance.diagonal().sqrt()
            reference = covariance / (scale[:, None] * scale[None, :])
            graphs.append(torch.eye(4, dtype=torch.float64) - torch.linalg.inv(reference))
        first, second = graphs
        torch.testing.assert_close(first.sum(), second.sum())
        torch.testing.assert_close(first.sum(-1), second.sum(-1))
        torch.testing.assert_close(first.diagonal(), second.diagonal())
        torch.testing.assert_close(first.norm(), second.norm())
        graph = torch.stack(graphs)
        graph = graph / (1 + torch.linalg.matrix_norm(graph, keepdim=True))
        logits = torch.zeros(2, 1, 6, 6, dtype=torch.float64)
        logits[:, :, 0, 2], logits[:, :, 1, 3] = 8, 8
        observed = torch.ones(2, 6, dtype=torch.bool)
        lifted = potential(logits, observed, graph, "coupled")
        broadcast = potential(logits, observed, graph, "broadcast")
        self.assertGreater(float((lifted[0, :, :2, :2] - lifted[1, :, :2, :2]).abs().max()), 0.1)
        torch.testing.assert_close(broadcast[0, :, :2], broadcast[1, :, :2])
        uniform = potential(logits, observed, graph, "uniform")
        torch.testing.assert_close(uniform[0, :, :2], uniform[1, :, :2])
        reconstructed = potential(logits, observed, graph, "summary")
        torch.testing.assert_close(reconstructed[0], reconstructed[1])

    def test_summary_reconstruction_preserves_summaries_and_is_orthogonal(self) -> None:
        torch.manual_seed(42)
        original = torch.randn(3, 6, 6, dtype=torch.float64, requires_grad=True)
        relation = (original + original.transpose(-1, -2)) / 2
        mask = torch.tensor([[True] * 6, [True, False, True, True, False, True], [False] * 6])
        pairs = mask[:, :, None] & mask[:, None, :]
        values = relation.masked_fill(~pairs, 0)
        reconstructed = summarize(relation, mask)
        torch.testing.assert_close(reconstructed.diagonal(dim1=-2, dim2=-1), values.diagonal(dim1=-2, dim2=-1))
        torch.testing.assert_close(reconstructed.sum(-1), values.sum(-1))
        torch.testing.assert_close(reconstructed, reconstructed.transpose(-1, -2))
        torch.testing.assert_close(summarize(reconstructed, mask), reconstructed)
        remainder = values - reconstructed
        torch.testing.assert_close((reconstructed * remainder).sum((1, 2)), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertTrue(torch.all(reconstructed.norm(dim=(-2, -1)) <= values.norm(dim=(-2, -1)) + 1e-12))
        self.assertTrue(torch.all(reconstructed[~pairs] == 0))
        reconstructed.square().sum().backward()
        self.assertTrue(torch.isfinite(original.grad).all())

    def test_summary_reconstruction_handles_small_support_and_masked_garbage(self) -> None:
        values = torch.randn(4, 5, 5, dtype=torch.float64)
        values = (values + values.transpose(-1, -2)) / 2
        mask = torch.arange(5)[None] < torch.arange(4)[:, None]
        pairs = mask[:, :, None] & mask[:, None, :]
        expected = values.masked_fill(~pairs, 0)
        damaged = values.masked_fill(~pairs, float("nan"))
        torch.testing.assert_close(summarize(damaged, mask), expected, atol=1e-12, rtol=0)

    def test_summary_reconstruction_rejects_invalid_active_relations(self) -> None:
        graph, mask = torch.eye(4)[None], torch.ones(1, 4, dtype=torch.bool)
        for relation, observed in ((graph[:, :, :3], mask), (graph, mask.float()), (graph.long(), mask), (None, mask), (graph, None)):
            with self.assertRaises(ValueError):
                summarize(relation, observed)
        for value in (float("nan"), float("inf"), 1.0):
            damaged = graph.clone()
            damaged[0, 0, 1] = value
            with self.assertRaisesRegex(ValueError, "finite symmetric"):
                summarize(damaged, mask)

    def test_summary_reconstruction_rejects_overflow_in_finite_inputs(self) -> None:
        graph = torch.full((1, 6, 6), torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(ValueError, "overflowed"):
            summarize(graph, torch.ones(1, 6, dtype=torch.bool))

    def test_summary_cli_and_experiment_metadata_share_the_same_controls(self) -> None:
        import deep
        from engine.package import configuration
        with patch.object(sys, "argv", ["deep.py", "train", "./physical.npz", "./deep.npz", "--mode", "Summary"]):
            arguments = deep.arguments()
        self.assertEqual(arguments.mode, "summary")
        extension = configuration("builtin:experiment")["neural_extension"]
        self.assertIn("summary", extension["modes"])
        self.assertEqual(extension["core_comparisons"], {"pair_information": ["summary", "coupled"], "content_readout": ["uniform", "coupled"]})

    def test_summary_control_keeps_the_same_input_nodes_and_query_parameters(self) -> None:
        model = ResidualAttention({name: 4 for name in embeddings.STREAMS}, 4, self.config).eval()
        features = {name: torch.randn(2, 4) for name in embeddings.STREAMS}
        observed, mask = torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool)
        ranks, graph = torch.randn(2, 4), torch.randn(2, 4, 4)
        graph = graph + graph.transpose(-1, -2)
        seen = []
        hook = model.blocks[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        try:
            model(features, observed, ranks, graph, mask)
            model.config = replace(self.config, mode="summary")
            result = model(features, observed, ranks, graph, mask)
        finally:
            hook.remove()
        torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(result["logit"]).all())

    def test_relation_mismatch_preserves_active_spectrum_and_parameter_budget(self) -> None:
        torch.manual_seed(3047)
        relation = torch.randn(2, 4, 4)
        relation = relation + relation.transpose(-1, -2)
        mask = torch.tensor([[True, True, True, False], [False, True, False, False]])
        relation *= mask[:, :, None] * mask[:, None, :]
        permuted = scramble(relation, mask)
        torch.testing.assert_close(torch.linalg.eigvalsh(relation), torch.linalg.eigvalsh(permuted))
        torch.testing.assert_close(relation.norm(dim=(-2, -1)), permuted.norm(dim=(-2, -1)))
        self.assertFalse(torch.allclose(relation[0], permuted[0]))
        budgets = []
        for mode in ("coupled", "uniform", "broadcast", "diagonal", "shuffled", "summary"):
            model = ResidualAttention({name: 4 for name in embeddings.STREAMS}, 4, replace(self.config, mode=mode))
            budgets.append(sum(value.numel() for value in model.parameters() if value.requires_grad))
        self.assertEqual(len(set(budgets)), 1)

    def test_cross_fitted_packets_ignore_target_test_annotations(self) -> None:
        altered = copy.deepcopy(self.bundle)
        for row in altered["rows"]:
            if row["split"] == "test":
                row["label"] = 1 - row["label"]
                row["physical_manipulated"] = True
        first, _ = prepare(self.bundle, self.archive, self.profile, 2)
        second, _ = prepare(altered, self.archive, self.profile, 2)
        np.testing.assert_array_equal(first["graph"], second["graph"])
        np.testing.assert_array_equal(first["ranks"], second["ranks"])
        first_path, second_path = self.root / "first", self.root / "second"
        fit(self.bundle, self.archive, first_path.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        fit(altered, self.archive, second_path.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        first_state = torch.load(first_path / "parameters.pt", weights_only=True)
        second_state = torch.load(second_path / "parameters.pt", weights_only=True)
        for name in first_state:
            torch.testing.assert_close(first_state[name], second_state[name], rtol=0, atol=0)

    def test_neural_training_rejects_text_reused_across_splits(self) -> None:
        bundle, archive = copy.deepcopy(self.bundle), copy.deepcopy(self.archive)
        train = next(index for index, row in enumerate(bundle["rows"]) if row["split"] == "train")
        validation = next(index for index, row in enumerate(bundle["rows"]) if row["split"] == "validation")
        bundle["rows"][validation]["text"] = bundle["rows"][train]["text"]
        archive["rows"][validation]["text"] = bundle["rows"][train]["text"]
        with self.assertRaisesRegex(ValueError, "Cross-split leakage"):
            prepare(bundle, archive, self.profile, 2)

    def test_neural_reload_rejects_training_text_under_a_test_identity(self) -> None:
        output = self.root / "text"
        fit(self.bundle, self.archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        bundle, archive = copy.deepcopy(self.bundle), copy.deepcopy(self.archive)
        train = next(index for index, row in enumerate(bundle["rows"]) if row["split"] == "train")
        query = next(index for index, row in enumerate(bundle["rows"]) if row["split"] == "test")
        bundle["rows"][query]["text"] = bundle["rows"][train]["text"]
        archive["rows"][query]["text"] = bundle["rows"][train]["text"]
        with patch.object(ResidualAttention, "forward", side_effect=AssertionError("Inference must not start")):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                predict(bundle, archive, output.as_posix(), device="cpu")

    def test_neural_fit_reload_and_encoder_provenance(self) -> None:
        directory = (self.root / "model").as_posix()
        report = fit(self.bundle, self.archive, directory, self.config, self.profile, "cpu")
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertEqual(report["image_supervised_samples"], 0)
        predictions = predict(self.bundle, self.archive, directory, device="cpu", explain=True)
        self.assertEqual(len(predictions), len(indices(self.bundle, "test")))
        self.assertTrue(all(np.isfinite(row["fusion_risk"]) for row in predictions))
        self.assertTrue(all(np.isfinite(row["potential_risk_delta"]) for row in predictions))
        plain = predict(self.bundle, self.archive, directory, device="cpu")
        np.testing.assert_array_equal([row["fusion_risk"] for row in predictions], [row["fusion_risk"] for row in plain])
        np.testing.assert_array_equal([row["probability"] for row in predictions], [row["probability"] for row in plain])
        for row in predictions:
            self.assertTrue(np.isfinite(row["uniform_readout_risk"]))
            self.assertAlmostEqual(row["query_readout_delta"], row["fusion_risk"] - row["uniform_readout_risk"])
            self.assertTrue(np.isfinite(row["summary_readout_risk"]))
            self.assertAlmostEqual(row["pair_readout_delta"], row["fusion_risk"] - row["summary_readout_risk"])
            self.assertGreaterEqual(row["attention_contrast_rms"], 0)
            self.assertTrue(0 <= row["attention_shift_tv"] <= 1)
            self.assertTrue(0 <= row["pair_relation_fraction"] <= 1)
            self.assertGreaterEqual(row["pair_contrast_rms"], 0)
            self.assertTrue(0 <= row["pair_attention_shift_tv"] <= 1)
        self.assertTrue(all(row["checkpoint_version"] == "flawprint-neural-v2" for row in predictions))
        self.assertEqual(read_json(Path(directory) / "model.json")["feature_order"], list(embeddings.STREAMS))
        history = read_json(Path(directory) / "model.json")["history"]
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            predict(self.bundle, self.archive, directory, split="train", device="cpu")
        damaged = copy.deepcopy(self.archive)
        damaged["provenance"]["encoders"]["mae"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "encoder"):
            predict(self.bundle, damaged, directory, device="cpu")
        path = Path(directory) / "parameters.pt"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            predict(self.bundle, self.archive, directory, device="cpu")

    def test_interrupted_checkpoint_write_preserves_the_destination(self) -> None:
        def interrupted(parameters, path):
            Path(path).write_bytes(b"partial checkpoint")
            raise OSError("Interrupted checkpoint write")

        for preexisting in (False, True):
            with self.subTest(preexisting=preexisting):
                output = self.root / ("empty" if preexisting else "new")
                if preexisting:
                    output.mkdir()
                with patch("engine.learning.torch.save", side_effect=interrupted):
                    with self.assertRaisesRegex(OSError, "Interrupted checkpoint"):
                        fit(self.bundle, self.archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
                self.assertEqual(output.exists(), preexisting)
                self.assertFalse(output.exists() and list(output.iterdir()))
                self.assertFalse(list(self.root.glob(".flawprint-*")))

    def test_failed_final_checkpoint_rename_preserves_the_empty_destination(self) -> None:
        for error in (OSError, KeyboardInterrupt):
            for preexisting in (False, True):
                output = self.root / (error.__name__.lower() + ("empty" if preexisting else "new"))
                if preexisting:
                    output.mkdir()
                with self.subTest(error=error.__name__, preexisting=preexisting), patch.object(Path, "rename", side_effect=error("Publication interrupted")):
                    with self.assertRaisesRegex(error, "Publication interrupted"):
                        fit(self.bundle, self.archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
                self.assertEqual(output.exists(), preexisting)
                self.assertFalse(output.exists() and list(output.iterdir()))
                self.assertFalse(list(self.root.glob(".flawprint-*")))

    def test_checkpoint_published_before_interrupt_remains_reloadable(self) -> None:
        output = self.root / "published"
        original = Path.rename
        def interrupted(stage, target):
            original(stage, target)
            raise KeyboardInterrupt("Publication completed before interruption")
        with patch.object(Path, "rename", interrupted):
            with self.assertRaisesRegex(KeyboardInterrupt, "Publication completed"):
                fit(self.bundle, self.archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        self.assertTrue(predict(self.bundle, self.archive, output.as_posix(), device="cpu"))
        self.assertFalse(list(self.root.glob(".flawprint-*")))

    def test_training_interrupt_can_restart_deterministically(self) -> None:
        output, baseline = self.root / "restarted", self.root / "baseline"
        output.mkdir()
        original = torch.optim.AdamW.step
        def interrupted(optimizer, *arguments, **options):
            original(optimizer, *arguments, **options)
            raise KeyboardInterrupt("Training interrupted")
        with patch.object(torch.optim.AdamW, "step", interrupted):
            with self.assertRaisesRegex(KeyboardInterrupt, "Training interrupted"):
                fit(self.bundle, self.archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        self.assertEqual(list(output.iterdir()), [])
        self.assertFalse(list(self.root.glob(".flawprint-*")))
        for destination in (output, baseline):
            fit(self.bundle, self.archive, destination.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        states = [torch.load(destination / "parameters.pt", map_location="cpu", weights_only=True) for destination in (output, baseline)]
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)

    def test_checkpoint_publication_rejects_a_competing_output_owner(self) -> None:
        output = self.root / "contended"
        save = torch.save

        def competing(parameters, path):
            save(parameters, path)
            output.mkdir(exist_ok=True)
            (output / "owner.txt").write_text("Another run", encoding="utf-8")

        with patch("engine.learning.torch.save", side_effect=competing):
            with self.assertRaisesRegex(ValueError, "became occupied"):
                fit(self.bundle, self.archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        self.assertEqual([path.name for path in output.iterdir()], ["owner.txt"])
        self.assertEqual((output / "owner.txt").read_text(encoding="utf-8"), "Another run")
        self.assertFalse(list(self.root.glob(".flawprint-*")))

    def test_complete_checkpoint_can_replace_an_empty_output_directory(self) -> None:
        output = self.root / "prepared"
        output.mkdir()
        fit(self.bundle, self.archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        metadata = read_json(output / "model.json")
        self.assertTrue(metadata["weights_sha256"] and metadata["profiles"])
        self.assertTrue(predict(self.bundle, self.archive, output.as_posix(), device="cpu"))
        self.assertFalse(list(self.root.glob(".flawprint-*")))

    def test_changed_training_inputs_never_publish_a_checkpoint(self) -> None:
        source = self.root / "source.json"
        original = torch.save
        for preexisting in (False, True):
            with self.subTest(preexisting=preexisting):
                source.write_bytes(b"original")
                provenance = {"inputs": [{"path": source.as_posix(), "sha256": digest(source)}]}
                output = self.root / ("empty" if preexisting else "new")
                if preexisting:
                    output.mkdir()
                def changing(parameters, path):
                    original(parameters, path)
                    source.write_bytes(b"changed")
                with patch("engine.learning.torch.save", side_effect=changing):
                    with self.assertRaisesRegex(ValueError, "not published"):
                        fit(self.bundle, self.archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu", provenance)
                self.assertEqual(output.exists(), preexisting)
                self.assertFalse(output.exists() and list(output.iterdir()))
                self.assertFalse(list(self.root.glob(".flawprint-*")))

    def test_failed_provenance_write_preserves_the_empty_model_directory(self) -> None:
        output = self.root / "empty"
        output.mkdir()
        source = self.root / "source.json"
        source.write_bytes(b"original")
        provenance = {"inputs": [{"path": source.as_posix(), "sha256": digest(source)}]}
        original = write_json
        def interrupted(path, value):
            if Path(path).name == "provenance.json":
                raise OSError("Interrupted provenance write")
            return original(path, value)
        with patch("engine.learning.write_json", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "Interrupted provenance"):
                fit(self.bundle, self.archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu", provenance)
        self.assertEqual(list(output.iterdir()), [])
        self.assertFalse(list(self.root.glob(".flawprint-*")))

    def test_provenance_is_present_before_checkpoint_publication(self) -> None:
        source = self.root / "source.json"
        source.write_bytes(b"original")
        provenance = {"inputs": [{"path": source.as_posix(), "sha256": digest(source)}], "dataset": "weibo"}
        output = self.root / "complete"
        original = Path.rename
        def inspected(stage, target):
            self.assertEqual(read_json(stage / "provenance.json"), provenance)
            self.assertTrue((stage / "parameters.pt").is_file())
            self.assertTrue((stage / "model.json").is_file())
            return original(stage, target)
        with patch.object(Path, "rename", inspected):
            fit(self.bundle, self.archive, output.as_posix(), replace(self.config, epochs=1), self.profile, "cpu", provenance)
        self.assertEqual(read_json(output / "provenance.json"), provenance)
        self.assertTrue(predict(self.bundle, self.archive, output.as_posix(), device="cpu"))

    def test_stale_or_malformed_training_provenance_fails_before_fitting(self) -> None:
        source = self.root / "source.json"
        source.write_bytes(b"original")
        stale = {"inputs": [{"path": source.as_posix(), "sha256": digest(source)}]}
        source.write_bytes(b"changed")
        for provenance in ([], {}, {"inputs": [None]}, {"inputs": [{"path": source.as_posix(), "sha256": "invalid"}]}, stale):
            with self.subTest(provenance=provenance):
                with patch("engine.learning.prepare", side_effect=AssertionError("Reference fitting must not start")):
                    with self.assertRaises(ValueError):
                        fit(self.bundle, self.archive, (self.root / "invalid").as_posix(), self.config, self.profile, "cpu", provenance)
        self.assertFalse((self.root / "invalid").exists())

    def test_unavailable_physical_references_preserve_deep_predictions(self) -> None:
        directory = (self.root / "deepfallback").as_posix()
        fit(self.bundle, self.archive, directory, replace(self.config, epochs=1), replace(self.profile, min_samples=1000), "cpu")
        rows = predict(self.bundle, self.archive, directory, device="cpu")
        self.assertTrue(rows and all(not row["physical_reference_available"] for row in rows))
        self.assertTrue(all(np.isfinite(row["fusion_risk"]) for row in rows))

    def test_image_auxiliary_loss_requires_independent_image_attributes(self) -> None:
        bundle = copy.deepcopy(self.bundle)
        for index, row in enumerate(bundle["rows"]):
            if row["split"] == "train":
                row["physical_manipulated"] = bool(index % 2)
        result = fit(bundle, self.archive, (self.root / "attributes").as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        self.assertEqual(result["image_supervised_samples"], len(indices(bundle, "train")))
        disabled = fit(bundle, self.archive, (self.root / "disabled").as_posix(), replace(self.config, epochs=1, auxiliary_weight=0), self.profile, "cpu")
        self.assertEqual(disabled["image_supervised_samples"], 0)

    def test_reloaded_predictions_roundtrip_across_all_export_formats(self) -> None:
        import deep
        from engine.results import prediction_rows
        physical, vectors, directory = self.root / "physical.npz", self.root / "vectors.npz", self.root / "exportmodel"
        index = next(index for index, row in enumerate(self.bundle["rows"]) if row["split"] == "test")
        text = 'Report, with "quotes"\nand context. ' + self.bundle["rows"][index]["text"]
        for rows in (self.bundle["rows"], self.archive["rows"]):
            rows[index]["text"] = text
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, epochs=1), self.profile, "cpu")
        save_bundle(physical, self.bundle["values"], self.bundle["observed"], self.bundle["patches"], self.bundle["rows"], self.bundle["names"].tolist())
        embeddings.save(vectors.as_posix(), self.archive)
        expected = prediction_rows(predict(self.bundle, self.archive, directory.as_posix(), device="cpu", explain=True))
        checksum = digest(directory / "model.json")
        self.assertTrue(all(row["checkpoint_sha256"] == checksum for row in expected))
        reports = []
        for suffix in ("jsonl", "ndjson", "json", "csv", "tsv"):
            output = self.root / ("predictions." + suffix)
            arguments = ["deep.py", "predict", physical.as_posix(), vectors.as_posix(), "--model", directory.as_posix(), "--device", "cpu", "--explain", "--output", output.as_posix()]
            with self.subTest(format=suffix), patch.object(sys, "argv", arguments), patch("builtins.print"):
                deep.main()
                self.assertEqual(prediction_rows(read_rows(output)), expected)
                reports.append(read_json(output.with_name("predictions.report.json")))
        self.assertTrue(all(report == reports[0] for report in reports))
        self.assertFalse(list(self.root.glob(".flawprint-*")))

    def test_neural_cli_trains_and_predicts_from_aligned_caches(self) -> None:
        physical, vectors = self.root / "physical.npz", self.root / "vectors.npz"
        save_bundle(physical, self.bundle["values"], self.bundle["observed"], self.bundle["patches"], self.bundle["rows"], self.bundle["names"].tolist())
        embeddings.save(vectors.as_posix(), self.archive)
        configuration, profile = self.root / "neural.json", self.root / "profile.json"
        write_json(configuration, asdict(replace(self.config, epochs=1)))
        write_json(profile, asdict(self.profile))
        model, output = self.root / "checkpoint", self.root / "predictions.jsonl"
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-B", "deep.py", "train", physical.as_posix(), vectors.as_posix(), "--config", configuration.as_posix(), "--profile-config", profile.as_posix(), "--dataset", "Weibo21", "--device", "cpu", "--output", model.as_posix()]
        training = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
        self.assertEqual(training.returncode, 0, training.stderr)
        prediction = subprocess.run([sys.executable, "-B", "deep.py", "predict", physical.as_posix(), vectors.as_posix(), "--model", model.as_posix(), "--device", "cpu", "--explain", "--output", output.as_posix()], capture_output=True, text=True, env=environment, timeout=60)
        self.assertEqual(prediction.returncode, 0, prediction.stderr)
        self.assertEqual(read_json(output.with_name("predictions.report.json"))["samples"], len(indices(self.bundle, "test")))
        self.assertTrue(all("potential_risk_delta" in row and "query_readout_delta" in row and "attention_shift_tv" in row for row in read_rows(output)))
        provenance = read_json(model / "provenance.json")
        self.assertEqual(provenance["dataset"], "weibo21")
        self.assertFalse(provenance["inputs_changed_during_training"])

    def test_cached_training_and_prediction_survive_a_workspace_move(self) -> None:
        original, moved = self.root / "first machine", self.root / "moved experiment"
        original.mkdir()
        shutil.copy2("deep.py", original / "deep.py")
        shutil.copytree("engine", original / "engine", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        bundle, archive = copy.deepcopy(self.bundle), copy.deepcopy(self.archive)
        for rows in (bundle["rows"], archive["rows"]):
            for row in rows:
                row["image"] = "assets/" + row["id"] + ".jpg"
        save_bundle(original / "data" / "physical.npz", bundle["values"], bundle["observed"], bundle["patches"], bundle["rows"], bundle["names"].tolist())
        embeddings.save((original / "data" / "vectors.npz").as_posix(), archive)
        write_json(original / "settings" / "neural.json", asdict(replace(self.config, epochs=1)))
        write_json(original / "settings" / "profile.json", asdict(self.profile))
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
        bootstrap = "import runpy, sys; sys.modules['transformers'] = None; runpy.run_path('deep.py', run_name='__main__')"
        training = ["train", "data\\physical.npz", "data\\vectors.npz", "--config", "settings\\neural.json", "--profile-config", "settings\\profile.json", "--device", "cpu", "--output", "models\\checkpoint"]
        prediction = ["predict", "data\\physical.npz", "data\\vectors.npz", "--model", "models\\checkpoint", "--device", "cpu", "--output", "outputs\\predictions.jsonl"]
        for arguments in (training, prediction):
            result = subprocess.run([sys.executable, "-B", "-c", bootstrap, *arguments], cwd=original, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
        before = read_rows(original / "outputs" / "predictions.jsonl")
        provenance = read_json(original / "models" / "checkpoint" / "provenance.json")
        self.assertTrue(all("\\" not in item["path"] for item in provenance["inputs"]))
        self.assertTrue(all(path.resolve().is_relative_to(self.root.resolve()) for path in (original, moved)))
        original.rename(moved)
        self.assertFalse(original.exists())
        retraining = training[:-1] + ["models\\retrained"]
        result = subprocess.run([sys.executable, "-B", "-c", bootstrap, *retraining], cwd=moved, env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        for filename in ("neural.json", "profile.json"):
            (moved / "settings" / filename).unlink()
        result = subprocess.run([sys.executable, "-B", "-c", bootstrap, *prediction], cwd=moved, env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        after = read_rows(moved / "outputs" / "predictions.jsonl")
        self.assertEqual(after, before)
        parameters = [torch.load(moved / "models" / name / "parameters.pt", map_location="cpu", weights_only=True) for name in ("checkpoint", "retrained")]
        for name in parameters[0]:
            torch.testing.assert_close(parameters[0][name], parameters[1][name], rtol=0, atol=0)
        relocated = moved / "cached inputs"
        relocated.mkdir()
        self.assertTrue(relocated.resolve().is_relative_to(self.root.resolve()))
        # Move caches and the complete model independently of their training paths.
        (moved / "data" / "physical.npz").rename(relocated / "traces.npz")
        (moved / "data" / "vectors.npz").rename(relocated / "content.npz")
        (moved / "models" / "checkpoint").rename(moved / "restored model")
        arguments = ["predict", "cached inputs\\traces.npz", "cached inputs/content.npz", "--model", "restored model", "--device", "cpu", "--output", "outputs/relocated.jsonl"]
        result = subprocess.run([sys.executable, "-B", "-c", bootstrap, *arguments], cwd=moved, env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_rows(moved / "outputs" / "relocated.jsonl"), before)

    def test_alignment_output_cannot_replace_encoder_files_or_source_images(self) -> None:
        import encode
        from PIL import Image
        model, manifest, image = self.root / "encoder", self.root / "manifest.jsonl", self.root / "image.png"
        model.mkdir()
        configuration = model / "config.json"
        configuration.write_text("{}", encoding="utf-8")
        Image.new("RGB", (32, 32)).save(image)
        write_rows(manifest, [{"id": "news", "image": image.as_posix(), "text": "Report."}])
        for output in (configuration, image):
            before = output.read_bytes()
            arguments = ["encode.py", manifest.as_posix(), "--model", model.as_posix(), "--output", output.as_posix()]
            with self.subTest(output=output.name), patch.object(sys, "argv", arguments), patch.dict(sys.modules, {"transformers": None}):
                with self.assertRaises(SystemExit):
                    encode.main()
            self.assertEqual(output.read_bytes(), before)

    def test_failed_metric_validation_preserves_existing_prediction_files(self) -> None:
        import deep
        physical, vectors, output = self.root / "physical.npz", self.root / "vectors.npz", self.root / "predictions.jsonl"
        for path in (physical, vectors):
            path.write_bytes(b"fixture")
        write_rows(output, [{"id": "existing"}])
        report = output.with_name("predictions.report.json")
        write_json(report, {"retained": True})
        before = output.read_bytes(), report.read_bytes()
        args = SimpleNamespace(command="predict", features=physical.as_posix(), embeddings=vectors.as_posix(), output=output.as_posix(), model=(self.root / "model").as_posix(), split="test", device="cpu", reference_domain=None, explain=False)
        with patch("deep.arguments", return_value=args), patch("deep.load_bundle", return_value=self.bundle), patch("engine.embeddings.load", return_value=self.archive), patch("engine.learning.predict", return_value=[{"id": "sample", "label": 0}]), patch("engine.metrics.evaluate", side_effect=ValueError("Invalid report inputs")):
            with self.assertRaisesRegex(SystemExit, "Invalid report inputs"):
                deep.main()
        self.assertEqual((output.read_bytes(), report.read_bytes()), before)
        with patch("deep.arguments", return_value=args), patch("deep.load_bundle", return_value=self.bundle), patch("engine.embeddings.load", return_value=self.archive), patch("engine.learning.predict", return_value=[{"id": "news", "label": 0}]), patch("engine.metrics.evaluate", return_value={"samples": 1}), patch("deep.write_json", side_effect=OSError("Report write failed")):
            with self.assertRaisesRegex(SystemExit, "Report write failed"):
                deep.main()
        self.assertEqual((output.read_bytes(), report.read_bytes()), before)
        with patch("deep.arguments", return_value=args), patch("deep.load_bundle", return_value=self.bundle), patch("engine.embeddings.load", return_value=self.archive), patch("engine.learning.predict", return_value=[{"id": "unlabeled", "fusion_risk": 0.2, "prediction": 1}]):
            with self.assertRaisesRegex(SystemExit, "fresh output path"):
                deep.main()
        self.assertEqual((output.read_bytes(), report.read_bytes()), before)
        args.output = (self.root / "unlabeled.jsonl").as_posix()
        with patch("deep.arguments", return_value=args), patch("deep.load_bundle", return_value=self.bundle), patch("engine.embeddings.load", return_value=self.archive), patch("engine.learning.predict", return_value=[{"id": "unlabeled", "label": None, "fusion_risk": 0.2, "prediction": 1}]), patch("builtins.print"):
            deep.main()
        self.assertEqual(read_rows(args.output)[0]["id"], "unlabeled")
        self.assertFalse((self.root / "unlabeled.report.json").exists())
        self.assertFalse(list(self.root.glob(".flawprint-*.tmp")))

    def test_prediction_exports_protect_images_referenced_by_the_cache(self) -> None:
        import deep
        from PIL import Image
        physical, vectors = self.root / "physical.npz", self.root / "vectors.npz"
        for path in (physical, vectors):
            path.write_bytes(b"fixture")
        for report in (False, True):
            with self.subTest(report=report):
                image = self.root / ("predictions.report.json" if report else "asset.json")
                Image.new("RGB", (64, 64), (20, 80, 100)).save(image, format="PNG")
                before = image.read_bytes()
                output = self.root / "predictions.jsonl" if report else image
                args = SimpleNamespace(command="predict", features=physical.as_posix(), embeddings=vectors.as_posix(), output=output.as_posix(), model=(self.root / "model").as_posix(), split="test", device="cpu", reference_domain=None, explain=False)
                with patch("deep.arguments", return_value=args), patch("deep.load_bundle", return_value=self.bundle), patch("engine.embeddings.load", return_value=self.archive), patch("engine.learning.predict", return_value=[{"id": "news", "image": image.as_posix()}]):
                    with self.assertRaisesRegex(SystemExit, "overwrite"):
                        deep.main()
                self.assertEqual(image.read_bytes(), before)

    def test_training_cli_input_changes_do_not_leave_a_published_model(self) -> None:
        import deep
        physical, vectors = self.root / "physical.npz", self.root / "vectors.npz"
        save_bundle(physical, self.bundle["values"], self.bundle["observed"], self.bundle["patches"], self.bundle["rows"], self.bundle["names"].tolist())
        embeddings.save(vectors.as_posix(), self.archive)
        config, profile = self.root / "config.json", self.root / "profile.json"
        write_json(config, asdict(replace(self.config, epochs=1)))
        write_json(profile, asdict(self.profile))
        output = self.root / "model"
        args = SimpleNamespace(command="train", features=physical.as_posix(), embeddings=vectors.as_posix(), config=config.as_posix(), profile_config=profile.as_posix(), output=output.as_posix(), seed=None, mode=None, dataset="weibo", device="cpu")
        original = torch.save
        def changed(parameters, path):
            original(parameters, path)
            with physical.open("ab") as stream:
                stream.write(b"changed")
        with patch("deep.arguments", return_value=args), patch("engine.learning.torch.save", side_effect=changed):
            with self.assertRaisesRegex(SystemExit, "not published"):
                deep.main()
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".flawprint-*")))

    def test_all_neural_controls_have_finite_predictions(self) -> None:
        for mode in ("unbiased", "independent", "deep", "concat", "uniform", "broadcast", "diagonal", "shuffled", "summary"):
            with self.subTest(mode=mode):
                directory = (self.root / mode).as_posix()
                fit(self.bundle, self.archive, directory, replace(self.config, mode=mode, epochs=1), self.profile, "cpu")
                rows = predict(self.bundle, self.archive, directory, device="cpu", explain=True)
                self.assertTrue(rows and all(np.isfinite(row["fusion_risk"]) for row in rows))
                if mode == "concat":
                    self.assertTrue(all(row["attention_contrast_rms"] is None and row["attention_shift_tv"] is None for row in rows))
                elif mode in ("unbiased", "independent", "deep"):
                    self.assertTrue(all(row["attention_contrast_rms"] == 0 and row["attention_shift_tv"] == 0 for row in rows))
                if mode == "uniform":
                    self.assertTrue(all(row["query_readout_delta"] == 0 for row in rows))
                else:
                    self.assertTrue(all(row["uniform_readout_risk"] is None for row in rows))
                if mode == "summary":
                    self.assertTrue(all(row["pair_readout_delta"] == 0 for row in rows))
                else:
                    self.assertTrue(all(row["summary_readout_risk"] is None for row in rows))
                self.assertTrue(all(row[name] is None for row in rows for name in ("pair_relation_fraction", "pair_contrast_rms", "pair_attention_shift_tv")))
                if mode == "deep":
                    self.assertFalse(read_json(Path(directory) / "model.json")["profiles"])

    def test_legacy_checkpoint_keeps_its_broadcast_operator(self) -> None:
        directory = self.root / "legacy"
        fit(self.bundle, self.archive, directory.as_posix(), replace(self.config, mode="broadcast", epochs=1), self.profile, "cpu")
        before = predict(self.bundle, self.archive, directory.as_posix(), device="cpu")
        metadata = read_json(directory / "model.json")
        metadata["version"], metadata["config"]["mode"] = "flawprint-neural-v1", "coupled"
        metadata.pop("relation_operator")
        write_json(directory / "model.json", metadata)
        after = predict(self.bundle, self.archive, directory.as_posix(), device="cpu")
        np.testing.assert_array_equal([row["fusion_risk"] for row in before], [row["fusion_risk"] for row in after])
        self.assertTrue(all(row["model_mode"] == "broadcast" for row in after))

    def test_deep_cli_protects_its_manifest_before_loading_models(self) -> None:
        path = self.root / "manifest.jsonl"
        write_rows(path, [{"id": "sample", "text": "A news report."}])
        original = path.read_bytes()
        result = subprocess.run([sys.executable, "-B", "deep.py", "extract", path.as_posix(), "--bert", "./missing", "--mae", "./missing", "--clip", "./missing", "--output", path.as_posix()], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("overwrite", result.stderr)
        self.assertEqual(path.read_bytes(), original)

    def test_local_transformer_extraction_preserves_vectors_and_missing_masks(self) -> None:
        try:
            from PIL import Image
            from transformers import BertConfig, BertModel, BertTokenizer, ViTMAEConfig, ViTMAEModel, ViTImageProcessor, CLIPConfig, CLIPModel, CLIPTokenizer, CLIPProcessor, CLIPImageProcessor
        except ImportError:
            self.skipTest("Optional Transformers image encoders are unavailable")
        import inspect
        directories = {name: self.root / name for name in ("bert", "mae", "clip")}
        for path in directories.values():
            path.mkdir()
        vocabulary = directories["bert"] / "vocab.txt"
        vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nnews\nreport\n", encoding="utf-8")
        if "vocab" in inspect.signature(BertTokenizer.__init__).parameters:
            tokenizer = BertTokenizer(vocab={token: index for index, token in enumerate(vocabulary.read_text().splitlines())})
        else:
            tokenizer = BertTokenizer(vocab_file=vocabulary.as_posix())
        tokenizer.save_pretrained(directories["bert"])
        BertModel(BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=8)).save_pretrained(directories["bert"])
        ViTMAEModel(ViTMAEConfig(image_size=32, patch_size=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)).save_pretrained(directories["mae"])
        ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(directories["mae"])
        (directories["clip"] / "vocab.json").write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1}), encoding="utf-8")
        (directories["clip"] / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        if "vocab" in inspect.signature(CLIPTokenizer.__init__).parameters:
            tokenizer = CLIPTokenizer(vocab={"<|startoftext|>": 0, "<|endoftext|>": 1}, merges=[])
        else:
            tokenizer = CLIPTokenizer(vocab_file=(directories["clip"] / "vocab.json").as_posix(), merges_file=(directories["clip"] / "merges.txt").as_posix(), bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>")
        processor = CLIPProcessor(tokenizer=tokenizer, image_processor=CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}))
        processor.save_pretrained(directories["clip"])
        config = CLIPConfig(text_config={"vocab_size": max(tokenizer.get_vocab().values()) + 1, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 16, "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}, vision_config={"image_size": 32, "patch_size": 16, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2}, projection_dim=8)
        CLIPModel(config).save_pretrained(directories["clip"])
        image = self.root / "image.png"
        Image.new("RGB", (32, 32), (90, 120, 150)).save(image)
        rows = [{"id": "image", "image": image.as_posix(), "text": "a news report " * 10}, {"id": "text", "image": "", "text": "news"}]
        paths = {name: path.as_posix() for name, path in directories.items()}
        first = embeddings.extract(rows, paths, batch_size=2, device="cpu", max_length=12)
        from transformers import AutoModel
        windows = {name: path.replace("/", "\\") for name, path in paths.items()}
        with patch("transformers.AutoModel.from_pretrained", wraps=AutoModel.from_pretrained) as loading:
            second = embeddings.extract(rows, windows, batch_size=2, device="cpu", max_length=12)
        self.assertTrue(all("\\" not in call.args[0] for call in loading.call_args_list))
        for name in embeddings.STREAMS:
            np.testing.assert_allclose(first["features"][name], second["features"][name], rtol=1e-6, atol=1e-7)
        self.assertEqual(first["features"]["bert"].shape, (2, 16))
        self.assertEqual(first["features"]["clip_image"].shape, (2, 8))
        self.assertFalse(first["observed"][1, 1])
        self.assertTrue(np.all(first["features"]["mae"][1] == 0))
        self.assertEqual(first["provenance"]["text_context"], {"bert": 8, "clip": 12})
        self.assertTrue(embeddings.protocol(first)["complete"])
        embeddings.save((self.root / "encoded.npz").as_posix(), first)
        self.assertEqual(embeddings.protocol(embeddings.load((self.root / "encoded.npz").as_posix())), embeddings.protocol(first))
        animated = self.root / "animated.gif"
        Image.new("RGB", (32, 32), (10, 20, 30)).save(animated, save_all=True, append_images=[Image.new("RGB", (32, 32), (100, 120, 140))], duration=100, loop=0)
        with self.assertRaisesRegex(ValueError, "Animated or multipage"):
            embeddings.extract([{"id": "animated", "image": animated.as_posix(), "text": "news"}], paths, batch_size=1, device="cpu", max_length=8)

    def test_row_constant_bias_is_invisible_to_masked_attention(self) -> None:
        torch.manual_seed(42)
        values = torch.randn(3, 2, 6, 6, dtype=torch.float64)
        observed = torch.tensor([[True, True, True, True, True, True], [True, False, True, False, True, False], [False, False, False, False, False, False]])
        offsets = torch.randn(3, 2, 6, 1, dtype=torch.float64)
        centered = contrast(values, observed)
        torch.testing.assert_close(centered, contrast(values + offsets, observed))
        self.assertTrue(torch.all(centered[2] == 0))
        self.assertTrue(torch.all(centered[1, :, ~observed[1]] == 0))
        torch.testing.assert_close(centered.sum(-1), torch.zeros_like(centered.sum(-1)), atol=1e-14, rtol=0)
        first = values[:2].masked_fill(~observed[:2, None, None, :], -torch.inf)
        second = (values + offsets)[:2].masked_fill(~observed[:2, None, None, :], -torch.inf)
        torch.testing.assert_close(first.softmax(-1), second.softmax(-1))
        with self.assertRaises(ValueError):
            contrast(values, observed.float())

    def test_query_contrast_equals_the_fixed_layer_log_odds_change(self) -> None:
        torch.manual_seed(52)
        values = torch.randn(1, 2, 7, 7, dtype=torch.float64)
        graph = torch.randn(1, 4, 4, dtype=torch.float64)
        graph = (graph + graph.transpose(-1, -2)) / 4
        observed = torch.ones(1, 7, dtype=torch.bool)
        observed[:, -1] = False
        graph[:, -1], graph[:, :, -1] = 0, 0
        bias = potential(values, observed, graph, "coupled")
        responsibility = values[:, :, :3, 3:].masked_fill(~observed[:, None, None, 3:], -torch.inf).softmax(-1)
        expected = torch.einsum("bhi,bij,bhj->bh", responsibility[:, :, 0], graph, responsibility[:, :, 1] - responsibility[:, :, 2])
        torch.testing.assert_close(bias[:, :, 0, 1] - bias[:, :, 0, 2], expected)
        base = values.masked_fill(~observed[:, None, None, :], -torch.inf).log_softmax(-1)
        shifted = (values + 0.7 * bias).masked_fill(~observed[:, None, None, :], -torch.inf).log_softmax(-1)
        measured = shifted[:, :, 0, 1] - shifted[:, :, 0, 2] - base[:, :, 0, 1] + base[:, :, 0, 2]
        torch.testing.assert_close(measured, 0.7 * expected)

    def test_summary_null_information_depends_on_responsibility_contrasts(self) -> None:
        graph = torch.tensor([[[0., 1., -1., 0.], [1., 0., 0., -1.], [-1., 0., 0., 1.], [0., -1., 1., 0.]]], dtype=torch.float64)
        observed = torch.ones(1, 7, dtype=torch.bool)
        delta = torch.tensor([[.1, -.1, 0., 0.], [0., .1, -.1, 0.], [.1, 0., -.1, 0.]], dtype=torch.float64)
        torch.testing.assert_close(summarize(graph, observed[:, 3:]), torch.zeros_like(graph))
        outputs = []
        for scale in (0., .5, 1.):
            values = torch.zeros(1, 1, 7, 7, dtype=torch.float64)
            responsibilities = .25 + scale * delta
            values[0, 0, :3, 3:] = responsibilities.log()
            outputs.append(potential(values, observed, graph, "coupled"))
            measured = outputs[-1][0, 0, 0, 1] - outputs[-1][0, 0, 0, 2]
            expected = (responsibilities[0] - .25) @ graph[0] @ (responsibilities[1] - responsibilities[2])
            torch.testing.assert_close(measured, expected)
            bound = torch.linalg.matrix_norm(graph[0], ord=2) * (responsibilities[0] - .25).norm() * (responsibilities[1] - responsibilities[2]).norm()
            self.assertLessEqual(measured.abs(), bound + 1e-14)
        torch.testing.assert_close(outputs[0][:, :, :3], torch.zeros_like(outputs[0][:, :, :3]), atol=1e-14, rtol=0)
        torch.testing.assert_close(outputs[1][:, :, :3, :3], outputs[2][:, :, :3, :3] / 4, atol=1e-14, rtol=0)
        torch.testing.assert_close(outputs[1][:, :, :3, 3:], outputs[2][:, :, :3, 3:] / 2, atol=1e-14, rtol=0)
        self.assertGreater(outputs[2][:, :, :3, :3].abs().max(), 0)

    def test_value_transport_matches_finite_attention_updates_and_value_offsets(self) -> None:
        torch.manual_seed(3407)
        observed = torch.tensor([[True, False, True, True], [False, False, False, False]])
        current, reference = [torch.softmax(torch.randn(2, 2, 4, 4, dtype=torch.float64).masked_fill(~observed[:, None, None, :], -1e20), -1) for _ in range(2)]
        values = torch.randn(2, 2, 4, 3, dtype=torch.float64)
        actual = transport(current, reference, values, observed)
        clean = values.masked_fill(~observed[:, None, :, None], 0)
        expected = ((current - reference) @ clean).masked_fill(~observed[:, None, :, None], 0)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(transport(current, reference, values + 19, observed), actual, atol=1e-14, rtol=1e-13)
        values[~observed[:, None, :, None].expand_as(values)] = float("nan")
        torch.testing.assert_close(transport(current, reference, values, observed), actual, atol=1e-14, rtol=1e-13)
        constant = torch.full_like(values, 17)
        torch.testing.assert_close(transport(current, reference, constant, observed), torch.zeros_like(actual), rtol=0, atol=0)

    def test_valid_copula_can_change_attention_without_transmitting_values(self) -> None:
        graph = torch.tensor([[0., 1., -1., 0.], [1., 0., 0., -1.], [-1., 0., 0., 1.], [0., -1., 1., 0.]], dtype=torch.float64)
        precision = torch.eye(4, dtype=torch.float64) - .2 * graph
        inverse = torch.linalg.inv(precision)
        scale = inverse.diag().mean()
        correlation = inverse / scale
        torch.testing.assert_close(correlation.diag(), torch.ones(4, dtype=torch.float64))
        self.assertGreater(float(torch.linalg.eigvalsh(correlation).min()), 0)
        relation = (torch.eye(4, dtype=torch.float64) - torch.linalg.inv(correlation))[None]
        relation = relation / (1 + torch.linalg.matrix_norm(relation, keepdim=True))
        observed = torch.ones(1, 5, dtype=torch.bool)
        summary = summarize(relation, observed[:, 1:])
        hidden = relation - summary
        torch.testing.assert_close(hidden.sum(-1), torch.zeros(1, 4, dtype=torch.float64), atol=1e-14, rtol=0)
        self.assertGreater(float(hidden.norm()), 0)
        logits = torch.zeros(1, 1, 5, 5, dtype=torch.float64)
        logits[0, 0, 0, 1:] = torch.tensor([.35, .25, .25, .15], dtype=torch.float64).log()
        base = logits + potential(logits, observed, summary, "coupled")
        direction = potential(logits, observed, relation, "coupled") - potential(logits, observed, summary, "coupled")
        reference = torch.softmax(base, -1)
        values = torch.zeros(1, 1, 5, 1, dtype=torch.float64)
        values[0, 0, 0, 0] = 1
        values[0, 0, 1, 0] = -reference[0, 0, 0, 0] / reference[0, 0, 0, 1]
        for strength in (.1, .5, 1., 2.):
            current = torch.softmax(base + strength * direction, -1)
            self.assertGreater(float((current - reference)[0, 0, 0].abs().sum()), 0)
            torch.testing.assert_close(transport(current, reference, values, observed)[0, 0, 0], torch.zeros(1, dtype=torch.float64), atol=1e-13, rtol=0)
            changed = values.clone()
            changed[0, 0, 2, 0] = 1
            self.assertGreater(float(transport(current, reference, changed, observed)[0, 0, 0].abs()), 1e-6)

    def test_zero_first_order_transport_does_not_imply_zero_finite_response(self) -> None:
        observed = torch.ones(1, 3, dtype=torch.bool)
        reference = torch.full((1, 1, 3, 3), 1 / 3, dtype=torch.float64)
        direction = torch.tensor([-1., 0., 1.], dtype=torch.float64).expand_as(reference)
        values = torch.tensor([1., -2., 1.], dtype=torch.float64).reshape(1, 1, 3, 1)
        strength = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        output = transport(torch.softmax(strength * direction, -1), reference, values, observed)[0, 0, 0, 0]
        derivative, = torch.autograd.grad(output, strength)
        torch.testing.assert_close(derivative, torch.tensor(0., dtype=torch.float64), atol=1e-14, rtol=0)
        current = torch.softmax(.5 * direction, -1)
        expected = (2 * math.cosh(.5) - 2) / (2 * math.cosh(.5) + 1)
        torch.testing.assert_close(transport(current, reference, values, observed)[0, 0, 0, 0], torch.tensor(expected, dtype=torch.float64))

    def test_value_transport_obeys_the_value_diameter_bound(self) -> None:
        torch.manual_seed(52)
        observed = torch.ones(1, 5, dtype=torch.bool)
        values = torch.randn(1, 2, 5, 4, dtype=torch.float64)
        reference, current = [torch.softmax(torch.randn(1, 2, 5, 5, dtype=torch.float64), -1) for _ in range(2)]
        diameter = torch.cdist(values[0], values[0]).amax((-1, -2))
        bound = diameter[None, :, None] * (current - reference).abs().sum(-1) / 2
        self.assertTrue(torch.all(transport(current, reference, values, observed).norm(dim=-1) <= bound + 1e-13))

    def test_post_projection_transport_detects_a_value_null_update(self) -> None:
        torch.manual_seed(3047)
        block = ResidualBlock(self.config).double().eval()
        tokens = torch.randn(1, 7, self.config.dimension, dtype=torch.float64)
        observed = torch.ones(1, 7, dtype=torch.bool)
        relation = torch.tensor([[[0., 1., -1., 0.], [1., 0., 0., -1.], [-1., 0., 0., 1.], [0., -1., 1., 0.]]], dtype=torch.float64) / 5
        measurements = []
        expected = block(tokens, observed, relation, "coupled", measurements)
        self.assertGreater(float(measurements[0]["pair_attention_shift_tv"].detach()), 0)
        self.assertGreater(float(measurements[0]["pair_value_transport_rms"].detach()), 0)
        torch.testing.assert_close(block(tokens, observed, relation, "coupled"), expected, rtol=0, atol=0)
        with torch.no_grad():
            block.output.weight.zero_()
        measurements = []
        block(tokens, observed, relation, "coupled", measurements)
        self.assertGreater(float(measurements[0]["pair_attention_shift_tv"].detach()), 0)
        torch.testing.assert_close(measurements[0]["pair_value_transport_rms"], torch.zeros(1, dtype=torch.float64), rtol=0, atol=0)

    def test_continuous_logit_transport_has_a_finite_remainder_bound(self) -> None:
        torch.manual_seed(3047)
        observed = torch.tensor([[True, True, False, True, True], [False, False, False, False, False]])
        base = torch.randn(2, 2, 5, 5, dtype=torch.float64)
        reference = torch.softmax(base.masked_fill(~observed[:, None, None, :], -1e20), -1)
        direction = torch.randn_like(base)
        values = torch.randn(2, 2, 5, 3, dtype=torch.float64)
        tangent, bound = sensitivity(reference, direction, values, observed)
        for strength in (-2., -.2, 0., .3, 1., 3.):
            current = torch.softmax((base + strength * direction).masked_fill(~observed[:, None, None, :], -1e20), -1)
            actual = transport(current, reference, values, observed)
            error = torch.linalg.vector_norm(actual - strength * tangent, dim=-1)
            self.assertTrue(torch.all(error <= strength ** 2 * bound + 1e-13))
        shifted, shifted_bound = sensitivity(reference, direction + 17, values + 23, observed)
        torch.testing.assert_close(shifted, tangent, atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(shifted_bound, bound, atol=1e-13, rtol=1e-12)
        step = 1e-5
        plus = torch.softmax((base + step * direction).masked_fill(~observed[:, None, None, :], -1e20), -1)
        minus = torch.softmax((base - step * direction).masked_fill(~observed[:, None, None, :], -1e20), -1)
        torch.testing.assert_close(transport(plus, minus, values, observed) / (2 * step), tangent, atol=1e-9, rtol=1e-8)

    def test_distinct_continuous_logits_can_have_zero_tangent_and_nonzero_response(self) -> None:
        observed = torch.ones(1, 4, dtype=torch.bool)
        reference = torch.full((1, 1, 4, 4), .25, dtype=torch.float64)
        delta = torch.tensor([-1.5, -.2, .6, 1.3], dtype=torch.float64)
        centered = delta - delta.mean()
        values = centered.square() - centered.square().mean()
        values = values - (values @ centered) / (centered @ centered) * centered
        values = values.reshape(1, 1, 4, 1)
        direction = delta.expand_as(reference)
        tangent, bound = sensitivity(reference, direction, values, observed)
        self.assertEqual(len(delta.unique()), 4)
        torch.testing.assert_close(tangent, torch.zeros_like(tangent), atol=1e-14, rtol=0)
        current = torch.softmax(.3 * direction, -1)
        actual = transport(current, reference, values, observed)
        self.assertGreater(float(actual.abs().max()), .001)
        self.assertTrue(torch.all(actual.abs().squeeze(-1) <= .3 ** 2 * bound + 1e-14))
        for scale in (1e-6, 1e-3):
            changed = values + scale * centered.reshape(1, 1, 4, 1)
            slope, _ = sensitivity(reference, direction, changed, observed)
            expected = scale * centered.square().mean()
            torch.testing.assert_close(slope, torch.full_like(slope, expected), atol=1e-14, rtol=1e-10)

    def test_transport_sensitivity_ignores_missing_nodes_and_rejects_bad_inputs(self) -> None:
        observed = torch.tensor([[True, False, False], [False, False, False]])
        reference = torch.zeros(2, 1, 3, 3, dtype=torch.float64)
        reference[0, 0, 0, 0] = 1
        values = torch.full((2, 1, 3, 2), float("nan"), dtype=torch.float64)
        values[0, 0, 0] = 3
        direction = torch.full_like(reference, float("inf"))
        direction[0, 0, 0, 0] = 2
        tangent, bound = sensitivity(reference, direction, values, observed)
        torch.testing.assert_close(tangent, torch.zeros_like(tangent), atol=0, rtol=0)
        torch.testing.assert_close(bound, torch.zeros_like(bound), atol=0, rtol=0)
        for damaged in (direction.float(), direction[:, :, :2], None):
            with self.assertRaises(ValueError):
                sensitivity(reference, damaged, values, observed)
        direction[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            sensitivity(reference, direction, values, observed)

    def test_projected_pair_tangent_and_bound_match_the_same_layer_path(self) -> None:
        torch.manual_seed(52)
        block = ResidualBlock(self.config).double().eval()
        tokens = torch.randn(2, 7, self.config.dimension, dtype=torch.float64)
        observed = torch.ones(2, 7, dtype=torch.bool)
        observed[1, 5:] = False
        relation = torch.randn(2, 4, 4, dtype=torch.float64)
        relation = (relation + relation.transpose(-1, -2)) / 10
        measurements = []
        expected = block(tokens, observed, relation, "coupled", measurements)
        qkv = block.qkv(block.norm(tokens.masked_fill(~observed[:, :, None], 0))).reshape(2, 7, 3, block.heads, block.width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        base = query @ key.transpose(-1, -2) / math.sqrt(block.width)
        relation = relation.masked_fill(~(observed[:, 3:, None] & observed[:, None, 3:]), 0)
        strength = block.limit * torch.sigmoid(block.strength)
        full = strength * potential(base, observed, relation, "coupled")
        reduced = strength * potential(base, observed, summarize(relation, observed[:, 3:]), "coupled")
        reference = torch.softmax((base + reduced).masked_fill(~observed[:, None, None, :], -torch.inf), -1)
        current = torch.softmax((base + full).masked_fill(~observed[:, None, None, :], -torch.inf), -1)
        tangent, _ = sensitivity(reference, full - reduced, value, observed)
        error = transport(current, reference, value, observed) - tangent
        error = torch.nn.functional.linear(error.transpose(1, 2).reshape(2, 7, self.config.dimension), block.output.weight)
        measured_error = (error[:, :3].square().sum((1, 2)) / (3 * self.config.dimension)).sqrt()
        self.assertTrue(torch.all(measured_error <= measurements[0]["pair_transport_remainder_bound"] + 1e-13))
        torch.testing.assert_close(block(tokens, observed, relation, "coupled"), expected, atol=0, rtol=0)
        with torch.no_grad():
            block.output.weight.zero_()
        measurements = []
        block(tokens, observed, relation, "coupled", measurements)
        for name in ("pair_tangent_transport_rms", "pair_transport_remainder_bound"):
            torch.testing.assert_close(measurements[0][name], torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)

    def test_reference_mixture_changes_pair_direction_with_the_content_state_fixed(self) -> None:
        observed = torch.ones(1, 3, dtype=torch.bool)
        logits = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
        logits[0, 0, 0, 1:] = torch.tensor([.8, .2], dtype=torch.float64).log()
        relations = []
        for proportion in (0., .5, 1.):
            coefficient = (1 - 2 * proportion) * .8
            correlation = torch.tensor([[1., coefficient], [coefficient, 1.]], dtype=torch.float64)
            raw = torch.eye(2, dtype=torch.float64) - torch.linalg.inv(correlation)
            relations.append((raw / (1 + raw.norm()))[None])
        biases = [potential(logits, observed, graph, "coupled") for graph in relations]
        endpoint_contrasts = [bias[0, 0, 0, 1] - bias[0, 0, 0, 2] for bias in biases]
        self.assertLess(float(endpoint_contrasts[0] * endpoint_contrasts[2]), 0)
        torch.testing.assert_close(relations[1], torch.zeros_like(relations[1]), atol=0, rtol=0)
        values = torch.tensor([0., 1., -1.], dtype=torch.float64).reshape(1, 1, 3, 1)
        reference = torch.softmax(logits + biases[0], -1)
        current = torch.softmax(logits + biases[2], -1)
        tangent, bound = sensitivity(reference, biases[2] - biases[0], values, observed)
        actual = transport(current, reference, values, observed)
        self.assertTrue(torch.all((actual - tangent).abs().squeeze(-1) <= bound + 1e-13))

    def test_shrunken_reference_changes_obey_the_relation_continuity_bound(self) -> None:
        torch.manual_seed(3407)
        for _ in range(12):
            matrices = []
            for _ in range(2):
                raw = torch.randn(4, 4, dtype=torch.float64)
                covariance = raw @ raw.T
                diagonal = covariance.diag().sqrt()
                matrices.append(.9 * covariance / diagonal[:, None] / diagonal[None] + .1 * torch.eye(4, dtype=torch.float64))
            first, second = matrices
            precision, changed = torch.linalg.inv(first), torch.linalg.inv(second)
            difference = torch.linalg.matrix_norm(first - second)
            self.assertLessEqual(float((precision - changed).norm()), 100 * float(difference) + 1e-12)
            ranks = torch.randn(4, dtype=torch.float64)
            relations = [torch.diag(ranks) @ (torch.eye(4, dtype=torch.float64) - inverse) @ torch.diag(ranks) for inverse in (precision, changed)]
            normalized = [graph / (1 + graph.norm()) for graph in relations]
            distance = torch.linalg.matrix_norm(normalized[0] - normalized[1])
            self.assertLessEqual(float(distance), float((relations[0] - relations[1]).norm()) + 1e-12)
            self.assertLessEqual(float(distance), float(ranks.abs().max().square() * 100 * difference) + 1e-12)

    def test_value_transport_rejects_invalid_observed_probabilities(self) -> None:
        weights = torch.full((1, 1, 3, 3), 1 / 3, dtype=torch.float64)
        values = torch.ones(1, 1, 3, 2, dtype=torch.float64)
        observed = torch.ones(1, 3, dtype=torch.bool)
        for damaged in (weights * 2, weights * -1, weights * float("nan")):
            with self.assertRaises(ValueError):
                transport(damaged, weights, values, observed)
        with self.assertRaises(ValueError):
            transport(weights, weights, values.float(), observed)

    def test_pair_diagnostics_separate_available_information_from_local_use(self) -> None:
        torch.manual_seed(42)
        block = ResidualBlock(self.config).double().eval()
        tokens = torch.randn(2, 7, self.config.dimension, dtype=torch.float64)
        observed = torch.ones(2, 7, dtype=torch.bool)
        observed[1, 3:] = False
        graph = torch.tensor([[[0., 1., -1., 0.], [1., 0., 0., -1.], [-1., 0., 0., 1.], [0., -1., 1., 0.]]], dtype=torch.float64).repeat(2, 1, 1) / 5
        graph[1] = 0
        original = copy.deepcopy(block.qkv.state_dict())
        with torch.no_grad():
            block.qkv.weight.zero_()
            block.qkv.bias.zero_()
        measurements = []
        block(tokens, observed, graph, "coupled", measurements)
        uniform = measurements[0]
        torch.testing.assert_close(uniform["pair_relation_fraction"], torch.tensor([1., 0.], dtype=torch.float64))
        torch.testing.assert_close(uniform["pair_contrast_rms"], torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
        torch.testing.assert_close(uniform["pair_attention_shift_tv"], torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
        block.qkv.load_state_dict(original)
        measurements = []
        expected = block(tokens, observed, graph, "coupled", measurements)
        selected = measurements[0]
        torch.testing.assert_close(selected["pair_contrast_rms"], selected["attention_contrast_rms"])
        torch.testing.assert_close(selected["pair_attention_shift_tv"], selected["attention_shift_tv"])
        self.assertGreater(selected["pair_contrast_rms"][0], 0)
        self.assertGreater(selected["pair_attention_shift_tv"][0], 0)
        self.assertEqual(selected["pair_contrast_rms"][1], 0)
        changed_tokens = tokens.masked_fill(~observed[:, :, None], float("nan"))
        mask = observed[:, 3:]
        changed_graph = graph.masked_fill(~(mask[:, :, None] & mask[:, None, :]), float("nan"))
        measured = []
        actual = block(changed_tokens, observed, changed_graph, "coupled", measured)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name in selected:
            torch.testing.assert_close(measured[0][name], selected[name], rtol=0, atol=0)

    def test_uniform_readout_ignores_queries_and_respects_missing_anchors(self) -> None:
        torch.manual_seed(3047)
        graph = torch.randn(2, 4, 4)
        observed = torch.tensor([[True, False, True, True, False, True, False], [True, True, True, False, False, False, False]])
        first = potential(torch.randn(2, 2, 7, 7), observed, graph, "uniform")
        second = potential(torch.randn(2, 2, 7, 7) * 20, observed, graph, "uniform")
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.all(first[1] == 0))
        self.assertTrue(torch.all(first[0, :, 1] == 0))
        self.assertTrue(torch.all(first[0, :, :, 4] == 0))
        damaged = graph.clone()
        damaged[0, 1], damaged[0, :, 1] = 1000, 1000
        damaged[0, 3], damaged[0, :, 3] = -1000, -1000
        torch.testing.assert_close(first, potential(torch.randn(2, 2, 7, 7), observed, damaged, "uniform"))

    def test_uniform_and_coupled_agree_when_content_has_no_anchor_preference(self) -> None:
        graph = torch.tensor([[[0.2, -0.1, 0.3], [-0.1, 0.0, 0.2], [0.3, 0.2, -0.2]]])
        observed = torch.ones(1, 6, dtype=torch.bool)
        values = torch.zeros(1, 2, 6, 6)
        torch.testing.assert_close(potential(values, observed, graph, "coupled"), potential(values, observed, graph, "uniform"))
        observed[:, 4:] = False
        values = torch.randn(1, 2, 6, 6)
        torch.testing.assert_close(potential(values, observed, graph, "coupled"), potential(values, observed, graph, "uniform"))

    def test_nonzero_relation_can_have_zero_effective_attention_contrast(self) -> None:
        torch.manual_seed(3407)
        block = ResidualBlock(self.config).eval()
        tokens = torch.randn(2, 7, self.config.dimension)
        observed = torch.ones(2, 7, dtype=torch.bool)
        graph = torch.full((2, 4, 4), 0.1)
        diagnostics = []
        coupled = block(tokens, observed, graph, "coupled", diagnostics)
        plain = block(tokens, observed, graph, "unbiased")
        self.assertGreater(float(graph.norm()), 0)
        torch.testing.assert_close(coupled, plain, atol=1e-6, rtol=1e-6)
        self.assertLess(float(diagnostics[0]["attention_contrast_rms"].max().detach()), 1e-6)
        self.assertLess(float(diagnostics[0]["attention_shift_tv"].max().detach()), 1e-6)

    def test_opt_in_attention_diagnostics_leave_the_prediction_unchanged(self) -> None:
        torch.manual_seed(42)
        model = ResidualAttention({name: 4 for name in embeddings.STREAMS}, 3, self.config).eval()
        features = {name: torch.randn(2, 4) for name in embeddings.STREAMS}
        observed = torch.ones(2, 4, dtype=torch.bool)
        ranks, graph = torch.randn(2, 3), torch.randn(2, 3, 3)
        graph = graph + graph.transpose(-1, -2)
        mask = torch.tensor([[True, True, True], [False, False, False]])
        ordinary = model(features, observed, ranks, graph, mask)
        measured = model(features, observed, ranks, graph, mask, explain=True)
        torch.testing.assert_close(ordinary["logit"], measured["logit"], rtol=0, atol=0)
        self.assertTrue(torch.all(measured["attention_contrast_rms"] >= 0))
        self.assertTrue(torch.all((measured["attention_shift_tv"] >= 0) & (measured["attention_shift_tv"] <= 1)))
        self.assertEqual(float(measured["attention_contrast_rms"][1].detach()), 0)
        self.assertEqual(float(measured["attention_shift_tv"][1].detach()), 0)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
