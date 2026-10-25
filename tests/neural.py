# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from engine import embeddings
from engine.copula import score
from engine.experiment import indices
from engine.profile import DomainProfile, ProfileConfig
from engine.storage import subset, write_rows, read_json, save_bundle, write_json
from engine.synthetic import fixture
try:
    import torch
    from engine.neural import NeuralConfig, ResidualAttention
    from engine.learning import prepare, relation, fit, predict
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

    def test_neural_fit_reload_and_encoder_provenance(self) -> None:
        directory = (self.root / "model").as_posix()
        report = fit(self.bundle, self.archive, directory, self.config, self.profile, "cpu")
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertEqual(report["image_supervised_samples"], 0)
        predictions = predict(self.bundle, self.archive, directory, device="cpu")
        self.assertEqual(len(predictions), len(indices(self.bundle, "test")))
        self.assertTrue(all(np.isfinite(row["fusion_risk"]) for row in predictions))
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
        prediction = subprocess.run([sys.executable, "-B", "deep.py", "predict", physical.as_posix(), vectors.as_posix(), "--model", model.as_posix(), "--device", "cpu", "--output", output.as_posix()], capture_output=True, text=True, env=environment, timeout=60)
        self.assertEqual(prediction.returncode, 0, prediction.stderr)
        self.assertEqual(read_json(output.with_name("predictions.report.json"))["samples"], len(indices(self.bundle, "test")))
        provenance = read_json(model / "provenance.json")
        self.assertEqual(provenance["dataset"], "weibo21")
        self.assertFalse(provenance["inputs_changed_during_training"])

    def test_all_neural_controls_have_finite_predictions(self) -> None:
        for mode in ("unbiased", "independent", "deep", "concat"):
            with self.subTest(mode=mode):
                directory = (self.root / mode).as_posix()
                fit(self.bundle, self.archive, directory, replace(self.config, mode=mode, epochs=1), self.profile, "cpu")
                rows = predict(self.bundle, self.archive, directory, device="cpu")
                self.assertTrue(rows and all(np.isfinite(row["fusion_risk"]) for row in rows))
                if mode == "deep":
                    self.assertFalse(read_json(Path(directory) / "model.json")["profiles"])

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
        import json
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
        BertModel(BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)).save_pretrained(directories["bert"])
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
        rows = [{"id": "image", "image": image.as_posix(), "text": "a news report"}, {"id": "text", "image": "", "text": "news"}]
        paths = {name: path.as_posix() for name, path in directories.items()}
        first = embeddings.extract(rows, paths, batch_size=2, device="cpu", max_length=12)
        second = embeddings.extract(rows, paths, batch_size=2, device="cpu", max_length=12)
        for name in embeddings.STREAMS:
            np.testing.assert_allclose(first["features"][name], second["features"][name], rtol=1e-6, atol=1e-7)
        self.assertEqual(first["features"]["bert"].shape, (2, 16))
        self.assertEqual(first["features"]["clip_image"].shape, (2, 8))
        self.assertFalse(first["observed"][1, 1])
        self.assertTrue(np.all(first["features"]["mae"][1] == 0))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
