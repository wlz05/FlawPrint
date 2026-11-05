# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from engine.storage import relative_path, save_bundle, load_bundle, subset
from engine.data import normalize, partition, audit_splits, join_predictions, text_fingerprint
from engine.traces import NAMES, TraceConfig, extract_one
from engine.storage import read_json, write_json
from engine.synthetic import fixture
from engine.profile import DomainProfile, ProfileConfig
from engine.conformal import Platt, calibrate_rows, apply_calibration, fuse
from engine.experiment import benchmark, indices
from engine.stream import pool, adapt
from engine.provenance import conventions
from engine.metrics import binary, group_bootstrap

class WorkflowTests(unittest.TestCase):
    def test_seed_summary_uses_four_equally_weighted_results(self) -> None:
        from deep import SEEDS, seed_summary
        records = [{"seed": seed, "metrics": {"macro_f1": value, "auc": None}, "protocol": {"test": "frozen"}} for seed, value in zip(SEEDS, (0.2, 0.4, 0.6, 0.8))]
        result = seed_summary(records)
        self.assertAlmostEqual(result["metrics"]["macro_f1"]["mean"], 0.5)
        self.assertAlmostEqual(result["metrics"]["macro_f1"]["sample_std"], 0.2581988897471611)
        self.assertIsNone(result["metrics"]["auc"]["mean"])
        for values in (records[:-1], records + [records[0]], [{**row, "protocol": {}} if index == 1 else row for index, row in enumerate(records)]):
            with self.assertRaises(ValueError):
                seed_summary(values)

    def setUp(self) -> None:
        self.bundle = fixture(per_split=16)
        self.config = ProfileConfig(bootstrap=5, min_samples=12)
        Path("./tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir="./tmp")
        self.root = Path(self.temporary.name).relative_to(Path.cwd())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def profile(self, domain: str = "target_gamma") -> DomainProfile:
        split = "profile" if domain.startswith("target") else "train"
        return DomainProfile.fit(subset(self.bundle, indices(self.bundle, split, domain)), domain, self.config)

    def benchmark_args(self, name="seedgrid", seeds=None):
        features, config = self.root / "features.npz", self.root / "benchmark.json"
        save_bundle(features, self.bundle["values"], self.bundle["observed"], self.bundle["patches"], self.bundle["rows"], self.bundle["names"])
        write_json(config, asdict(replace(self.config, bootstrap=2)))
        return SimpleNamespace(features=features.as_posix(), config=config.as_posix(), output=(self.root / name).as_posix(), seeds=[42, 52, 3047, 3407] if seeds is None else seeds, dataset="weibo", alpha=0.1, bootstrap=2, ablations=False)

    def benchmark_stub(self, bundle, output, config, alpha, bootstraps, ablations):
        score = {42: 0.2, 52: 0.4, 3047: 0.6, 3407: 0.8}[config.seed]
        result = {"methods": {"control": {"accuracy": score, "auc": None, "bootstrap": {"lower": score - 0.1, "upper": score + 0.1}}}, "skipped": {}, "config": asdict(config)}
        write_json(Path(output) / "report.json", result)
        write_json(Path(output) / "provenance.json", {"configuration": asdict(config)})
        return result

    def test_benchmark_four_seed_outputs_average_individual_metrics(self):
        from main import command_benchmark
        from engine.results import SEEDS, seed_summary
        from deep import seed_summary as neural_summary
        self.assertIs(seed_summary, neural_summary)
        args = self.benchmark_args()
        with patch("main.benchmark", side_effect=self.benchmark_stub) as runner:
            result = command_benchmark(args)
        self.assertEqual([call.args[2].seed for call in runner.call_args_list], list(SEEDS))
        summary = read_json(Path(args.output) / "summary.json")
        self.assertEqual(result["seed_count"], 4)
        self.assertEqual(summary["seeds"], list(SEEDS))
        self.assertAlmostEqual(summary["metrics"]["control.accuracy"]["mean"], 0.5)
        self.assertIsNone(summary["metrics"]["control.auc"]["mean"])
        self.assertFalse(any("bootstrap" in key for key in summary["metrics"]))
        self.assertNotIn("seed", summary["protocol"]["profile"])
        for seed in SEEDS:
            directory = Path(args.output) / ("seed" + str(seed))
            report = read_json(directory / "report.json")
            self.assertEqual(report["config"]["seed"], seed)
            self.assertEqual(summary["byseed"][str(seed)]["control"]["accuracy"], report["methods"]["control"]["accuracy"])
            self.assertTrue((directory / "result.json").is_file())
            self.assertIn("sha256", read_json(directory / "provenance.json")["input"])

    def test_benchmark_single_seed_is_declared_without_a_four_seed_claim(self):
        from main import command_benchmark
        args = self.benchmark_args(seeds=[52])
        with patch("main.benchmark", side_effect=self.benchmark_stub):
            command_benchmark(args)
        summary = read_json(Path(args.output) / "summary.json")
        self.assertEqual((summary["seeds"], summary["seed_count"]), ([52], 1))
        self.assertAlmostEqual(summary["metrics"]["control.accuracy"]["mean"], 0.4)
        self.assertIsNone(summary["metrics"]["control.accuracy"]["sample_std"])

    def test_benchmark_rejects_invalid_seed_grids_before_execution(self):
        from main import command_benchmark
        for seeds in ([], [42, 42], [17], [True]):
            with self.subTest(seeds=seeds), patch("main.benchmark") as runner:
                args = self.benchmark_args(seeds=seeds)
                with self.assertRaisesRegex(ValueError, "distinct seeds"):
                    command_benchmark(args)
                runner.assert_not_called()
                self.assertFalse(Path(args.output).exists())

    def test_benchmark_interruption_keeps_completed_seeds_without_a_mean(self):
        from main import command_benchmark
        args = self.benchmark_args()
        def interrupted(bundle, output, config, *rest):
            if config.seed == 3047:
                raise OSError("Interrupted fixture")
            return self.benchmark_stub(bundle, output, config, *rest)
        with patch("main.benchmark", side_effect=interrupted), self.assertRaises(OSError):
            command_benchmark(args)
        for seed in (42, 52):
            self.assertTrue((Path(args.output) / ("seed" + str(seed)) / "result.json").is_file())
        self.assertFalse((Path(args.output) / "summary.json").exists())

    def test_benchmark_stops_when_features_or_configuration_change(self):
        from main import command_benchmark
        for field in ("features", "config"):
            args = self.benchmark_args(name=field)
            def changed(*values):
                result = self.benchmark_stub(*values)
                path = Path(getattr(args, field))
                path.write_bytes(path.read_bytes() + b" ")
                return result
            with self.subTest(field=field), patch("main.benchmark", side_effect=changed), self.assertRaisesRegex(ValueError, "input changed"):
                command_benchmark(args)
            self.assertFalse((Path(args.output) / "summary.json").exists())

    def test_benchmark_stops_when_source_changes(self):
        from main import command_benchmark
        args = self.benchmark_args()
        state = {"source": "before"}
        def changed(*values):
            result = self.benchmark_stub(*values)
            state["source"] = "after"
            return result
        with patch("main.source_digest", side_effect=lambda: state["source"]), patch("main.benchmark", side_effect=changed), self.assertRaisesRegex(ValueError, "source changed"):
            command_benchmark(args)
        self.assertFalse((Path(args.output) / "summary.json").exists())

    def test_benchmark_rejects_inconsistent_method_sets(self):
        from main import command_benchmark
        args = self.benchmark_args()
        def changed(bundle, output, config, *rest):
            result = self.benchmark_stub(bundle, output, config, *rest)
            if config.seed == 52:
                result["methods"]["extra"] = {"accuracy": 0.5}
            return result
        with patch("main.benchmark", side_effect=changed), self.assertRaisesRegex(ValueError, "same protocol and methods"):
            command_benchmark(args)
        self.assertFalse((Path(args.output) / "summary.json").exists())

    def test_benchmark_preserves_existing_outputs_and_protects_inputs(self):
        from main import command_benchmark
        args = self.benchmark_args()
        root = Path(args.output)
        root.mkdir()
        marker = root / "retained.txt"
        marker.write_text("retained", encoding="utf-8")
        with patch("main.benchmark") as runner, self.assertRaisesRegex(ValueError, "fresh or empty"):
            command_benchmark(args)
        runner.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "retained")
        args.output = self.root.as_posix()
        with self.assertRaisesRegex(ValueError, "cannot contain input"):
            command_benchmark(args)

    def test_benchmark_real_fixture_keeps_four_independent_reports(self):
        from main import command_benchmark
        args = self.benchmark_args()
        with patch("engine.experiment.METHODS", []):
            command_benchmark(args)
        summary = read_json(Path(args.output) / "summary.json")
        scores = [summary["byseed"][str(seed)]["physical_full"]["accuracy"] for seed in args.seeds]
        self.assertAlmostEqual(summary["metrics"]["physical_full.accuracy"]["mean"], sum(scores) / 4)
        for seed in args.seeds:
            directory = Path(args.output) / ("seed" + str(seed))
            self.assertTrue((directory / "predictions" / "physical_full.jsonl").is_file())
            self.assertTrue((directory / "calibration.json").is_file())

    def test_absolute_paths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            relative_path(Path.cwd())

    def test_relative_paths_normalize_windows_separators_on_posix(self) -> None:
        with patch("engine.storage.Path", PurePosixPath):
            self.assertEqual(relative_path("data\\cache\\features.npz").as_posix(), "data/cache/features.npz")
            self.assertEqual(relative_path(PureWindowsPath("data\\cache\\features.npz")).as_posix(), "data/cache/features.npz")
            self.assertEqual(relative_path("models\\checkpoint\\profiles\\domain.json").as_posix(), "models/checkpoint/profiles/domain.json")

    def test_relative_paths_reject_implicit_roots_and_remote_locations(self) -> None:
        for value in ("", "  ", "\0", "~", "~/models", "~\\models", "https://example.test/model", "file://host/model", chr(90) + ":model", "\\\\host\\model"):
            with self.subTest(path=value):
                with self.assertRaises(ValueError):
                    relative_path(value)

    def test_finefake_labels_are_explicitly_inverted(self) -> None:
        rows = normalize([{"id": "a", "image": "a.jpg", "label": 0}, {"id": "b", "image": "b.jpg", "label": 1}], label_convention="real-one")
        self.assertEqual([row["label"] for row in rows], [1, 0])

    def test_profile_labels_are_removed(self) -> None:
        rows = normalize([{"id": "a", "image": "a.jpg", "label": 1, "split": "profile"}])
        self.assertNotIn("label", rows[0])

    def test_duplicate_ids_fail(self) -> None:
        with self.assertRaises(ValueError):
            normalize([{"id": "a", "image": "a.jpg"}, {"id": "a", "image": "b.jpg"}])

    def test_group_leakage_is_rejected(self) -> None:
        rows = copy.deepcopy(self.bundle["rows"])
        rows[1]["group"] = rows[0]["group"]
        rows[1]["split"] = "test"
        with self.assertRaises(ValueError):
            audit_splits(rows)

    def test_content_leakage_is_rejected(self) -> None:
        rows = copy.deepcopy(self.bundle["rows"])
        rows[0]["content_sha256"] = "same"
        rows[-1]["content_sha256"] = "same"
        with self.assertRaises(ValueError):
            audit_splits(rows)

    def test_group_partition_removes_target_labels(self) -> None:
        rows = normalize(self.bundle["rows"])
        result = partition(rows, ["target_gamma"])
        audit_splits(result)
        self.assertTrue(all("label" not in row for row in result if row["split"] == "profile"))
        self.assertTrue(all(row["domain"] == "target_gamma" for row in result if row["split"] == "profile"))

    def test_normalized_duplicate_text_cannot_cross_partitions(self) -> None:
        rows = normalize([{"id": "a", "group": "a", "text": "Cafe\u0301 news\nreport", "split": "train"}, {"id": "b", "group": "b", "text": "Caf\u00e9  news report", "split": "test"}])
        self.assertEqual(text_fingerprint(rows[0]["text"]), text_fingerprint(rows[1]["text"]))
        with self.assertRaisesRegex(ValueError, "Cross-split leakage"):
            audit_splits(rows)
        rows[1]["split"] = "train"
        self.assertEqual(audit_splits(rows)["leakage"]["text"], [])
        self.assertIsNone(text_fingerprint(" \n "))
        for value in ("null", "none", "nan"):
            self.assertIsNotNone(text_fingerprint(value))

    def test_partition_keeps_transitive_evidence_components_together(self) -> None:
        rows = normalize([{"id": str(index), "text": "Independent report " + str(index), "group": str(index), "domain": "source" if index < 12 else "target", "label": index % 2} for index in range(18)])
        rows[0]["group"] = rows[1]["group"] = "bridge"
        rows[1]["image"] = rows[2]["image"] = "./bridge.jpg"
        rows[3]["text"] = rows[2]["text"]
        first = partition(rows, ["target"], seed=42)
        second = partition(rows, ["target"], seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first, partition(rows[::-1], ["target"], seed=42))
        lookup = {row["id"]: row["split"] for row in first}
        self.assertEqual(len({lookup[str(index)] for index in range(4)}), 1)
        self.assertEqual(audit_splits(first)["samples"], 18)
        self.assertTrue(all("label" not in row for row in first if row["split"] == "profile"))
        for seed in (True, -1, 2 ** 32, 1.5):
            with self.assertRaises(ValueError):
                partition(rows, ["target"], seed=seed)

    def test_cross_domain_evidence_bridges_cannot_enter_holdout_splits(self) -> None:
        original = normalize([{"id": str(index), "text": "Independent report " + str(index), "group": str(index), "domain": "source" if index < 8 else "target"} for index in range(12)])
        for field, value in (("image", "./shared.jpg"), ("content_sha256", "shared-content"), ("text", "The same independent report.")):
            rows = copy.deepcopy(original)
            rows[0][field] = rows[-1][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "crosses source and target"):
                    partition(rows, ["target"])

    def test_available_image_bytes_are_checked_without_declared_hashes(self) -> None:
        first, second = self.root / "first.png", self.root / "second.png"
        Image.new("RGB", (64, 64), (30, 60, 90)).save(first)
        second.write_bytes(first.read_bytes())
        rows = normalize([{"id": str(index), "group": str(index), "text": "Report " + str(index), "domain": "source" if index < 8 else "target", "split": "train" if index < 8 else "test"} for index in range(12)])
        rows[0]["image"], rows[-1]["image"] = first.as_posix(), second.as_posix()
        with self.assertRaisesRegex(ValueError, "Cross-split leakage"):
            audit_splits(rows)
        with self.assertRaisesRegex(ValueError, "crosses source and target"):
            partition(rows, ["target"])
        rows[-1]["image"] = ""
        rows[0]["content_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "checksum is stale"):
            audit_splits(rows)

    def test_semantic_join_checks_identifiers_and_direction(self) -> None:
        rows = self.bundle["rows"][:2]
        predictions = [{"id": row["id"], "logit": 2.0} for row in rows]
        self.assertEqual(join_predictions(rows, predictions, real_positive=True)[0]["semantic_logit"], -2)
        with self.assertRaises(ValueError):
            join_predictions(rows, predictions[:1])

    def test_profile_fitting_is_label_independent(self) -> None:
        data = subset(self.bundle, indices(self.bundle, "profile", "target_gamma"))
        altered = copy.deepcopy(data)
        for row in altered["rows"]:
            row["label"] = 1
        first = DomainProfile.fit(data, "target_gamma", self.config)
        second = DomainProfile.fit(altered, "target_gamma", self.config)
        np.testing.assert_array_equal(first.reference, second.reference)
        np.testing.assert_array_equal(first.weights, second.weights)

    def test_profile_serialization_preserves_predictions(self) -> None:
        profile = self.profile()
        path = (self.root / "profile.json").as_posix()
        profile.save(path)
        restored = DomainProfile.load(path)
        test = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        first, second = profile.predict(test), restored.predict(test)
        np.testing.assert_allclose([row["physical_risk"] for row in first], [row["physical_risk"] for row in second])

    def test_bundle_roundtrip_uses_no_pickle(self) -> None:
        path = (self.root / "features.npz").as_posix()
        save_bundle(path, self.bundle["values"], self.bundle["observed"], self.bundle["patches"], self.bundle["rows"], NAMES)
        restored = load_bundle(path)
        np.testing.assert_array_equal(restored["values"], self.bundle["values"])
        self.assertEqual(restored["rows"], self.bundle["rows"])

    def test_missing_and_constant_features_are_excluded(self) -> None:
        data = subset(self.bundle, indices(self.bundle, "profile", "target_gamma"))
        data["observed"][:, 0] = False
        data["values"][:, 1] = 0
        profile = DomainProfile.fit(data, "target_gamma", self.config)
        self.assertNotIn(0, profile.columns)
        self.assertNotIn(1, profile.columns)

    def test_physical_fusion_has_correct_orientation(self) -> None:
        profile = self.profile()
        data = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        row = profile.predict(data)[0]
        self.assertAlmostEqual(row["physical_evidence"], row["copula_score"] + self.config.sign_weight * row["sign_score"])
        self.assertEqual(int(row["physical_evidence"] <= profile.quality["threshold"]), row["prediction"])
        np.testing.assert_array_equal(fuse(np.array([-1, 1]), np.array([1, -1])), [1, 1])

    def test_mma_patch_inconsistency_is_not_silently_scored(self) -> None:
        data = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        data["rows"][0]["patch_signs_valid_after_attack"] = False
        with self.assertRaises(ValueError):
            self.profile().predict(data)
        self.assertTrue(self.profile().predict(data, signs=False))

    def test_genuine_calibration_rejects_test_data(self) -> None:
        test = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        with self.assertRaises(ValueError):
            self.profile().calibrate_genuine(test)

    def test_source_genuine_quantile_is_available(self) -> None:
        validation = subset(self.bundle, indices(self.bundle, "validation", "source_alpha"))
        profile = self.profile()
        profile.calibrate_genuine(validation)
        self.assertEqual(profile.quality["genuine_calibration_samples"], 8)

    def test_labeled_calibration_is_required(self) -> None:
        rows = [{"id": "a", "fusion_risk": 0}]
        with self.assertRaises(ValueError):
            calibrate_rows(rows, rows)

    def test_probability_calibration_cannot_flip_orientation(self) -> None:
        platt = Platt.fit(np.array([-2, -1, 1, 2]), np.array([0, 0, 1, 1]))
        self.assertGreaterEqual(platt.slope, 0)
        self.assertGreater(platt.probability(np.array([2]))[0], platt.probability(np.array([-2]))[0])

    def test_pooled_profile_is_positive_definite(self) -> None:
        prior = pool([self.profile("source_alpha"), self.profile("source_beta")])
        self.assertGreater(np.linalg.eigvalsh(prior.reference).min(), 0)
        self.assertEqual(prior.quality["target_samples"], 0)

    def test_cold_start_shrinkage_has_exact_weight(self) -> None:
        prior = pool([self.profile("source_alpha"), self.profile("source_beta")])
        target = subset(self.bundle, indices(self.bundle, "profile", "target_gamma"))
        result = adapt(prior, target, "target_gamma", n0=64)
        self.assertAlmostEqual(result.quality["shrinkage_weight"], 16 / 80)
        self.assertGreater(np.linalg.eigvalsh(result.reference).min(), 0)

    def test_pixel_extraction_is_deterministic_and_finite(self) -> None:
        rng = np.random.default_rng(6)
        path = self.root / "image.jpg"
        Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(path, quality=80)
        row = {"id": "image", "image": path.as_posix(), "text": "A small test!", "domain": "a", "split": "test"}
        first = extract_one(row, TraceConfig())
        second = extract_one(row, TraceConfig())
        np.testing.assert_array_equal(first[0], second[0])
        self.assertTrue(np.isfinite(first[2]).all())
        self.assertEqual(first[2].shape, (16, len(NAMES)))
        self.assertTrue(first[1][12])

    def test_single_class_auc_is_unavailable(self) -> None:
        result = binary(np.array([0, 0]), np.array([-1, 1]))
        self.assertIsNone(result["auc"])

    def test_three_partition_benchmark_preserves_the_supplied_samples(self) -> None:
        selected = [index for index, row in enumerate(self.bundle["rows"]) if row["domain"].startswith("source") and row["split"] in ("train", "validation", "test")]
        bundle = subset(self.bundle, selected)
        before = copy.deepcopy(bundle["rows"])
        output = self.root / "official"
        result = benchmark(bundle, output.as_posix(), replace(self.config, bootstrap=2), bootstraps=20)
        self.assertEqual(bundle["rows"], before)
        states = read_json(output / "calibration.json")
        self.assertTrue(result["methods"])
        for state in states.values():
            self.assertEqual(state["version"], "flawprint-probability-v1")
            self.assertIsNone(state["conformal"])
        for method, metrics in result["methods"].items():
            self.assertEqual(metrics["samples"], sum(row["split"] == "test" for row in before))

    def test_complete_benchmark_runs_controls_and_ablations(self) -> None:
        result = benchmark(self.bundle, (self.root / "benchmark").as_posix(), self.config, bootstraps=2, ablations=True)
        self.assertIn("flawprint_fusion", result["methods"])
        self.assertIn("without_capture", result["methods"])
        self.assertIn("source_profile_transfer", result["methods"])
        self.assertIn("image_only_fusion", result["methods"])
        self.assertIn("shuffled_sign_weights", result["methods"])
        for method in ("trace_rbf", "image_rbf", "trace_rbf_fusion", "image_rbf_fusion"):
            self.assertIn(method, result["methods"])
        self.assertEqual(len(result["methods"]), 22)
        state = read_json(self.root / "benchmark" / "calibration.json")
        self.assertEqual(state["image_rbf_fusion"]["head_fit"]["features"], list(NAMES[:20]))
        self.assertTrue((self.root / "benchmark" / "summary.csv").exists())

    def test_benchmark_records_missing_semantic_control_inputs(self) -> None:
        bundle = copy.deepcopy(self.bundle)
        for row in bundle["rows"]:
            row.pop("semantic_logit", None)
        result = benchmark(bundle, (self.root / "withoutsemantic").as_posix(), self.config, bootstraps=2)
        for method in ("semantic_external", "trace_rbf_fusion", "image_rbf_fusion", "flawprint_fusion"):
            self.assertIn(method, result["skipped"])
            self.assertNotIn(method, result["methods"])
        for method in ("trace_rbf", "image_rbf"):
            self.assertIn(method, result["methods"])

    def test_repository_conventions_are_clean(self) -> None:
        self.assertTrue(conventions()["passed"])

class OfficialTests(unittest.TestCase):
    def test_small_frequency_tiles_stay_finite(self):
        from engine.traces import pixel_traces
        rng = np.random.default_rng(42)
        for shape in ((8, 8, 3), (8, 19, 3), (32, 32, 3)):
            for values in (np.zeros(shape), rng.random(shape)):
                self.assertTrue(np.isfinite(pixel_traces(values)).all())

    def test_noise_slope_matches_patch_definition(self):
        from engine.traces import noise_slope
        rng = np.random.default_rng(42)
        for shape in ((8, 8), (16, 16), (57, 79), (192, 192)):
            gray, residual = rng.random(shape), rng.normal(size=shape)
            values = np.asarray([(gray[y:y+8, x:x+8].mean(), residual[y:y+8, x:x+8].var(), gray[y:y+8, x:x+8].var()) for y in range(0, shape[0]-7, 8) for x in range(0, shape[1]-7, 8)])
            expected = 0.0
            if len(values) >= 4:
                values = values[values[:, 2] <= np.quantile(values[:, 2], 0.5)]
                if len(values) >= 3 and np.ptp(values[:, 0]) >= 1e-8:
                    expected = float(np.polyfit(values[:, 0], values[:, 1], 1)[0])
            self.assertAlmostEqual(noise_slope(gray, residual), expected, places=11)

    def test_parallel_trace_output_matches_serial(self):
        from engine.traces import extract, TraceConfig
        rows = [{"id": str(i), "image": "", "text": "Example caption " + str(i)} for i in range(4)]
        serial = extract(rows, TraceConfig(), workers=1)
        parallel = extract(rows, TraceConfig(), workers=2)
        for first, second in zip(serial[:3], parallel[:3]):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(serial[3], parallel[3])

    def paper(self, ordinal, role):
        return {"id": "finefake:" + str(ordinal), "source_ordinal": ordinal, "group": "finefake:" + str(ordinal), "split": role, "benchmark_protocol": "finefake-paper-3407", "text": "Shared published caption.", "image": "", "domain": "Politics", "label": 0}

    def test_paper_protocol_reports_overlap_and_preserves_rows(self):
        from engine.data import audit_splits
        rows = [self.paper(0, "train"), self.paper(1, "test")]
        report = audit_splits(rows)
        self.assertEqual(len(report["leakage"]["text"]), 1)
        self.assertEqual(report["benchmark_protocol"], "finefake-paper-3407")
        self.assertEqual([row["split"] for row in rows], ["train", "test"])

    def test_paper_history_preserves_within_role_exclusion(self):
        from engine.lineage import collect, overlaps
        history = collect([self.paper(0, "train")])
        self.assertEqual(overlaps([self.paper(1, "test")], history), [False])
        self.assertEqual(overlaps([self.paper(2, "train")], history), [True])
        self.assertEqual(overlaps([self.paper(0, "test")], history), [True])

    def test_paper_contract_rejects_unknown_and_mixed_policies(self):
        from engine.data import audit_splits
        first, second = self.paper(0, "train"), self.paper(1, "test")
        second.pop("benchmark_protocol")
        with self.assertRaises(ValueError):
            audit_splits([first, second])
        first["benchmark_protocol"] = "finefake-paper-42"
        with self.assertRaises(ValueError):
            audit_splits([first])
        first = self.paper(0, "profile")
        with self.assertRaises(ValueError):
            audit_splits([first])

    def test_strict_content_policy_still_rejects_shared_text(self):
        from engine.data import audit_splits
        rows = [self.paper(0, "train"), self.paper(1, "test")]
        for row in rows:
            row.pop("benchmark_protocol")
        with self.assertRaises(ValueError):
            audit_splits(rows)

    def test_first_frame_and_physical_fallback_keep_news_images(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from engine.traces import extract_one, TraceConfig
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory).resolve().relative_to(Path.cwd().resolve()) / "animated.gif"
            first = Image.new("RGB", (64, 64), "red")
            second = Image.new("RGB", (64, 64), "blue")
            first.save(path, save_all=True, append_images=[second], duration=100)
            row = {"id": "news", "text": "A caption.", "image": path.as_posix(), "image_frame": 0}
            values, observed, patches, metadata = extract_one(row, TraceConfig())
            self.assertTrue(metadata["has_image"])
            self.assertTrue(observed[:20].any())
            row["physical_trace_disabled"] = True
            values, observed, patches, metadata = extract_one(row, TraceConfig())
            self.assertTrue(metadata["has_image"])
            self.assertFalse(observed[:20].any())
            self.assertTrue(observed[20:].all())
            row["image_frame"] = True
            with self.assertRaises(ValueError):
                extract_one(row, TraceConfig())

    def test_disabled_physical_traces_reject_stale_image_identity(self):
        from engine.storage import digest
        from engine.traces import extract_one, TraceConfig
        with tempfile.TemporaryDirectory(dir="./tmp") as directory:
            path = Path(directory).resolve().relative_to(Path.cwd().resolve()) / "narrow.png"
            Image.new("RGB", (8, 64), "red").save(path)
            row = {"id": "narrow", "text": "A caption.", "image": path.as_posix(), "physical_trace_disabled": True, "content_sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "checksum is stale"):
                extract_one(row, TraceConfig())
            row["content_sha256"] = digest(path)
            _, observed, _, metadata = extract_one(row, TraceConfig())
            self.assertEqual(metadata["content_sha256"], row["content_sha256"])
            self.assertFalse(observed[:20].any())
            self.assertTrue(metadata["has_image"])

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
