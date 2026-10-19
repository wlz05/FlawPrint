# -*- coding: utf-8 -*-
# Test Workflow: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from engine.storage import relative_path, save_bundle, load_bundle, subset
from engine.data import normalize, partition, audit_splits, join_predictions
from engine.traces import NAMES, TraceConfig, extract_one
from engine.storage import read_json
from engine.synthetic import fixture
from engine.profile import DomainProfile, ProfileConfig
from engine.conformal import Platt, calibrate_rows, apply_calibration, fuse
from engine.experiment import benchmark, indices
from engine.stream import pool, adapt
from engine.provenance import conventions
from engine.metrics import binary, group_bootstrap
class WorkflowTests(unittest.TestCase):
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
    def test_absolute_paths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            relative_path(Path.cwd())
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
if __name__ == "__main__":
    unittest.main()
