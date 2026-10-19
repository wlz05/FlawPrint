# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import replace
import unittest
import numpy as np
from engine.data import normalize, audit_splits
from engine.traces import extract_one, TraceConfig
from engine.synthetic import fixture
from engine.storage import subset
from engine.experiment import indices
from engine.profile import DomainProfile, ProfileConfig
from engine.attacks import joint_replay
from engine.metrics import binary
from engine.decision import GatePolicy, physical_status, route_rows, deploy, calibrate_routes, apply_routes, deployment_report
class DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GatePolicy()
        self.status = {"physical_available": True, "physical_gate_reasons": [], "image_observed_fraction": 1.0, "image_profile_features": 20, "physical_power_screen": {"power_screen": True}}
    def routed(self, identifier: str, label: int, split: str, route: str = "semantic") -> dict:
        status = dict(self.status, physical_available=route == "fusion")
        row = {"id": identifier, "domain": "a", "platform": "p", "split": split, "label": label, "semantic_logit": 2.0 if label else -2.0, "physical_risk": -1.0}
        return route_rows([row], [status], self.policy)[0]
    def test_text_only_extraction_marks_image_features_missing(self) -> None:
        row = normalize([{"id": "text", "text": "A factual claim!", "domain": "a", "split": "test"}])[0]
        values, observed, patches, metadata = extract_one(row, TraceConfig())
        self.assertFalse(observed[:20].any())
        self.assertTrue(observed[20:].all())
        self.assertFalse(metadata["has_image"])
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(patches == 0))
    def test_missing_images_are_not_shared_leakage_identifiers(self) -> None:
        rows = normalize([{"id": "a", "split": "train"}, {"id": "b", "split": "test"}])
        self.assertEqual(audit_splits(rows, True)["leakage"]["paths"], [])
    def test_fractional_binary_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize([{"id": "a", "label": 0.5}])
    def test_real_photo_cannot_override_fake_text(self) -> None:
        row = {"id": "fake_text", "physical_risk": -4.0, "semantic_logit": 2.0}
        result = route_rows([row], [self.status], self.policy)[0]
        self.assertEqual(result["decision_route"], "fusion")
        self.assertEqual(result["prediction"], 1)
    def test_unavailable_physics_cannot_override_semantics(self) -> None:
        row = {"id": "compressed", "physical_risk": 9.0, "semantic_logit": -2.0}
        status = dict(self.status, physical_available=False, physical_gate_reasons=["domain_power_screen_failed"])
        result = route_rows([row], [status], self.policy)[0]
        self.assertEqual(result["decision_route"], "semantic")
        self.assertEqual(result["fusion_risk"], -2)
    def test_no_evidence_returns_abstention_without_fake_score(self) -> None:
        status = dict(self.status, physical_available=False)
        result = apply_routes(route_rows([{"id": "empty"}], [status], self.policy))[0]
        self.assertTrue(result["abstain"])
        self.assertIsNone(result["prediction"])
        self.assertIsNone(result["fusion_risk"])
        self.assertIsNone(result["probability"])
    def test_identity_correlation_fails_physical_power_gate(self) -> None:
        bundle = fixture(per_split=16)
        profile = DomainProfile.fit(subset(bundle, indices(bundle, "profile", "target_gamma")), "target_gamma", ProfileConfig(bootstrap=5, min_samples=12))
        profile.reference = np.eye(len(profile.reference))
        test = subset(bundle, indices(bundle, "test", "target_gamma"))
        statuses = physical_status(profile, test, self.policy)
        self.assertTrue(all("domain_power_screen_failed" in status["physical_gate_reasons"] for status in statuses))
        test["rows"][0]["has_image"] = False
        test["observed"][0, :20] = False
        first = physical_status(profile, test, self.policy)[0]
        self.assertIn("missing_image", first["physical_gate_reasons"])
        self.assertIn("insufficient_observed_image_features", first["physical_gate_reasons"])
    def test_deployment_without_profile_uses_semantics(self) -> None:
        bundle = fixture(per_split=16)
        test = subset(bundle, indices(bundle, "test", "target_gamma"))
        rows = deploy(test, None, self.policy)
        self.assertTrue(all(row["decision_route"] == "semantic" for row in rows))
    def test_route_calibration_preserves_abstention(self) -> None:
        validation = [self.routed(f"v{i}", i % 2, "validation") for i in range(8)]
        calibration = [self.routed(f"c{i}", i % 2, "calibration") for i in range(2)]
        state = calibrate_routes(validation, calibration, alpha=0.1)
        row = self.routed("test", 1, "test")
        result = apply_routes([row], state)[0]
        self.assertEqual(result["prediction_set"], [0, 1])
        self.assertIsNone(result["prediction"])
        self.assertTrue(result["abstain"])
        with self.assertRaises(ValueError):
            apply_routes([validation[0]], state)
        altered = dict(row, routing_policy=vars(replace(self.policy, min_observed=0.9)))
        with self.assertRaises(ValueError):
            apply_routes([altered], state)
    def test_cross_route_calibration_overlap_is_rejected(self) -> None:
        validation = [self.routed("same", 0, "validation")]
        calibration = [self.routed("same", 1, "calibration", "fusion")]
        with self.assertRaises(ValueError):
            calibrate_routes(validation, calibration)
    def test_deployment_coverage_counts_unscored_samples(self) -> None:
        first = self.routed("scored", 1, "test")
        second = route_rows([{"id": "missing", "label": 0}], [dict(self.status, physical_available=False)], self.policy)[0]
        report = deployment_report(apply_routes([first, second]))
        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["scored_samples"], 1)
        self.assertEqual(report["decision_coverage"], 0.5)
        self.assertEqual(report["accepted_risk"], 0)
    def test_narrative_features_cannot_pass_image_feature_gate(self) -> None:
        bundle = fixture(per_split=16)
        profile = DomainProfile.fit(subset(bundle, indices(bundle, "profile", "target_gamma")), "target_gamma", ProfileConfig(bootstrap=5, min_samples=12), list(range(20, 26)))
        test = subset(bundle, indices(bundle, "test", "target_gamma"))
        statuses = physical_status(profile, test, self.policy)
        self.assertTrue(all("insufficient_image_features" in status["physical_gate_reasons"] for status in statuses))
    def test_joint_replay_erases_all_supplied_physical_information(self) -> None:
        bundle = fixture(per_split=16)
        profile = DomainProfile.fit(subset(bundle, indices(bundle, "profile", "target_gamma")), "target_gamma", ProfileConfig(bootstrap=5, min_samples=12))
        test = subset(bundle, indices(bundle, "test", "target_gamma"))
        genuine = [index for index, row in enumerate(test["rows"]) if row["label"] == 0]
        generated = [index for index, row in enumerate(test["rows"]) if row["label"] == 1]
        attacked = joint_replay(test, genuine, generated)
        rows = profile.predict(attacked)
        result = binary(np.array([row["label"] for row in rows]), np.array([row["physical_risk"] for row in rows]))
        self.assertAlmostEqual(result["auc"], 0.5)
        self.assertEqual([row["semantic_logit"] for row in test["rows"]], [row["semantic_logit"] for row in attacked["rows"]])
        with self.assertRaises(ValueError):
            joint_replay(test, genuine, generated[:-1])
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
