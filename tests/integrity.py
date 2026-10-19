# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from engine.storage import validate_bundle, save_bundle, load_bundle, subset, write_json
from engine.synthetic import fixture
from engine.experiment import indices
from engine.profile import DomainProfile, ProfileConfig
from engine.copula import Marginals, partial_correlation, score, moments, correlation
from engine.conformal import Platt, SplitConformal, calibrate_rows, apply_calibration, fuse
from engine.stream import pool, adapt
from engine.attacks import joint_replay, quantile_match, sinkhorn
from engine.provenance import conventions
from engine.decision import GatePolicy, route_rows, calibrate_routes, apply_routes
from engine.metrics import binary, evaluate, risk_coverage, calibration_error
from engine.baselines import Baseline
from engine.traces import NAMES
from engine.dependence import tail_report, student_score, attenuation, availability, false_discovery, platform_alignment
from engine.robustness import corrupt
from study import metrics as study_metrics

class IntegrityTests(unittest.TestCase):

    def setUp(self) -> None:
        self.bundle = fixture(per_split=16)
        self.config = ProfileConfig(bootstrap=3, min_samples=12)
        Path("./tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir="./tmp")
        self.root = Path(self.temporary.name).relative_to(Path.cwd())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def profile(self, domain: str = "target_gamma") -> DomainProfile:
        split = "profile" if domain == "target_gamma" else "train"
        return DomainProfile.fit(subset(self.bundle, indices(self.bundle, split, domain)), domain, self.config)

    def scored(self, prefix: str, split: str, size: int = 8) -> list[dict]:
        return [{"id": f"{prefix}{index}", "split": split, "label": index % 2, "fusion_risk": 2.0 if index % 2 else -2.0} for index in range(size)]

    def test_bundle_rejects_misaligned_patch_schema(self) -> None:
        damaged = dict(self.bundle, patches=self.bundle["patches"][:, :, :-1])
        with self.assertRaises(ValueError):
            validate_bundle(damaged)
        with self.assertRaises(ValueError):
            save_bundle(self.root / "broken.npz", damaged["values"], damaged["observed"], damaged["patches"], damaged["rows"], damaged["names"].tolist())
        self.assertFalse((self.root / "broken.npz").exists())

    def test_bundle_rejects_numeric_observation_masks(self) -> None:
        with self.assertRaises(ValueError):
            validate_bundle(dict(self.bundle, observed=self.bundle["observed"].astype(float)))

    def test_bundle_rejects_duplicate_names_and_identifiers(self) -> None:
        damaged = copy.deepcopy(self.bundle)
        damaged["names"][1] = damaged["names"][0]
        with self.assertRaises(ValueError):
            validate_bundle(damaged)
        damaged = copy.deepcopy(self.bundle)
        damaged["rows"][1]["id"] = damaged["rows"][0]["id"]
        with self.assertRaises(ValueError):
            validate_bundle(damaged)

    def test_bundle_loader_rejects_corrupt_archives(self) -> None:
        path = self.root / "broken.npz"
        with path.open("wb") as stream:
            np.savez_compressed(stream, values=np.ones((2, 2)), observed=np.ones((2, 2), dtype=bool), patches=np.ones((2, 4, 3)), names=np.asarray(["a", "b"]), metadata=np.asarray('[{"id":"a","domain":"x"},{"id":"b","domain":"x"}]'))
        with self.assertRaises(ValueError):
            load_bundle(path)

    def test_subset_does_not_mutate_parent_metadata(self) -> None:
        selected = subset(self.bundle, [0])
        selected["rows"][0]["attack"] = "changed"
        self.assertNotIn("attack", self.bundle["rows"][0])
        for indices in ([-1], [0, 0], [0.5], [len(self.bundle["rows"])]):
            with self.assertRaises(ValueError):
                subset(self.bundle, indices)

    def test_asymmetric_references_are_rejected(self) -> None:
        reference = np.array([[1.0, 0.9], [0.1, 1.0]])
        for operation in (partial_correlation, moments):
            with self.assertRaises(ValueError):
                operation(reference)
        with self.assertRaises(ValueError):
            score(np.ones((2, 2)), reference)

    def test_covariance_cannot_be_used_as_copula_correlation(self) -> None:
        covariance = np.array([[4.0, 0.2], [0.2, 1.0]])
        self.assertTrue(np.isfinite(partial_correlation(covariance)).all())
        with self.assertRaises(ValueError):
            moments(covariance)
        with self.assertRaises(ValueError):
            score(np.ones((2, 2)), covariance)

    def test_nonpositive_and_nonfinite_references_fail(self) -> None:
        for reference in (np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([[1.0, np.nan], [np.nan, 1.0]])):
            with self.assertRaises(ValueError):
                partial_correlation(reference)
        with self.assertRaises(ValueError):
            correlation(np.ones((4, 2)), eigen_floor=np.nan)

    def test_unsorted_marginal_state_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Marginals(np.array([[2.0, 0.0], [1.0, 1.0]]))

    def test_profile_loader_rejects_inconsistent_score_scale(self) -> None:
        document = self.profile().document()
        document["quality"]["real_variance"] += 10
        path = self.root / "damaged.json"
        write_json(path, document)
        with self.assertRaises(ValueError):
            DomainProfile.load(path.as_posix())

    def test_profile_loader_rejects_invalid_sign_weights(self) -> None:
        document = self.profile().document()
        document["weights"][0, 1] = -1
        path = self.root / "damaged.json"
        write_json(path, document)
        with self.assertRaises(ValueError):
            DomainProfile.load(path.as_posix())

    def test_profile_rejects_duplicate_columns_and_nonfinite_settings(self) -> None:
        data = subset(self.bundle, indices(self.bundle, "profile", "target_gamma"))
        with self.assertRaises(ValueError):
            DomainProfile.fit(data, "target_gamma", self.config, [0, 0, 1])
        for config in (ProfileConfig(sign_weight=np.nan), ProfileConfig(kappa=np.inf), ProfileConfig(bootstrap=2.5)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_stream_rejects_reordered_feature_schema(self) -> None:
        prior = pool([self.profile("source_alpha"), self.profile("source_beta")])
        target = subset(self.bundle, indices(self.bundle, "profile", "target_gamma"))
        target["names"][[0, 1]] = target["names"][[1, 0]]
        with self.assertRaises(ValueError):
            adapt(prior, target, "target_gamma")

    def test_stream_rejects_test_data_history_replay_and_nan_strength(self) -> None:
        prior = pool([self.profile("source_alpha"), self.profile("source_beta")])
        for target, strength in ((subset(self.bundle, indices(self.bundle, "test", "target_gamma")), 100), (subset(self.bundle, indices(self.bundle, "train", "source_alpha")), 100), (None, np.nan)):
            with self.assertRaises(ValueError):
                adapt(prior, target, target["rows"][0]["domain"] if target else "target_gamma", strength)
        with self.assertRaises(ValueError):
            pool([self.profile("source_alpha"), self.profile("source_alpha")])

    def test_calibration_rejects_duplicate_and_shared_event_samples(self) -> None:
        validation, calibration = self.scored("v", "validation"), self.scored("c", "calibration")
        with self.assertRaises(ValueError):
            calibrate_rows(validation + [validation[0]], calibration)
        validation[0]["group"], calibration[0]["group"] = "same", "same"
        with self.assertRaises(ValueError):
            calibrate_rows(validation, calibration)

    def test_calibration_rejects_equivalent_image_paths_and_profile_overlap(self) -> None:
        validation, calibration = self.scored("v", "validation"), self.scored("c", "calibration")
        validation[0]["image"], calibration[0]["image"] = "./same.jpg", "same.jpg"
        with self.assertRaises(ValueError):
            calibrate_rows(validation, calibration)
        validation, calibration = self.scored("v", "validation"), self.scored("c", "calibration")
        validation[0]["profile_overlap"] = True
        with self.assertRaises(ValueError):
            calibrate_rows(validation, calibration)

    def test_loaded_calibration_cannot_flip_score_orientation(self) -> None:
        for parameters in ({"slope": -1}, {"slope": np.nan}, {"intercept": np.inf}):
            with self.assertRaises(ValueError):
                Platt(**parameters)
        state = calibrate_rows(self.scored("v", "validation"), self.scored("c", "calibration"))
        state["version"] = "unknown"
        with self.assertRaises(ValueError):
            apply_calibration(self.scored("t", "test"), state)

    def test_conformal_state_checks_sample_count_and_probabilities(self) -> None:
        for identifiers in (["same", "same"], ["only"]):
            with self.assertRaises(ValueError):
                SplitConformal(0.1, 0.8, 2, identifiers)
        conformal = SplitConformal(0.1, 0.8, 2, ["a", "b"])
        for probabilities in (np.array([np.nan]), np.array([1.1]), np.array([[0.5]])):
            with self.assertRaises(ValueError):
                conformal.prediction_sets(probabilities)
        with self.assertRaises(ValueError):
            SplitConformal(0.1, -0.1, 2, [])

    def test_fusion_rejects_nonfinite_evidence(self) -> None:
        with self.assertRaises(ValueError):
            fuse(np.array([np.nan]))
        with self.assertRaises(ValueError):
            fuse(np.array([0.0]), np.array([np.inf]))

    def test_joint_replay_rejects_invalid_indices_and_sample_roles(self) -> None:
        target = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        genuine = [index for index, row in enumerate(target["rows"]) if row["label"] == 0]
        generated = [index for index, row in enumerate(target["rows"]) if row["label"] == 1]
        for first, second in ((genuine + [genuine[0]], generated + [generated[0]]), ([-1], [generated[0]]), (generated, genuine)):
            with self.assertRaises(ValueError):
                joint_replay(target, first, second)
        with self.assertRaises(ValueError):
            quantile_match(np.array([[np.nan], [1.0]]), np.ones((2, 1)))

    def test_nonfinite_physical_scores_cannot_report_available_evidence(self) -> None:
        status = {"physical_available": True, "physical_gate_reasons": []}
        row = {"id": "invalid", "physical_risk": np.nan, "semantic_logit": -2.0}
        result = route_rows([row], [status], GatePolicy())[0]
        self.assertEqual(result["decision_route"], "semantic")
        self.assertFalse(result["physical_available"])
        self.assertIn("missing_or_nonfinite_physical_score", result["physical_gate_reasons"])

    def test_sparse_route_does_not_break_other_route_calibration(self) -> None:
        status = {"physical_available": False, "physical_gate_reasons": []}
        validation = route_rows([dict(row, semantic_logit=row["fusion_risk"]) for row in self.scored("v", "validation")], [status] * 8, GatePolicy())
        calibration = route_rows([dict(row, semantic_logit=row["fusion_risk"]) for row in self.scored("c", "calibration")], [status] * 8, GatePolicy())
        validation.append(dict(validation[0], id="sparsev", decision_route="physical"))
        calibration.append(dict(calibration[0], id="sparsec", decision_route="physical"))
        state = calibrate_routes(validation, calibration)
        self.assertIn("semantic", state["routes"])
        self.assertIn("physical", state["skipped_routes"])
        row = dict(calibration[-1], id="test", split="test")
        self.assertEqual(apply_routes([row], state)[0]["calibration_status"], "uncalibrated_route")

    def test_unknown_routes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_routes([{"id": "invalid", "decision_route": "unknown"}])

    def test_filename_audit_rejects_underscores(self) -> None:
        path = self.root / "bad_name.py"
        path.write_text("value = 1\n", encoding="utf-8")
        report = conventions(self.root.as_posix())
        self.assertFalse(report["passed"])
        self.assertTrue(any(issue["rule"] == "lowercase-word-filename" for issue in report["issues"]))

    def test_metrics_reject_fractional_labels_without_truncation(self) -> None:
        with self.assertRaises(ValueError):
            binary(np.array([0, 0.5]), np.array([-1, 1]))
        rows = [{"label": 0, "fusion_risk": -1}, {"label": 0.5, "fusion_risk": 1}]
        with self.assertRaises(ValueError):
            evaluate(rows)

    def test_metrics_reject_broadcast_or_fractional_predictions(self) -> None:
        for predictions in (np.array([0]), np.array([0, 0.5])):
            with self.assertRaises(ValueError):
                binary(np.array([0, 1]), np.array([-1, 1]), predictions=predictions)

    def test_risk_and_calibration_metrics_validate_probabilities(self) -> None:
        for probabilities in (np.array([np.nan, 0.5]), np.array([-0.1, 1.1])):
            with self.assertRaises(ValueError):
                calibration_error(np.array([0, 1]), probabilities)
            with self.assertRaises(ValueError):
                risk_coverage(np.array([0, 1]), probabilities)
        with self.assertRaises(ValueError):
            risk_coverage(np.array([0, 1]), np.array([0.1, 0.9]), points=0)

    def test_baseline_rejects_fractional_training_labels(self) -> None:
        train = subset(self.bundle, indices(self.bundle, "train", "source_alpha"))
        train["rows"][0]["label"] = 0.5
        with self.assertRaises(ValueError):
            Baseline.fit("trace_logistic", train)

    def test_rbf_control_reads_joint_information_with_equal_marginals(self) -> None:
        train = subset(self.bundle, indices(self.bundle, "train"))
        test = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        for bundle in (train, test):
            for index, row in enumerate(bundle["rows"]):
                first, second = (-1.0, 1.0)[index % 2], (-1.0, 1.0)[index // 2 % 2]
                bundle["values"][index] = 0.0
                bundle["values"][index, :2] = (first, second)
                bundle["observed"][index] = True
                row["label"] = int(first * second > 0)
        target = np.asarray([row["label"] for row in test["rows"]])
        for column in (0, 1):
            np.testing.assert_array_equal(np.sort(test["values"][target == 0, column]), np.sort(test["values"][target == 1, column]))
        nonlinear = Baseline.fit("trace_rbf", train)
        linear = Baseline.fit("trace_logistic", train)
        self.assertGreater(binary(target, nonlinear.predict(test))["auc"], 0.99)
        self.assertAlmostEqual(binary(target, linear.predict(test))["auc"], 0.5)

    def test_image_rbf_excludes_narrative_and_test_labels(self) -> None:
        train = subset(self.bundle, indices(self.bundle, "train"))
        query = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        model = Baseline.fit("image_rbf", train)
        original = model.predict(query)
        changed = copy.deepcopy(query)
        changed["values"][:, 20:] += 10000
        for row in changed["rows"]:
            row["label"] = 1 - row["label"]
            row["text"] = "Unrelated narrative statistics."
        np.testing.assert_array_equal(original, model.predict(changed))
        self.assertEqual(model.fit_state["features"], list(NAMES[:20]))
        self.assertFalse(model.fit_state["target_profile_used"])
        self.assertEqual(model.fit_state["training_identifiers"], [row["id"] for row in train["rows"]])
        reordered = copy.deepcopy(query)
        reordered["names"][[0, 1]] = reordered["names"][[1, 0]]
        with self.assertRaises(ValueError):
            model.predict(reordered)

    def test_rbf_missing_values_and_frozen_fusion_contract(self) -> None:
        train = subset(self.bundle, indices(self.bundle, "train"))
        query = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        train["observed"][:, 0] = False
        query["observed"][:, 0] = False
        train["values"][:, 0] = 10000
        query["values"][:, 0] = -10000
        for prefix in ("trace", "image"):
            physical = Baseline.fit(prefix + "_rbf", train)
            combined = Baseline.fit(prefix + "_rbf_fusion", train)
            expected = fuse(physical.predict(query), np.asarray([row["semantic_logit"] for row in query["rows"]]))
            np.testing.assert_allclose(combined.predict(query), expected)
            self.assertTrue(np.isfinite(expected).all())
            self.assertEqual(combined.fit_state["imputation_medians"][0], 0)
            missing = copy.deepcopy(query)
            missing["rows"][0].pop("semantic_logit")
            with self.assertRaises(ValueError):
                combined.predict(missing)
            missing["rows"][0]["semantic_logit"] = np.inf
            with self.assertRaises(ValueError):
                combined.predict(missing)

    def test_rbf_rejects_calibration_training_and_selection(self) -> None:
        train = subset(self.bundle, indices(self.bundle, "train"))
        calibration = subset(self.bundle, indices(self.bundle, "calibration"))
        for method in ("trace_rbf", "image_rbf", "trace_logistic"):
            with self.assertRaises(ValueError):
                Baseline.fit(method, calibration)
            with self.assertRaises(ValueError):
                Baseline.fit(method, train, calibration)
        narrative = copy.deepcopy(train)
        narrative["names"] = np.asarray(["custom" + str(index) for index in range(len(NAMES))])
        with self.assertRaises(ValueError):
            Baseline.fit("image_rbf", narrative)

    def test_baseline_rejects_reordered_prediction_features(self) -> None:
        train = subset(self.bundle, indices(self.bundle, "train", "source_alpha"))
        test = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        model = Baseline.fit("trace_logistic", train)
        test["names"][[0, 1]] = test["names"][[1, 0]]
        with self.assertRaises(ValueError):
            model.predict(test)

    def test_baseline_rejects_validation_overlap_and_test_training(self) -> None:
        train = subset(self.bundle, indices(self.bundle, "train", "source_alpha"))
        with self.assertRaises(ValueError):
            Baseline.fit("single_trace", train, train)
        test = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        with self.assertRaises(ValueError):
            Baseline.fit("trace_logistic", test)

    def test_study_metrics_reject_profile_overlap_and_reordering(self) -> None:
        target = subset(self.bundle, indices(self.bundle, "test", "target_gamma"))
        rows = self.profile().predict(target)
        with self.assertRaises(ValueError):
            study_metrics(target, list(reversed(rows)), 0)
        rows[0]["profile_overlap"] = True
        with self.assertRaises(ValueError):
            study_metrics(target, rows, 0)

    def test_tail_and_student_diagnostics_reject_invalid_uniforms(self) -> None:
        values = np.full((8, 2), 0.5)
        for bad in (np.nan, -0.1, 1.1):
            changed = values.copy()
            changed[0, 0] = bad
            with self.assertRaises(ValueError):
                tail_report(changed, ["a", "b"])
            with self.assertRaises(ValueError):
                student_score(changed, np.eye(2))
        with self.assertRaises(ValueError):
            student_score(values, np.eye(2), degrees=np.nan)

    def test_attenuation_and_availability_reject_nan_parameters(self) -> None:
        for settings in ({"epsilon": np.nan}, {"tolerance": np.inf}, {"iterations": 2.5}):
            with self.assertRaises(ValueError):
                sinkhorn(np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.eye(2), **settings)
        with self.assertRaises(ValueError):
            attenuation(np.eye(2), np.array([np.nan, 0.5]))
        with self.assertRaises(ValueError):
            availability(np.eye(2), kappa=np.nan)
        with self.assertRaises(ValueError):
            availability(np.eye(2), batch_size=1.5)
        with self.assertRaises(ValueError):
            false_discovery(np.array([np.nan]))

    def test_degenerate_platform_axes_do_not_report_fake_alignment(self) -> None:
        reference = np.full((3, 3), 0.4)
        np.fill_diagonal(reference, 1)
        report = platform_alignment({"empty": np.eye(3), "signal": reference})
        self.assertIsNone(report["directions"]["empty"])
        self.assertTrue(report["diagnostics"]["signal"]["direction_identifiable"])
        cross = next(pair for pair in report["pairs"] if pair["first"] == "empty" and pair["second"] == "signal")
        self.assertIsNone(cross["absolute_cosine"])

    def test_nonfinite_corruption_fails_before_image_io(self) -> None:
        with self.assertRaises(ValueError):
            corrupt([{"id": "image", "image": "missing.png"}], self.root.as_posix(), "noise", np.nan)

    def test_classwise_metrics_preserve_real_and_fake_orientation(self) -> None:
        result = binary(np.array([0, 0, 0, 1, 1]), np.array([-1, 1, 1, -1, 1]))
        self.assertAlmostEqual(result["accuracy"], 0.4)
        self.assertAlmostEqual(result["fake_precision"], 1 / 3)
        self.assertAlmostEqual(result["fake_recall"], 0.5)
        self.assertAlmostEqual(result["real_precision"], 0.5)
        self.assertAlmostEqual(result["real_recall"], 1 / 3)
        self.assertAlmostEqual(result["real_f1"], 0.4)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
