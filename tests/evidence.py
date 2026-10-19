# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from engine.evidence import paired, mixture, mechanism, stratified, orientation
from engine.experiment import indices
from engine.profile import DomainProfile, ProfileConfig
from engine.metrics import evaluate
from engine.decision import deployment_report
from engine.results import prediction_rows
from engine.storage import read_rows, write_rows, subset
from engine.synthetic import fixture
class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        Path("./tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir="./tmp")
        self.root = Path(self.temporary.name).relative_to(Path.cwd())
        self.rows = [{"id": str(index), "label": index % 2, "fusion_risk": (index % 2) * 2 - 1, "prediction": index % 2, "probability": 0.8 if index % 2 else 0.2, "probability_calibrated": False, "prediction_set": [index % 2], "abstain": False, "profile_overlap": False, "split": "test", "domain": "first" if index < 8 else "second", "group": str(index)} for index in range(16)]
        self.config = ProfileConfig(min_samples=12, bootstrap=2)
    def tearDown(self) -> None:
        self.temporary.cleanup()
    def pool(self) -> tuple[dict, dict]:
        bundle = fixture(per_split=24)
        pool = subset(bundle, indices(bundle, "test", "target_gamma"))
        for row in pool["rows"]:
            row.update(id="pool:" + row["id"], group="pool:" + row["group"], image="./synthetic/pool/" + row["id"] + ".png", split="validation")
        test = subset(fixture(seed=901, per_split=24), indices(bundle, "test", "target_gamma"))
        return pool, test
    def roles(self) -> tuple[list[dict], list[dict], list[dict]]:
        reference, candidate, annotations = [], [], []
        for index in range(24):
            row = dict(self.rows[index % 16], id=str(index), label=index % 2, group=f"event{index // 2}", domain="first" if index % 4 < 2 else "second")
            for output, wrong in ((reference, 8 <= index < 16 and index % 2 == 1), (candidate, index < 8 and index % 2 == 0)):
                decision = 1 - row["label"] if wrong else row["label"]
                output.append(dict(row, prediction=decision, fusion_risk=2 * decision - 1, probability=0.8 if decision else 0.2, prediction_set=[decision]))
            annotation = {"id": row["id"], "physical_manipulated": index >= 8, "annotation_source": "synthetic-control"} if index < 16 else {"id": row["id"]}
            annotations.append(annotation)
        return reference, candidate, annotations
    def test_strata_reveal_harm_and_rescue_hidden_by_equal_aggregate_accuracy(self) -> None:
        reference, candidate, annotations = self.roles()
        result = stratified(reference, candidate, annotations, repeats=20)
        self.assertEqual(result["overall"]["delta"]["accuracy"], 0)
        self.assertAlmostEqual(result["overall"]["delta"]["auc"], 0)
        self.assertEqual(result["physical_status"]["pristine"]["transitions"]["harmed"], 4)
        self.assertEqual(result["physical_status"]["altered"]["transitions"]["rescued"], 4)
        self.assertEqual(result["accuracy_accounting"]["cell_joint_decided_total"], 24)
        self.assertEqual(result["accuracy_accounting"]["matched_accuracy_delta"], 0)
        self.assertTrue(all(cell["comparison"]["reference"]["auc"] is None for cell in result["cells"]))
    def test_strata_keep_unknown_status_and_inputs_unchanged(self) -> None:
        reference, candidate, annotations = self.roles()
        originals = copy.deepcopy((reference, candidate, annotations))
        result = stratified(reference, candidate, annotations, repeats=20)
        self.assertEqual((reference, candidate, annotations), originals)
        self.assertEqual(result["annotation_coverage"]["known"], 16)
        self.assertEqual(result["physical_status"]["unknown"]["samples"], 8)
        self.assertEqual(result["overall"]["samples"], 24)
    def test_strata_annotations_require_exact_unique_test_identifiers(self) -> None:
        reference, candidate, annotations = self.roles()
        for invalid in (annotations[:-1], annotations + [{"id": "extra"}], annotations + [annotations[0]], [dict(row, id="") if index == 0 else row for index, row in enumerate(annotations)], [{key: value for key, value in row.items() if key != "id"} for row in annotations]):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                stratified(reference, candidate, invalid, repeats=20)
    def test_strata_reject_conflicting_labels_attributes_and_missing_sources(self) -> None:
        reference, candidate, annotations = self.roles()
        for values in ({"label": 1}, {"physical_manipulated": "unknown"}, {"annotation_source": ""}, {"annotation_source": "null"}, {"annotation_source": None}):
            invalid = copy.deepcopy(annotations)
            invalid[0].update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                stratified(reference, candidate, invalid, repeats=20)
        reference[0]["physical_manipulated"] = candidate[0]["physical_manipulated"] = True
        with self.assertRaises(ValueError):
            stratified(reference, candidate, annotations, repeats=20)
    def test_strata_roundtrip_boolean_and_unknown_attributes(self) -> None:
        reference, candidate, annotations = self.roles()
        expected = stratified(reference, candidate, annotations, repeats=20)
        for suffix in ("csv", "tsv", "json", "jsonl"):
            path = self.root / ("attributes." + suffix)
            write_rows(path, annotations)
            self.assertEqual(stratified(reference, candidate, read_rows(path), repeats=20), expected)
    def test_strata_are_invariant_to_independent_input_ordering(self) -> None:
        reference, candidate, annotations = self.roles()
        result = stratified(reference, candidate, annotations, repeats=20)
        reordered = stratified(reference[::-1], candidate[::2] + candidate[1::2], annotations[::-1], repeats=20)
        self.assertEqual(result, reordered)
    def test_strata_exclude_abstention_from_error_transitions_only(self) -> None:
        reference, candidate, annotations = self.roles()
        candidate[0]["abstain"] = True
        result = stratified(reference, candidate, annotations, repeats=20)
        self.assertEqual(result["overall"]["samples"], 24)
        self.assertEqual(result["transitions"]["joint_decided"], 23)
        self.assertEqual(result["transitions"]["harmed"], 3)
        self.assertEqual(result["transitions"]["matched_accuracy_delta"], 1 / 23)
        self.assertFalse(result["accuracy_accounting"]["full_population_identity_applicable"])
    def test_strata_preserve_empty_cells_and_undefined_transition_rates(self) -> None:
        reference, candidate, annotations = self.roles()
        for row in annotations:
            row.pop("physical_manipulated", None)
            row.pop("annotation_source", None)
        result = stratified(reference, candidate, annotations, repeats=20)
        self.assertEqual(result["annotation_coverage"]["fraction"], 0)
        self.assertIsNone(result["physical_status"]["altered"]["comparison"])
        self.assertIsNone(result["physical_status"]["pristine"]["transitions"]["rescue_fraction"])
        candidate = [dict(row, abstain=True) for row in candidate]
        self.assertIsNone(stratified(reference, candidate, annotations, repeats=20)["transitions"]["matched_accuracy_delta"])
    def test_paired_rejects_disagreement_on_physical_and_narrative_attributes(self) -> None:
        for field in ("physical_manipulated", "narrative_false"):
            reference, candidate = copy.deepcopy(self.rows), copy.deepcopy(self.rows)
            reference[0][field], candidate[0][field] = False, True
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired(reference, candidate, repeats=20)
    def test_prediction_formats_preserve_metrics_and_false_flags(self) -> None:
        expected = evaluate(self.rows)
        for suffix in ("csv", "tsv", "json", "jsonl"):
            path = self.root / ("predictions." + suffix)
            write_rows(path, self.rows)
            self.assertEqual(evaluate(read_rows(path)), expected)
            self.assertIs(prediction_rows(read_rows(path))[0]["profile_overlap"], False)
    def test_mechanism_exposes_orientation_loss_and_nonlinear_dependence(self) -> None:
        result = mechanism(samples=1000, pool_size=200, repeats=1, fractions=[0, 0.5, 1])
        first, middle, last = result["population"]
        self.assertGreater(first["metrics"]["auc"], 0.8)
        self.assertEqual(middle["metrics"]["auc"], 0.5)
        self.assertGreater(result["oracle_class_likelihood_ratio_metrics"]["auc"], 0.8)
        self.assertLess(last["metrics"]["auc"], 0.2)
        self.assertTrue(first["gaussian_power_screen"] and last["gaussian_power_screen"])
        self.assertEqual(middle["expected_gap"], 0)
        self.assertGreater(middle["mixture_fourth_moment_excess"], 1)
        self.assertEqual(middle["gaussian_summary_fourth_moment_excess"], 0)
        for row in result["population"]:
            self.assertLess(row["identity_error"], 1e-12)
            self.assertLessEqual(abs(row["observed_gap"] - row["expected_gap"]), 5 * row["gap_standard_error"] + 1e-12)
    def test_mechanism_keeps_empirical_draws_reproducible_and_independent(self) -> None:
        first = mechanism(samples=100, pool_size=24, repeats=2, fractions=[0.23], seed=19)
        second = mechanism(samples=100, pool_size=24, repeats=2, fractions=[0.23], seed=19)
        self.assertEqual(first, second)
        self.assertEqual(len(first["estimated"]), 2)
        self.assertEqual(first["estimated"][0]["actual_fake_fraction"], 6 / 24)
        self.assertNotEqual(first["estimated"][0]["estimated_correlation"], first["estimated"][1]["estimated_correlation"])
    def test_mechanism_rejects_invalid_theoretical_settings(self) -> None:
        for arguments in ({"rho": 1}, {"rho": float("nan")}, {"samples": True}, {"samples": 99}, {"pool_size": 11}, {"repeats": 0}, {"shrinkage": 1.01}, {"fractions": []}, {"fractions": [1.1]}, {"fractions": [[0.5]]}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                mechanism(**arguments)
    def test_deployment_csv_preserves_abstention_and_coverage(self) -> None:
        rows = [dict(self.rows[0], physical_available=True, physical_gate_reasons=[], decision_route="physical", calibration_status="uncalibrated_route"), dict(self.rows[1], prediction=None, probability=None, prediction_set=None, fusion_risk=None, abstain=True, physical_available=False, physical_gate_reasons=["missing_image"], decision_route="abstain", calibration_status="no_available_evidence")]
        path = self.root / "deployment.csv"
        write_rows(path, rows)
        self.assertEqual(deployment_report(read_rows(path)), deployment_report(rows))
    def test_invalid_prediction_sets_and_probabilities_are_rejected(self) -> None:
        for values in ({"prediction_set": "[0,0]"}, {"prediction_set": "[2]"}, {"probability": "1.1"}, {"profile_overlap": "unknown"}):
            with self.assertRaises(ValueError):
                prediction_rows([dict(self.rows[0], **values)])
    def test_paired_comparison_is_invariant_to_row_order(self) -> None:
        result = paired(self.rows, self.rows[::-1], repeats=20)
        self.assertEqual(result["delta"]["auc"], 0)
        self.assertEqual(result["paired_intervals"]["macro_domain_f1"]["lower"], 0)
    def test_paired_improvement_is_computed_on_same_groups(self) -> None:
        reference = [dict(row, fusion_risk=-row["fusion_risk"], prediction=1 - row["prediction"], probability=1 - row["probability"]) for row in self.rows]
        result = paired(reference, self.rows, repeats=20)
        self.assertEqual(result["delta"]["auc"], 1)
        self.assertLess(result["delta"]["log_loss"], 0)
        self.assertEqual(result["paired_intervals"]["auc"]["lower"], 1)
        self.assertEqual(result["independent_groups"], 2)
    def test_paired_comparison_rejects_id_label_or_group_disagreement(self) -> None:
        for key, value in (("id", "unknown"), ("label", 1), ("group", "different")):
            candidate = copy.deepcopy(self.rows)
            candidate[0][key] = value
            with self.assertRaises(ValueError):
                paired(self.rows, candidate, repeats=20)
    def test_paired_comparison_rejects_test_selection_and_missing_scores(self) -> None:
        for key, value in (("split", "validation"), ("profile_overlap", True), ("fusion_risk", None)):
            candidate = copy.deepcopy(self.rows)
            candidate[0][key] = value
            with self.assertRaises(ValueError):
                paired(self.rows, candidate, repeats=20)
    def test_one_domain_does_not_produce_a_spurious_group_interval(self) -> None:
        rows = [dict(row, domain="only") for row in self.rows]
        self.assertIsNone(paired(rows, rows, repeats=20)["paired_intervals"]["auc"])
    def test_mixture_labels_are_removed_before_profile_fitting(self) -> None:
        pool, test = self.pool()
        original = DomainProfile.fit
        seen = []
        def checked(bundle: dict, domain: str, config: ProfileConfig, columns: list[int] | None = None) -> DomainProfile:
            seen.append(all("label" not in row and row["split"] == "profile" for row in bundle["rows"]))
            return original(bundle, domain, config, columns)
        with patch.object(DomainProfile, "fit", checked):
            result = mixture(pool, test, "target_gamma", self.config, [0, 0.5, 1], size=12, repeats=1, image_only=True)
        self.assertEqual(len(result["results"]), 3)
        self.assertTrue(all(seen))
        self.assertTrue(all("text_length" not in row["retained_features"] for row in result["results"]))
        self.assertTrue(all("label" in row for row in pool["rows"]))
    def test_mixture_sampling_is_reproducible_without_replacement(self) -> None:
        pool, test = self.pool()
        first = mixture(pool, test, "target_gamma", self.config, [0.5], size=12, repeats=2)
        second = mixture(pool, test, "target_gamma", self.config, [0.5], size=12, repeats=2)
        self.assertEqual(first, second)
        self.assertTrue(all(len(set(row["profile_ids"])) == 12 for row in first["results"]))
    def test_mixture_insufficient_classes_are_recorded_as_skips(self) -> None:
        pool, test = self.pool()
        result = mixture(pool, test, "target_gamma", self.config, [0, 0.5, 1], size=24, repeats=1)
        self.assertEqual((len(result["results"]), len(result["skipped"])), (1, 2))
    def test_mixture_rejects_overlap_or_unlabeled_construction_pools(self) -> None:
        pool, test = self.pool()
        pool["rows"][0]["group"] = test["rows"][0]["group"]
        with self.assertRaises(ValueError):
            mixture(pool, test, "target_gamma", self.config, [0.5], size=12)
        pool, test = self.pool()
        pool["rows"][0].pop("label")
        with self.assertRaises(ValueError):
            mixture(pool, test, "target_gamma", self.config, [0.5], size=12)
    def test_profile_predictions_retain_scenario_and_image_leakage_metadata(self) -> None:
        pool, test = self.pool()
        profile = DomainProfile.fit(pool, "target_gamma", self.config)
        test["rows"][0].update(scenario="real_photo_false_text", content_sha256="hash")
        result = profile.predict(test)[0]
        self.assertEqual(result["scenario"], "real_photo_false_text")
        self.assertEqual(result["image"], test["rows"][0]["image"])
        self.assertEqual(result["content_sha256"], "hash")
    def test_general_orientation_derivative_and_entropy_identity(self) -> None:
        first = np.array([[1, 0.7, 0.1], [0.7, 1, -0.2], [0.1, -0.2, 1]])
        second = first[np.ix_([1, 2, 0], [1, 2, 0])]
        self.assertGreater(np.linalg.norm(first @ second - second @ first), 0.1)
        report = orientation(first, second, [0, 0.3, 0.5, 1])
        self.assertAlmostEqual(report["crossing_fraction"], 0.5300753748558952, places=10)
        self.assertEqual(report["crossing_status"], "unique_interior_crossing")
        self.assertTrue(all(point["gap_derivative"] < 0 for point in report["points"]))
        q, step = 0.3, 1e-5
        adjacent = orientation(first, second, [q - step, q, q + step])["points"]
        derivative = (adjacent[2]["mean_gap"] - adjacent[0]["mean_gap"]) / (2 * step)
        entropy_slope = (adjacent[2]["log_determinant"] - adjacent[0]["log_determinant"]) / (2 * step)
        self.assertAlmostEqual(derivative, adjacent[1]["gap_derivative"], places=7)
        self.assertAlmostEqual(entropy_slope, adjacent[1]["mean_gap"], places=7)
        self.assertLess(max(point["identity_error"] for point in report["points"]), 1e-12)
    def test_orientation_recovers_mirror_case_and_no_crossing_case(self) -> None:
        first, second = [[1, 0.8], [0.8, 1]], [[1, -0.8], [-0.8, 1]]
        report = orientation(first, second)
        self.assertAlmostEqual(report["crossing_fraction"], 0.5)
        for point in report["points"]:
            reference = (1 - 2 * point["fake_fraction"]) * 0.8
            self.assertAlmostEqual(point["mean_gap"], 4 * 0.8 * reference / (1 - reference ** 2))
        same = orientation(first, first)
        self.assertIsNone(same["crossing_fraction"])
        self.assertTrue(all(point["mean_gap"] == 0 for point in same["points"]))
        stronger = orientation([[1, 0.2], [0.2, 1]], [[1, 0.8], [0.8, 1]])
        self.assertTrue(all(point["mean_gap"] < 0 for point in stronger["points"]))
        self.assertEqual(stronger["crossing_status"], "no_sign_crossing")
        self.assertEqual(stronger["gaussian_entropy_maximizing_fraction"], 0)
    def test_orientation_model_means_match_independent_sampling(self) -> None:
        from engine.copula import score
        first, second = np.array([[1, 0.4], [0.4, 1]]), np.array([[1, -0.7], [-0.7, 1]])
        point = orientation(first, second, [0.3])["points"][0]
        rng = np.random.default_rng(20261011)
        count = 30000
        means = []
        for reference, key, variance in ((first, "mean_genuine", "genuine_variance"), (second, "mean_fake", "fake_variance")):
            samples = rng.multivariate_normal(np.zeros(2), reference, count)
            observed = float(score(samples, np.asarray(point["reference"])).mean())
            self.assertLess(abs(observed - point[key]), 5 * np.sqrt(point[variance] / count))
            means.append(observed)
        self.assertLess(abs(means[0] - means[1] - point["mean_gap"]), 5 * np.sqrt((point["genuine_variance"] + point["fake_variance"]) / count))
    def test_orientation_rejects_invalid_models_and_fractions(self) -> None:
        valid = np.array([[1, 0.5], [0.5, 1]])
        for other in (np.eye(3), [[1, 1], [1, 1]], [[2, 0], [0, 1]], [[1, np.nan], [np.nan, 1]], [[1, 0.2], [0.4, 1]], np.eye(2, dtype=bool)):
            with self.subTest(other=other), self.assertRaises(ValueError):
                orientation(valid, other)
        for fractions in ([], [True], [np.nan], [-0.1], [1.1], [[0.2]]):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                orientation(valid, valid, fractions)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
