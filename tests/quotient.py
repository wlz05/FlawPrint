# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from engine.dependence import attenuation
from engine.quotient import Panel, audit, correlations, pearson, separation
from engine.storage import read_json, write_json

class QuotientTests(unittest.TestCase):

    def setUp(self) -> None:
        self.reference = np.array([[1, 0.35, 0.2, 0.15], [0.35, 1, 0.25, 0.18], [0.2, 0.25, 1, 0.3], [0.15, 0.18, 0.3, 1]])
        self.coefficients = np.array([0.7, 0.6, 0.8, 0.9])
        self.observed = attenuation(self.reference, self.coefficients)
        self.panel = Panel.fit(self.reference)

    def document(self) -> dict:
        return {"representation": "pearson", "observation_unit": "independent trace vector", "columns": ["noise", "demosaicing", "quantization", "resampling"], "reference": {"matrix": self.reference.tolist(), "relative_error": 0, "uncertainty_source": "exact synthetic population"}, "queries": [{"id": "attenuated", "matrix": self.observed.tolist(), "relative_error": 0, "uncertainty_source": "exact synthetic population"}]}

    def test_exact_independent_noise_is_eliminated(self) -> None:
        result = self.panel.compare(self.observed, 0, 0)
        self.assertLess(result["quotient_distance"], 1e-12)
        self.assertLess(result["attenuation_distance"], 1e-12)
        self.assertFalse(result["attenuation_rejected"])
        np.testing.assert_allclose(result["fitted_coefficients"], self.coefficients, atol=1e-12)
        self.assertGreater(np.linalg.norm(self.reference - self.observed), 0.1)

    def test_four_variable_ratio_survives_attenuation(self) -> None:
        ratio = lambda matrix: matrix[0, 1] * matrix[2, 3] / (matrix[0, 2] * matrix[1, 3])
        self.assertAlmostEqual(ratio(self.reference), ratio(self.observed), places=12)

    def test_single_edge_change_is_not_node_attenuation(self) -> None:
        altered = self.observed.copy()
        altered[0, 1] *= 0.6
        altered[1, 0] = altered[0, 1]
        result = self.panel.compare(altered, 0.01, 0.01)
        self.assertGreater(result["quotient_distance"], 0.2)
        self.assertTrue(result["attenuation_rejected"])
        self.assertEqual(len(result["edges"]), 6)

    def test_quotient_equality_does_not_prove_attenuation(self) -> None:
        base = attenuation(self.reference, np.full(4, 0.6))
        result = Panel.fit(base).compare(self.reference, 0, 0)
        self.assertLess(result["quotient_distance"], 1e-12)
        self.assertGreater(result["attenuation_distance"], 1)
        self.assertTrue(result["attenuation_rejected"])

    def test_conditional_margin_separates_two_attenuated_prototypes(self) -> None:
        alternative = self.reference.copy()
        alternative[0, 1] *= 0.5
        alternative[1, 0] = alternative[0, 1]
        for matrix, choice in ((self.reference, "reference"), (alternative, "alternative")):
            result = separation(self.panel, alternative, attenuation(matrix, self.coefficients), 0.01, 0.01, 0.01)
            self.assertEqual(result["prototype_choice"], choice)
            self.assertTrue(result["margin_sufficient"])

    def test_zero_quotient_gap_cannot_identify_same_orbit_prototypes(self) -> None:
        result = separation(self.panel, self.observed, self.reference, 0, 0, 0)
        self.assertIsNone(result["prototype_choice"])
        self.assertFalse(result["margin_sufficient"])

    def test_margin_requires_every_bound_and_same_signs(self) -> None:
        with self.assertRaises(ValueError):
            separation(self.panel, self.observed, self.reference, 0.1, None, 0.1)
        reflection = np.diag([-1, 1, 1, 1])
        result = separation(self.panel, reflection @ self.reference @ reflection, self.reference, 0, 0, 0)
        self.assertEqual(result["status"], "unresolved")
        self.assertIsNone(result["margin_sufficient"])

    def test_optional_alternative_uses_the_reference_frozen_panel(self) -> None:
        document = self.document()
        alternative = self.reference.copy()
        alternative[0, 1] *= 0.5
        alternative[1, 0] = alternative[0, 1]
        document["alternative"] = dict(document["reference"], matrix=alternative.tolist())
        result = audit(document)
        self.assertEqual(result["queries"][0]["separation"]["prototype_choice"], "reference")
        self.assertTrue(result["queries"][0]["separation"]["margin_sufficient"])

    def test_same_orbit_classes_cannot_be_separated_by_quotient(self) -> None:
        left = self.panel.compare(self.reference)
        right = self.panel.compare(self.observed)
        self.assertLess(abs(left["quotient_distance"] - right["quotient_distance"]), 1e-12)
        self.assertEqual(right["status"], "geometry_only")
        self.assertIsNone(right["attenuation_rejected"])

    def test_sign_changes_are_retained_outside_log_magnitude(self) -> None:
        reflection = np.diag([-1, 1, 1, 1])
        result = self.panel.compare(reflection @ self.reference @ reflection, 0.1, 0.1)
        self.assertLess(result["quotient_distance"], 1e-12)
        self.assertEqual(len(result["sign_conflicts"]), 3)
        self.assertTrue(result["attenuation_rejected"])

    def test_complete_graph_dimension_obstruction(self) -> None:
        for dimension in range(2, 9):
            matrix = 0.2 * np.ones((dimension, dimension)) + 0.8 * np.eye(dimension)
            panel = Panel.fit(matrix)
            expected = 0 if dimension == 2 else dimension * (dimension - 3) // 2
            with self.subTest(dimension=dimension):
                self.assertEqual(len(panel.edges) - panel.rank, expected)
                np.testing.assert_allclose(panel.projector @ panel.incidence, 0, atol=1e-12)
                np.testing.assert_allclose(panel.projector @ panel.projector, panel.projector, atol=1e-12)

    def test_sparse_even_cycle_retains_one_relation(self) -> None:
        panel = Panel.fit(self.reference, [[0, 1], [0, 3], [1, 2], [2, 3]])
        self.assertEqual(panel.rank, 3)
        self.assertEqual(len(panel.edges) - panel.rank, 1)
        self.assertLess(panel.compare(self.observed)["quotient_distance"], 1e-12)

    def test_sparse_odd_cycle_has_no_magnitude_relation(self) -> None:
        panel = Panel.fit(self.reference, [[0, 1], [0, 2], [1, 2]])
        result = panel.compare(self.observed)
        self.assertEqual(result["invariant_dimensions"], 0)
        self.assertIn("No quotient", result["magnitude_information"])

    def test_disconnected_components_and_isolated_node(self) -> None:
        reference = 0.2 * np.ones((6, 6)) + 0.8 * np.eye(6)
        panel = Panel.fit(reference, [[0, 1], [0, 3], [1, 2], [2, 3], [4, 5]])
        self.assertEqual(panel.rank, 4)
        self.assertEqual(len(panel.edges) - panel.rank, 1)
        isolated = Panel.fit(reference, [[0, 1], [0, 2], [1, 2]])
        self.assertEqual(isolated.rank, 3)

    def test_weak_query_abstains_on_the_entire_frozen_panel(self) -> None:
        query = self.observed.copy()
        query[0, 1] = query[1, 0] = 0
        result = self.panel.compare(query, 0.01, 0.01)
        self.assertEqual(result["status"], "unresolved")
        self.assertIsNone(result["quotient_distance"])
        self.assertEqual(len(self.panel.edges), 6)

    def test_empty_panel_does_not_manufacture_evidence(self) -> None:
        panel = Panel.fit(np.eye(4))
        result = panel.compare(np.eye(4))
        self.assertEqual(result["panel_edges"], 0)
        self.assertEqual(result["status"], "unresolved")

    def test_error_radius_contains_the_true_distances(self) -> None:
        random = np.random.default_rng(20261011)
        truth = self.observed.copy()
        truth[0, 1] *= 0.6
        truth[1, 0] = truth[0, 1]
        exact = self.panel.compare(truth)
        for _ in range(30):
            noisy_reference, noisy_query = self.reference.copy(), truth.copy()
            for matrix in (noisy_reference, noisy_query):
                for left, right in self.panel.edges:
                    matrix[left, right] *= 1 + random.uniform(-0.04, 0.04)
                    matrix[right, left] = matrix[left, right]
            result = Panel.fit(noisy_reference, self.panel.edges).compare(noisy_query, 0.04, 0.04)
            for field in ("quotient", "attenuation"):
                lower, upper = result[field + "_interval"]
                self.assertLessEqual(lower, exact[field + "_distance"] + 1e-12)
                self.assertGreaterEqual(upper, exact[field + "_distance"] - 1e-12)

    def test_large_declared_error_cannot_certify_a_small_violation(self) -> None:
        query = self.observed.copy()
        query[0, 1] *= 0.95
        query[1, 0] = query[0, 1]
        result = self.panel.compare(query, 0.1, 0.1)
        self.assertEqual(result["status"], "not_rejected")
        self.assertFalse(result["exact_geometry_compatible"])

    def test_vector_error_bounds_keep_every_edge(self) -> None:
        result = self.panel.compare(self.observed, np.full(6, 0.02), np.full(6, 0.03))
        self.assertAlmostEqual(result["error_radius"], np.sqrt(6) * (-np.log(0.98) - np.log(0.97)))

    def test_error_contract_rejects_partial_or_invalid_bounds(self) -> None:
        for left, right in ((None, 0.1), (0.1, None), (True, 0.1), (-0.1, 0.1), (1, 0.1), (float("nan"), 0.1), ([0.1] * 5, 0.1), ([0.1] * 7, 0.1), ([False] * 6, 0.1)):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                self.panel.compare(self.observed, left, right)

    def test_panel_state_roundtrip_and_corruption(self) -> None:
        restored = Panel.load(json.loads(json.dumps(self.panel.document())))
        self.assertEqual(restored.compare(self.observed), self.panel.compare(self.observed))
        for key, value in (("rank", 2), ("invariant_dimensions", True), ("version", "unknown"), ("minimum_edge", False)):
            invalid = self.panel.document()
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                Panel.load(invalid)

    def test_reference_panel_arrays_are_readonly(self) -> None:
        for values in (self.panel.reference, self.panel.edges, self.panel.incidence, self.panel.projector):
            with self.assertRaises(ValueError):
                values.flat[0] = 0
        np.testing.assert_array_equal(self.reference, self.panel.reference)

    def test_invalid_correlation_matrices_are_rejected(self) -> None:
        for invalid in ([[1, 2], [2, 1]], [[1, 0.2], [0.3, 1]], [[0.9, 0.2], [0.2, 1]], [[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]], [[1, True], [True, 1]], [[1, "0.2"], ["0.2", 1]], [[1]], [[1, 0], [0]], [[1, float("inf")], [float("inf"), 1]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                correlations(invalid)

    def test_semidefinite_sample_correlation_is_supported(self) -> None:
        matrix = pearson([[index, 2 * index, 3 * index, 4 * index] for index in range(5)])
        result = Panel.fit(matrix).compare(matrix)
        self.assertLess(result["quotient_distance"], 1e-12)

    def test_sample_pearson_matches_numpy_without_shrinkage(self) -> None:
        random = np.random.default_rng(9)
        samples = random.multivariate_normal(np.zeros(4), self.reference, 500)
        np.testing.assert_allclose(pearson(samples), np.corrcoef(samples, rowvar=False), atol=1e-12)
        for invalid in ([[1, 2]] * 4, [[1, 2], [2, 4], [3, 6]], [[1, 2], [2, 3], [3, 4], [4, True]]):
            with self.assertRaises(ValueError):
                pearson(invalid)

    def test_invalid_edge_panels_are_rejected(self) -> None:
        for edges in ([[0, 0]], [[1, 0]], [[0, 4]], [[0.0, 1]], [[False, 1]], [[0, 1], [0, 1]], [[0, 2], [0, 1]], [[0]], "edges"):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                Panel.fit(self.reference, edges)
        with self.assertRaises(ValueError):
            Panel.fit(self.reference, [[0, 3]], 0.2)

    def test_measurement_contract_preserves_inputs_and_denominators(self) -> None:
        document = self.document()
        document["queries"].append({"id": "weak", "matrix": np.eye(4).tolist(), "relative_error": 0, "uncertainty_source": "exact synthetic population"})
        original = copy.deepcopy(document)
        result = audit(document)
        self.assertEqual(document, original)
        self.assertEqual(result["coverage"], {"input": 2, "measurable": 1, "conditionally_bounded": 1})

    def test_measurement_contract_rejects_units_columns_and_labels(self) -> None:
        for mutation in (lambda item: item.update(representation="rank_normal"), lambda item: item["queries"][0].update(observation_unit="image instead of patch"), lambda item: item["queries"][0].update(columns=list(reversed(item["columns"]))), lambda item: item["reference"].update(label=0), lambda item: item["queries"][0].update(label=1), lambda item: item["reference"].pop("uncertainty_source"), lambda item: item["queries"].append(copy.deepcopy(item["queries"][0])), lambda item: item.update(columns=["a"] * 4), lambda item: item["queries"][0].update(id=1)):
            invalid = self.document()
            mutation(invalid)
            with self.assertRaises(ValueError):
                audit(invalid)

    def test_raw_sample_adapter_does_not_invent_uncertainty(self) -> None:
        random = np.random.default_rng(33)
        document = self.document()
        document["reference"] = {"samples": random.multivariate_normal(np.zeros(4), self.reference, 2000).tolist()}
        document["queries"] = [{"id": "sample", "samples": random.multivariate_normal(np.zeros(4), self.observed, 2000).tolist()}]
        result = audit(document)
        self.assertEqual(result["queries"][0]["status"], "geometry_only")
        self.assertEqual(result["reference_measurement"]["samples"], 2000)

    def test_correlated_platform_noise_is_not_silently_removed(self) -> None:
        covariance = self.reference + np.outer([0.4, -0.35, 0.2, -0.25], [0.4, -0.35, 0.2, -0.25])
        scales = np.sqrt(np.diag(covariance))
        query = covariance / np.outer(scales, scales)
        result = self.panel.compare(query, 0, 0)
        self.assertTrue(result["attenuation_rejected"])

    def test_cli_records_provenance_and_protects_inputs(self) -> None:
        Path("./tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="./tmp") as temporary:
            root = Path(temporary).relative_to(Path.cwd())
            source, output = str(root / "input.json"), str(root / "output.json")
            write_json(source, self.document())
            original = Path(source).read_bytes()
            arguments = [sys.executable, "-B", "research.py", "quotient", source, "--scope", "synthetic", "--dataset", "exact-control", "--output"]
            successful = subprocess.run(arguments + [output], capture_output=True, text=True)
            self.assertEqual(successful.returncode, 0, successful.stderr)
            result = read_json(output)
            self.assertEqual(len(result["source_sha256"]), 64)
            self.assertEqual(result["inputs"][0]["path"], source)
            rejected = subprocess.run(arguments + [source], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(Path(source).read_bytes(), original)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
