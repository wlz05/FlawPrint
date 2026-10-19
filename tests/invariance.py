# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import unittest
import numpy as np
from engine.copula import Marginals, rank_normal
from engine.invariance import compare, coordinate_map, order_report, pairing, verify
from engine.attacks import quantile_match

class InvarianceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(20261009)
        self.reference = self.rng.normal(size=(160, 4))
        self.query = self.rng.normal(size=(40, 4))

    def test_common_increasing_maps_preserve_heldout_scores(self) -> None:
        for kind in ("affine", "asinh", "cubic"):
            with self.subTest(kind=kind):
                report = compare(self.reference, self.query, kind)
                self.assertTrue(report["invariance_verified"])
                self.assertEqual(report["maximum_query_pit_error"], 0)
                self.assertEqual(report["maximum_score_error"], 0)

    def test_query_only_transform_does_not_claim_invariance(self) -> None:
        correlated = self.reference.copy()
        correlated[:, 1] += 0.8 * correlated[:, 0]
        report = compare(correlated, self.query, "asinh")
        self.assertGreater(report["query_only_with_stale_reference"]["maximum_score_change"], 0.1)

    def test_existing_ties_are_preserved_by_a_strict_map(self) -> None:
        values = np.round(self.reference, decimals=1)
        transformed = coordinate_map(values)
        report = order_report(values, transformed)
        self.assertTrue(report["order_preserved"])
        self.assertTrue(any(report["ties_before"]))
        np.testing.assert_array_equal(rank_normal(values), rank_normal(transformed))

    def test_tie_creation_is_reported_as_rank_loss(self) -> None:
        report = order_report(self.reference, np.round(self.reference))
        self.assertFalse(report["order_preserved"])
        self.assertTrue(all(value > 0 for value in report["lost_unique_values"]))

    def test_target_ties_can_break_quantile_attack_invariance(self) -> None:
        target = np.round(self.reference)
        matched = quantile_match(self.query, target)
        self.assertFalse(order_report(self.query, matched)["order_preserved"])

    def test_decreasing_map_is_outside_increasing_invariance(self) -> None:
        self.assertFalse(order_report(self.reference, -self.reference)["order_preserved"])

    def test_combined_reference_order_is_checked(self) -> None:
        reference = np.array([[0, 0], [2, 2], [4, 4]], dtype=float)
        query = np.array([[1, 1], [3, 3]], dtype=float)
        original = Marginals.fit(reference).transform(query)
        transformed = Marginals.fit(coordinate_map(reference)).transform(coordinate_map(query))
        np.testing.assert_array_equal(original, transformed)

    def test_simulated_marginal_collapse_has_explicit_oracle_scope(self) -> None:
        report = verify(samples=128)
        attack = report["marginal_attack"]
        self.assertEqual(attack["maximum_sorted_marginal_error"], 0)
        self.assertEqual(attack["histogram_auc_after_refitting"], 0.5)
        self.assertEqual(attack["maximum_classwise_copula_score_error"], 0)
        self.assertGreater(attack["fixed_common_profile_maximum_score_change"], 0)
        self.assertIn("Oracle", attack["scope"])
        self.assertGreater(report["cross_coordinate_mixing"]["maximum_correlation_change"], 0)

    def test_invalid_simulation_and_map_inputs_fail(self) -> None:
        for settings in ({"samples": 0}, {"dimensions": 1}, {"seed": -1}):
            with self.assertRaises(ValueError):
                verify(**settings)
        with self.assertRaises(ValueError):
            coordinate_map(self.query, "unknown")
        with self.assertRaises(ValueError):
            compare(self.reference, self.query[:, :3])

    def test_pairing_changes_only_cross_terms_under_frozen_reference(self) -> None:
        reference = self.rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], 800)
        query = self.rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], 300)
        report = pairing(reference, query, repetitions=8)
        self.assertTrue(report["summary"]["all_identities_verified"])
        self.assertLess(report["summary"]["score_mean_change_average"], -1)
        self.assertTrue(report["common_row_reordering"]["invariance_verified"])
        for item in report["column_reorderings"]:
            self.assertEqual(item["maximum_sorted_marginal_error"], 0)
            self.assertEqual(item["maximum_sorted_pit_error"], 0)
            self.assertAlmostEqual(item["score_mean_change"], item["pair_mean_change"], places=10)

    def test_pairing_is_reproducible_without_mutating_inputs(self) -> None:
        reference, query = self.reference.copy(), self.query.copy()
        first = pairing(reference, query, seed=17, repetitions=3)
        second = pairing(reference, query, seed=17, repetitions=3)
        self.assertEqual(first, second)
        self.assertNotEqual(first["reference_sha256"], first["query_sha256"])
        np.testing.assert_array_equal(reference, self.reference)
        np.testing.assert_array_equal(query, self.query)

    def test_pairing_keeps_ties_and_constant_columns(self) -> None:
        reference, query = np.round(self.reference), np.round(self.query)
        reference[:, 0], query[:, 0] = 2, 2
        report = pairing(reference, query, repetitions=3)
        self.assertTrue(report["summary"]["all_identities_verified"])

    def test_identity_reference_has_no_pairing_evidence(self) -> None:
        report = pairing(self.reference, self.query, repetitions=3, shrinkage=1)
        self.assertEqual(report["original"]["score_mean"], 0)
        self.assertEqual(report["summary"]["score_mean_change_maximum"], 0)
        self.assertEqual(report["summary"]["score_mean_change_minimum"], 0)

    def test_constant_query_does_not_manufacture_separation(self) -> None:
        report = pairing(self.reference, np.ones((6, 4)), repetitions=2)
        self.assertEqual(report["summary"]["score_mean_change_average"], 0)
        self.assertTrue(report["summary"]["all_identities_verified"])

    def test_pairing_uses_reference_only_to_transform_queries(self) -> None:
        reference = self.reference.copy()
        query = self.query.copy() + 100
        report = pairing(reference, query, repetitions=2)
        self.assertEqual(report["summary"]["score_mean_change_average"], 0)
        self.assertIn("Reference only", report["reference_fit"])

    def test_single_pairing_repeat_has_no_sample_deviation(self) -> None:
        report = pairing(self.reference, self.query, repetitions=1)
        self.assertIsNone(report["summary"]["score_mean_change_standard_deviation"])
        self.assertTrue(report["summary"]["all_identities_verified"])

    def test_pairing_rejects_invalid_settings_and_missing_values(self) -> None:
        for options in ({"seed": True}, {"seed": -1}, {"repetitions": True}, {"repetitions": 0}, {"repetitions": 4097}, {"shrinkage": True}, {"shrinkage": float("nan")}, {"shrinkage": -0.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pairing(self.reference, self.query, **options)
        for query in (self.query[:1], self.query[:, :3], np.full_like(self.query, np.nan), np.full_like(self.query, np.inf)):
            with self.assertRaises(ValueError):
                pairing(self.reference, query)
        with self.assertRaises(ValueError):
            pairing(self.reference[:, :1], self.query[:, :1])

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
