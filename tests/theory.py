# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import unittest
import numpy as np
from scipy.special import ndtr
from scipy.stats import multivariate_normal, norm
from engine.copula import correlation, partial_correlation, score, moments, rank_normal, Marginals, decompose
from engine.signs import poisson_binomial, count_quantile
from engine.attacks import sinkhorn, exact_transport, reachable, quantile_match, independent_shuffle
from engine.dependence import student_score, tail_report, attenuation, availability, permutation_calibration
from engine.conformal import SplitConformal

class TheoryTests(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.reference = np.array([[1.0, 0.55, 0.2], [0.55, 1.0, 0.15], [0.2, 0.15, 1.0]])

    def test_density_ratio_orientation(self) -> None:
        z = self.rng.normal(size=(80, 3))
        direct = 2 * (multivariate_normal.logpdf(z, cov=self.reference) - norm.logpdf(z).sum(axis=1))
        np.testing.assert_allclose(score(z, self.reference), direct, atol=1e-12)

    def test_independent_reference_is_zero(self) -> None:
        z = self.rng.normal(size=(50, 3))
        np.testing.assert_allclose(score(z, np.eye(3)), 0, atol=1e-12)

    def test_closed_moments_under_both_models(self) -> None:
        expected = moments(self.reference)
        for covariance, prefix in ((self.reference, "real"), (np.eye(3), "independent")):
            values = self.rng.multivariate_normal(np.zeros(3), covariance, 120000)
            observations = score(values, self.reference)
            standard_error = np.sqrt(expected[f"{prefix}_variance"] / len(values))
            self.assertLess(abs(observations.mean() - expected[f"{prefix}_mean"]), 5 * standard_error)
            self.assertLess(abs(observations.var() / expected[f"{prefix}_variance"] - 1), 0.04)

    def test_rank_ties_and_outside_support(self) -> None:
        marginals = Marginals.fit(np.array([[1, 4], [1, 5], [2, 6], [3, 7]], dtype=float))
        z = marginals.transform(np.array([[1, 4], [-100, 100]], dtype=float))
        np.testing.assert_allclose(ndtr(z[0]), [0.25, 0.125])
        self.assertTrue(np.isfinite(z).all())

    def test_monotonic_marginal_invariance(self) -> None:
        values = self.rng.normal(size=(200, 3))
        np.testing.assert_allclose(rank_normal(values), rank_normal(np.exp(values)), atol=1e-12)

    def test_partial_positive_scale_invariance(self) -> None:
        scaling = np.diag([0.2, 5, 2])
        np.testing.assert_allclose(partial_correlation(self.reference), partial_correlation(scaling @ self.reference @ scaling), atol=1e-12)

    def test_coordinate_reflection(self) -> None:
        reflection = np.diag([-1, 1, 1])
        values = self.rng.normal(size=(80, 3))
        np.testing.assert_allclose(score(values, self.reference), score(values @ reflection, reflection @ self.reference @ reflection), atol=1e-12)

    def test_explanation_reconstructs_score(self) -> None:
        z = self.rng.normal(size=3)
        explained = decompose(z, self.reference, ["a", "b", "c"])
        self.assertAlmostEqual(explained["reconstructed_score"], score(z[None], self.reference)[0], places=12)

    def test_singular_sample_estimate_is_regularized(self) -> None:
        values = np.tile(np.arange(8)[:, None], (1, 20))
        reference = correlation(values, 0.1)
        self.assertGreater(np.linalg.eigvalsh(reference).min(), 0)
        np.testing.assert_allclose(np.diag(reference), 1)

    def test_poisson_binomial_has_correct_moments(self) -> None:
        probabilities = np.array([0.1, 0.2, 0.7, 0.9])
        distribution = poisson_binomial(probabilities)
        support = np.arange(len(distribution))
        self.assertAlmostEqual(distribution.sum(), 1)
        self.assertAlmostEqual(support @ distribution, probabilities.sum())
        self.assertAlmostEqual((support ** 2) @ distribution - (support @ distribution) ** 2, np.sum(probabilities * (1 - probabilities)))
        self.assertGreaterEqual(count_quantile(probabilities), 0)

    def test_transport_feasibility_and_cost_order(self) -> None:
        first, second = np.array([0.3, 0.7]), np.array([0.2, 0.3, 0.5])
        cost = np.array([[0, 1, 2], [2, 1, 0]], dtype=float)
        entropic = sinkhorn(first, second, cost, epsilon=0.2)
        exact = exact_transport(first, second, cost)
        self.assertTrue(entropic["converged"])
        self.assertTrue(reachable(entropic["plan"], first, second)["feasible"])
        self.assertGreaterEqual(entropic["transport_cost"] + 1e-8, exact["transport_cost"])

    def test_zero_marginal_residual_does_not_allow_negative_plan(self) -> None:
        plan = np.array([[0.7, -0.2], [-0.2, 0.7]])
        result = reachable(plan, np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        self.assertFalse(result["feasible"])

    def test_mma_preserves_rank_order_and_matches_empirical_values(self) -> None:
        first, second = self.rng.normal(size=(100, 3)), self.rng.exponential(size=(100, 3))
        matched = quantile_match(first, second)
        np.testing.assert_allclose(np.sort(matched, axis=0), np.sort(second, axis=0), atol=1e-12)
        np.testing.assert_array_equal(np.argsort(matched, axis=0), np.argsort(first, axis=0))

    def test_shuffle_preserves_every_empirical_marginal(self) -> None:
        values = self.rng.normal(size=(100, 3))
        np.testing.assert_array_equal(np.sort(values, axis=0), np.sort(independent_shuffle(values), axis=0))

    def test_student_density_is_finite(self) -> None:
        uniform = self.rng.uniform(0.01, 0.99, size=(40, 3))
        self.assertTrue(np.isfinite(student_score(uniform, self.reference, 4)).all())

    def test_empirical_tail_count_null_and_small_sample_flag(self) -> None:
        uniform = ndtr(rank_normal(self.rng.normal(size=(100, 3))))
        result = tail_report(uniform, ["a", "b", "c"], 0.1)
        self.assertFalse(result["tail_sample_size_adequate"])
        self.assertTrue(all(0 <= pair["upper_bh_q"] <= 1 for pair in result["pairs"]))

    def test_noise_attenuation_approaches_identity(self) -> None:
        np.testing.assert_allclose(attenuation(self.reference, np.zeros(3)), np.eye(3))
        np.testing.assert_allclose(attenuation(self.reference, np.ones(3)), self.reference)
        self.assertLess(np.linalg.cond(attenuation(self.reference, np.full(3, 0.2))), np.linalg.cond(self.reference))

    def test_batch_availability_relaxes_screen(self) -> None:
        first = availability(self.reference, batch_size=1)
        second = availability(self.reference, batch_size=4)
        self.assertGreater(second["genuine_lower"], first["genuine_lower"])

    def test_permutation_statistic_uses_heldout_dependency(self) -> None:
        fitting = self.rng.multivariate_normal(np.zeros(3), self.reference, 300)
        heldout = self.rng.multivariate_normal(np.zeros(3), self.reference, 300)
        result = permutation_calibration(heldout, self.reference, repeats=99, reference_samples=len(fitting))
        self.assertLessEqual(result["dependence_p_value"], 0.05)
        self.assertEqual(result["reference_samples"], 300)
        with self.assertRaises(ValueError):
            permutation_calibration(heldout, self.reference, reference_samples=1)

    def test_small_calibration_uses_full_prediction_set(self) -> None:
        predictor = SplitConformal.fit(np.array([0.2, 0.8]), np.array([0, 1]), 0.01)
        self.assertEqual(predictor.prediction_sets(np.array([0.5])), [[0, 1]])

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
