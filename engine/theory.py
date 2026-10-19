# -*- coding: utf-8 -*-
# Theory: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from engine.copula import score, moments, partial_correlation, rank_normal, decompose
from engine.attacks import sinkhorn, exact_transport, reachable, quantile_match, independent_shuffle, marginal_report
from engine.signs import poisson_binomial
from engine.conformal import SplitConformal
def verify(seed: int = 42, samples: int = 100000, repetitions: int = 40) -> dict:
    if samples < 1000 or repetitions < 5:
        raise ValueError("Theory checks require sufficient Monte Carlo samples")
    rng = np.random.default_rng(seed)
    dimension = 4
    reference = np.full((dimension, dimension), 0.3)
    np.fill_diagonal(reference, 1.0)
    between = -0.5 * np.sqrt(1.4 ** 2 - np.linalg.det(reference) / 0.6 ** 2)
    alternative = np.array([[1, 0.4, between, between], [0.4, 1, between, between], [between, between, 1, 0.4], [between, between, 0.4, 1]])
    own_means = [-float(np.linalg.slogdet(matrix)[1]) for matrix in (reference, alternative)]
    fixed_mean = -float(np.linalg.slogdet(alternative)[1]) - float(np.trace((np.linalg.inv(alternative) - np.eye(dimension)) @ reference))
    sign_difference = int(np.count_nonzero(np.triu(np.sign(partial_correlation(reference)) != np.sign(partial_correlation(alternative)), 1)))
    scalar_counterexample = {"equal_own_reference_means": own_means, "same_distribution_scored_with_second_reference": fixed_mean, "fixed_reference_mean_change": fixed_mean - own_means[0], "different_partial_sign_pairs": sign_difference, "between_cluster_correlation": float(between), "scope": "Equal self-reference KL summaries do not imply identical samplewise quadratic scores"}
    genuine = rng.multivariate_normal(np.zeros(dimension), reference, samples)
    generated = rng.normal(size=(samples, dimension))
    closed = moments(reference)
    precision = np.linalg.inv(reference)
    logdet = np.linalg.slogdet(reference)[1]
    direct_log_density = -0.5 * (np.einsum("ni,ij,nj->n", genuine, precision - np.eye(dimension), genuine) + logdet)
    scored = score(genuine, reference)
    density_error = float(np.max(np.abs(scored - 2 * direct_log_density)))
    decomposition_error = abs(decompose(genuine[0], reference, [f"trace_{i}" for i in range(dimension)])["reconstructed_score"] - scored[0])
    scale = np.diag([0.1, 2.0, 5.0, 0.7])
    partial_error = float(np.max(np.abs(partial_correlation(reference) - partial_correlation(scale @ reference @ scale))))
    ranks = rank_normal(genuine[:2000])
    transformed = rank_normal(np.exp(np.clip(genuine[:2000], -10, 10)))
    rank_error = float(np.max(np.abs(ranks - transformed)))
    reflection = np.diag([-1, 1, 1, 1])
    reflection_error = float(np.max(np.abs(score(genuine, reference) - score(genuine @ reflection, reflection @ reference @ reflection))))
    observed_moments = {}
    for name, values, prefix in (("genuine", genuine, "real"), ("independent", generated, "independent")):
        measurements = score(values, reference)
        observed_moments[name] = {"mean": float(measurements.mean()), "expected_mean": closed[f"{prefix}_mean"], "variance": float(measurements.var()), "expected_variance": closed[f"{prefix}_variance"], "mean_standard_error": float(np.sqrt(closed[f"{prefix}_variance"] / samples))}
    first = np.array([0.2, 0.3, 0.5])
    second = np.array([0.1, 0.2, 0.3, 0.4])
    cost = (np.arange(3)[:, None] / 2 - np.arange(4)[None, :] / 3) ** 2
    entropic = sinkhorn(first, second, cost)
    exact = exact_transport(first, second, cost)
    plan = reachable(entropic["plan"], first, second)
    probabilities = np.array([0.1, 0.2, 0.4, 0.6, 0.8])
    distribution = poisson_binomial(probabilities)
    support = np.arange(len(distribution))
    sign_mean_error = float(abs(support @ distribution - probabilities.sum()))
    sign_variance_error = float(abs((support ** 2) @ distribution - (support @ distribution) ** 2 - np.sum(probabilities * (1 - probabilities))))
    labels = (generated[:, 0] + generated[:, 1] > 0).astype(int)
    fixed_scores = generated[:, 0] - 0.2 * generated[:, 2]
    auc_sum = float(roc_auc_score(labels, fixed_scores) + roc_auc_score(1 - labels, fixed_scores))
    matched = quantile_match(generated[:2000], genuine[:2000])
    mma = marginal_report(matched, genuine[:2000], [f"trace_{index}" for index in range(dimension)])
    independent = independent_shuffle(genuine[:2000], seed)
    shuffled_marginal_error = float(np.max(np.abs(np.sort(independent, axis=0) - np.sort(genuine[:2000], axis=0))))
    coverages = []
    for _ in range(repetitions):
        features = rng.normal(size=2000)
        probabilities_true = norm.cdf(features)
        targets = (rng.random(2000) < probabilities_true).astype(int)
        calibration = SplitConformal.fit(probabilities_true[:500], targets[:500], 0.1)
        sets = calibration.prediction_sets(probabilities_true[500:])
        coverages.append(np.mean([label in value for label, value in zip(targets[500:], sets)]))
    errors = {"twice_log_copula_density": density_error, "quadratic_decomposition": float(decomposition_error), "positive_scale_partial_correlation": partial_error, "increasing_transform_rank": rank_error, "coordinate_reflection": reflection_error, "poisson_binomial_mean": sign_mean_error, "poisson_binomial_variance": sign_variance_error, "auc_complementarity": abs(auc_sum - 1), "shuffled_marginals": shuffled_marginal_error}
    if max(errors.values()) > 1e-9 or not plan["feasible"]:
        raise AssertionError("A deterministic theory identity failed")
    errors["equal_scalar_summary"] = abs(own_means[0] - own_means[1])
    if errors["equal_scalar_summary"] > 1e-10 or sign_difference != 4:
        raise AssertionError("The scalar-summary counterexample failed")
    return {"seed": seed, "samples": samples, "scalar_summary_counterexample": scalar_counterexample, "identity_errors": errors, "moments": observed_moments, "moment_formulas": closed, "mma": mma, "transport": {"sinkhorn_error": entropic["marginal_error"], "exact_error": exact["marginal_error"], "sinkhorn_cost": entropic["transport_cost"], "exact_cost": exact["transport_cost"]}, "conformal": {"repetitions": repetitions, "mean_coverage": float(np.mean(coverages)), "coverage_standard_error": float(np.std(coverages, ddof=1) / np.sqrt(repetitions)), "nominal": 0.9}, "scope": "Algebraic identities and synthetic distributions only"}
