# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import numpy as np
from scipy.special import gammaln, ndtr
from scipy.stats import t, hypergeom
from scipy.linalg import cho_factor, cho_solve
from engine.copula import rank_normal, correlation, score, moments, matrix, positive_matrix
from engine.attacks import independent_shuffle
def student_score(uniform: np.ndarray, reference: np.ndarray, degrees: float = 4.0) -> np.ndarray:
    uniform = matrix(uniform, 1)
    reference = positive_matrix(reference, True)
    if uniform.shape[1] != len(reference) or not np.isfinite(degrees) or degrees <= 2 or np.any((uniform <= 0) | (uniform >= 1)):
        raise ValueError("Invalid Student copula parameters")
    z = t.ppf(uniform, degrees)
    factor = cho_factor(reference, lower=True)
    precision_z = cho_solve(factor, z.T).T
    quadratic = np.sum(z * precision_z, axis=1)
    dimension = len(reference)
    logdet = 2 * np.log(np.diag(factor[0])).sum()
    joint = gammaln((degrees + dimension) / 2) - gammaln(degrees / 2) - 0.5 * (dimension * np.log(degrees * np.pi) + logdet) - (degrees + dimension) / 2 * np.log1p(quadratic / degrees)
    marginal = np.sum(t.logpdf(z, degrees), axis=1)
    return 2 * (joint - marginal)
def student_selection(uniform: np.ndarray, references: list[float] | None = None, shrinkage: float = 0.1) -> dict:
    candidates = references or [3.0, 4.0, 6.0, 10.0, 30.0]
    results = []
    for degrees in candidates:
        z = t.ppf(uniform, degrees)
        reference = correlation(z, shrinkage)
        likelihood = float(student_score(uniform, reference, degrees).mean() / 2)
        results.append({"degrees": degrees, "mean_log_density": likelihood, "reference": reference})
    best = max(results, key=lambda result: result["mean_log_density"])
    return {"selected": best, "candidates": [{key: value for key, value in item.items() if key != "reference"} for item in results], "scope": "Exploratory profile likelihood; use held-out data to compare copula families"}
def permutation_calibration(heldout_z: np.ndarray, reference: np.ndarray, repeats: int = 500, alpha: float = 0.05, seed: int = 42, reference_samples: int | None = None) -> dict:
    heldout_z = matrix(heldout_z, 4)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 10 or not 0 < alpha < 1 or (reference_samples is not None and (isinstance(reference_samples, bool) or not isinstance(reference_samples, int) or reference_samples < 2)):
        raise ValueError("Invalid permutation calibration settings")
    observed = float(score(heldout_z, reference).mean())
    null_statistics = []
    for index in range(repeats):
        null_statistics.append(float(score(independent_shuffle(heldout_z, seed + index), reference).mean()))
    null = np.asarray(null_statistics)
    p_value = float((1 + np.sum(null >= observed)) / (repeats + 1))
    quantile = float(np.quantile(null, 1 - alpha, method="higher"))
    return {"observed_mean_score": observed, "independence_upper_quantile": quantile, "dependence_p_value": p_value, "dependence_detected": p_value <= alpha, "reference_samples": reference_samples, "heldout_samples": len(heldout_z), "repeats": repeats, "alpha": alpha, "null_mean": float(null.mean()), "null_standard_deviation": float(null.std(ddof=1)), "scope": "Fixed reference with fitting disjointness checked by caller identifiers; tests independence rather than authenticity"}
def false_discovery(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Multiple-testing correction requires valid p-values")
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1.0)
    return result
def tail_report(uniform: np.ndarray, names: list[str], threshold: float = 0.1) -> dict:
    uniform = matrix(uniform, 4)
    if uniform.shape[1] != len(names) or len(names) < 2 or len(set(names)) != len(names) or not 0 < threshold < 0.5 or np.any((uniform < 0) | (uniform > 1)):
        raise ValueError("Invalid tail dependence inputs")
    count, pairs = len(uniform), []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            upper_i, upper_j = uniform[:, i] > 1 - threshold, uniform[:, j] > 1 - threshold
            lower_i, lower_j = uniform[:, i] < threshold, uniform[:, j] < threshold
            upper, lower = int(np.sum(upper_i & upper_j)), int(np.sum(lower_i & lower_j))
            # Condition on empirical tail counts under permutation.
            upper_p = float(hypergeom.sf(upper - 1, count, int(upper_i.sum()), int(upper_j.sum())))
            lower_p = float(hypergeom.sf(lower - 1, count, int(lower_i.sum()), int(lower_j.sum())))
            pairs.append({"left": names[i], "right": names[j], "upper_count": upper, "lower_count": lower, "upper_lambda": upper / (count * threshold), "lower_lambda": lower / (count * threshold), "asymmetry": (upper - lower) / (count * threshold), "upper_p_value": upper_p, "lower_p_value": lower_p})
    adjusted = false_discovery(np.asarray([value for pair in pairs for value in (pair["upper_p_value"], pair["lower_p_value"])]))
    for index, pair in enumerate(pairs):
        pair["upper_bh_q"] = float(adjusted[2 * index])
        pair["lower_bh_q"] = float(adjusted[2 * index + 1])
    denominator = count * threshold ** 2 * (1 - threshold ** 2)
    statistic = sum((pair["upper_count"] - count * threshold ** 2) ** 2 / denominator for pair in pairs)
    return {"samples": count, "threshold": threshold, "iid_uniform_expected_count": count * threshold ** 2, "tail_sample_size_adequate": count * threshold ** 2 >= 50, "upper_statistic": float(statistic), "pairs": pairs, "scope": "Finite-threshold estimates; conditional permutation count null is hypergeometric for empirical ranks, not iid binomial"}
def attenuation(reference: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    reference = positive_matrix(reference, True)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (len(reference),) or not np.isfinite(coefficients).all() or np.any((coefficients < 0) | (coefficients > 1)):
        raise ValueError("Attenuation coefficients must lie in the unit interval")
    return reference * np.outer(coefficients, coefficients) + np.diag(1 - coefficients ** 2)
def availability(reference: np.ndarray, alpha: float = 0.05, batch_size: int = 1, kappa: float | None = None) -> dict:
    if not 0 < alpha < 1 or isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("Invalid availability settings")
    from scipy.stats import norm
    multiplier = float(norm.ppf(1 - alpha)) if kappa is None else float(kappa)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Kappa must be positive")
    result = moments(reference)
    deviation = np.sqrt(result["real_variance"] / batch_size)
    gap = result["mean_gap"]
    result.update(alpha=alpha, kappa=multiplier, batch_size=batch_size, genuine_lower=float(result["real_mean"] - multiplier * deviation), power_screen=bool(gap > multiplier * deviation), gap_to_real_sd=float(gap / max(np.sqrt(result["real_variance"]), 1e-12)), minimum_batch_size=max(1, int(np.floor((multiplier * np.sqrt(result["real_variance"]) / max(gap, 1e-12)) ** 2)) + 1), scope="F16/F22 usability screen; normal quantiles do not imply distribution-free error control")
    return result
def platform_alignment(references: dict[str, np.ndarray]) -> dict:
    if not references:
        raise ValueError("Platform alignment requires reference matrices")
    directions, diagnostics = {}, {}
    dimensions = set()
    for domain, reference in references.items():
        reference = positive_matrix(reference, True)
        dimensions.add(len(reference))
        eigenvalues, eigenvectors = np.linalg.eigh(reference - np.eye(len(reference)))
        magnitudes = np.sort(np.abs(eigenvalues))
        peak = float(magnitudes[-1])
        gap = peak - float(magnitudes[-2]) if len(magnitudes) > 1 else peak
        identifiable = peak > 1e-10 and gap > 1e-8 * max(peak, 1.0)
        directions[domain] = eigenvectors[:, np.argmax(np.abs(eigenvalues))] if identifiable else None
        diagnostics[domain] = {"dominant_absolute_eigenvalue": peak, "absolute_spectral_gap": gap, "direction_identifiable": identifiable}
    if len(dimensions) != 1:
        raise ValueError("Platform references must use the same feature dimensions")
    pairs = [{"first": first, "second": second, "absolute_cosine": float(abs(directions[first] @ directions[second])) if directions[first] is not None and directions[second] is not None else None} for first in sorted(directions) for second in sorted(directions)]
    return {"pairs": pairs, "directions": directions, "diagnostics": diagnostics, "scope": "F23 diagnostic; missing or degenerate dominant directions have null alignment and do not identify platform causes"}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
