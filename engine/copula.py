# -*- coding: utf-8 -*-
# Copula: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtri
from scipy.linalg import cho_factor, cho_solve
def matrix(values: np.ndarray, minimum_rows: int = 2) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or len(result) < minimum_rows or result.shape[1] < 1 or not np.isfinite(result).all():
        raise ValueError("Expected a finite nonempty feature matrix")
    return result
def correlation(values: np.ndarray, shrinkage: float = 0.1, eigen_floor: float = 1e-6) -> np.ndarray:
    values = matrix(values)
    if not 0 <= shrinkage <= 1 or not np.isfinite(eigen_floor) or eigen_floor <= 0:
        raise ValueError("Invalid correlation regularization")
    centered = values - values.mean(axis=0)
    # Shrink unstable correlations before spectral repair.
    covariance = centered.T @ centered / max(len(values) - 1, 1)
    scales = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    result = covariance / np.outer(scales, scales)
    result = (1 - shrinkage) * result + shrinkage * np.eye(result.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh((result + result.T) / 2)
    result = (eigenvectors * np.maximum(eigenvalues, eigen_floor)) @ eigenvectors.T
    scales = np.sqrt(np.diag(result))
    result /= np.outer(scales, scales)
    np.fill_diagonal(result, 1.0)
    return result
def positive_matrix(values: np.ndarray, unit_diagonal: bool = False) -> np.ndarray:
    result = matrix(values, 1)
    if result.shape[0] != result.shape[1] or not np.allclose(result, result.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Reference matrices must be square and symmetric")
    if unit_diagonal and not np.allclose(np.diag(result), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Copula references must have a unit diagonal")
    try:
        cho_factor(result, lower=True)
    except np.linalg.LinAlgError as error:
        raise ValueError("Reference matrices must be positive definite") from error
    return result
def partial_correlation(covariance: np.ndarray) -> np.ndarray:
    covariance = positive_matrix(covariance)
    precision = cho_solve(cho_factor(covariance, lower=True), np.eye(len(covariance)))
    if np.any(np.diag(precision) <= 0):
        raise ValueError("Precision diagonal must be positive")
    partial = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    np.fill_diagonal(partial, 1.0)
    return partial
def score(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = matrix(values, 1)
    reference = positive_matrix(reference, True)
    sign, logdet = np.linalg.slogdet(reference)
    if sign <= 0 or values.shape[1] != len(reference):
        raise ValueError("Invalid copula reference")
    solved = cho_solve(cho_factor(reference, lower=True), values.T).T
    return np.sum(values * (values - solved), axis=1) - logdet
def moments(reference: np.ndarray) -> dict:
    reference = positive_matrix(reference, True)
    dimension = len(reference)
    precision = cho_solve(cho_factor(reference, lower=True), np.eye(len(reference)))
    sign, logdet = np.linalg.slogdet(reference)
    if sign <= 0:
        raise ValueError("Reference must be positive definite")
    operator = np.eye(dimension) - precision
    return {"real_mean": float(-logdet), "real_variance": float(max(0, 2 * (np.trace(reference @ reference) - dimension))), "independent_mean": float(dimension - np.trace(precision) - logdet), "independent_variance": float(max(0, 2 * np.trace(operator @ operator))), "mean_gap": float(np.trace(precision) - dimension), "dimension": dimension, "logdet": float(logdet)}
def separation(reference: np.ndarray, kappa: float = 2.0) -> dict:
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("Kappa must be positive")
    result = moments(reference)
    genuine_lower = result["real_mean"] - kappa * np.sqrt(result["real_variance"])
    independent_upper = result["independent_mean"] + kappa * np.sqrt(result["independent_variance"])
    separated = genuine_lower > independent_upper
    result.update(genuine_lower=float(genuine_lower), independent_upper=float(independent_upper), moment_intervals_separate=bool(separated), separation_midpoint=float((genuine_lower + independent_upper) / 2) if separated else None, threshold=0.0, threshold_policy="Independence zero; moment bounds are diagnostics", cantelli_bound=float(1 / (1 + kappa ** 2)), bound_scope="Known Gaussian reference and exact PIT only")
    return result
@dataclass
class Marginals:
    sorted_values: np.ndarray
    def __post_init__(self) -> None:
        self.sorted_values = matrix(self.sorted_values)
        if np.any(np.diff(self.sorted_values, axis=0) < 0):
            raise ValueError("Marginal reference columns must be sorted")
    def transform(self, values: np.ndarray) -> np.ndarray:
        values = matrix(values, 1)
        if values.shape[1] != self.sorted_values.shape[1]:
            raise ValueError("Marginal dimensions do not match")
        count = len(self.sorted_values)
        uniform = np.empty_like(values)
        for index in range(values.shape[1]):
            # Use midpoint ranks for tied observations.
            reference = self.sorted_values[:, index]
            left = np.searchsorted(reference, values[:, index], side="left")
            right = np.searchsorted(reference, values[:, index], side="right")
            uniform[:, index] = (left + right) / (2 * count)
        return ndtri(np.clip(uniform, 0.5 / count, 1 - 0.5 / count))
    @classmethod
    def fit(cls, values: np.ndarray) -> Marginals:
        return cls(np.sort(matrix(values), axis=0))
def rank_normal(values: np.ndarray) -> np.ndarray:
    return Marginals.fit(values).transform(values)
def decompose(z: np.ndarray, reference: np.ndarray, names: list[str], top: int = 8) -> dict:
    reference = positive_matrix(reference, True)
    z = np.asarray(z, dtype=float)
    if z.shape != (len(reference),) or len(names) != len(z) or not np.isfinite(z).all() or not isinstance(top, int) or top < 1:
        raise ValueError("Explanation dimensions and limits must be valid")
    operator = np.eye(len(reference)) - cho_solve(cho_factor(reference, lower=True), np.eye(len(reference)))
    diagonal = np.diag(operator) * np.square(z)
    pairs = [{"left": names[i], "right": names[j], "contribution": float(2 * operator[i, j] * z[i] * z[j])} for i in range(len(z)) for j in range(i + 1, len(z))]
    constant = float(-np.linalg.slogdet(reference)[1])
    total = float(diagonal.sum() + sum(pair["contribution"] for pair in pairs) + constant)
    return {"constant": constant, "diagonal_sum": float(diagonal.sum()), "pair_sum": float(sum(pair["contribution"] for pair in pairs)), "reconstructed_score": total, "most_negative_pairs": sorted(pairs, key=lambda pair: pair["contribution"])[:top]}
