# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import numpy as np
from engine.copula import correlation, partial_correlation, rank_normal
PATCH_COLUMNS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 19)
def bootstrap_weights(z: np.ndarray, reference: np.ndarray, repeats: int = 100, seed: int = 42, shrinkage: float = 0.1, tolerance: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if repeats < 1 or tolerance < 0:
        raise ValueError("Invalid sign bootstrap settings")
    partial = partial_correlation(reference)
    direction = np.sign(partial)
    agreement = np.zeros_like(partial)
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        selected = z[rng.integers(0, len(z), len(z))]
        estimate = partial_correlation(correlation(selected, shrinkage))
        agreement += np.sign(estimate) == direction
    stability = agreement / repeats
    weights = np.maximum(0, 2 * stability - 1) * np.abs(partial)
    weights[np.abs(partial) < tolerance] = 0
    np.fill_diagonal(weights, 0)
    return direction, weights, stability
def local_signs(patches: np.ndarray, columns: list[int], shrinkage: float = 0.35, tolerance: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    dimension = len(columns)
    directions = np.zeros((dimension, dimension))
    usable = np.zeros((dimension, dimension), dtype=bool)
    eligible = [index for index, column in enumerate(columns) if column in PATCH_COLUMNS and np.ptp(patches[:, column]) > 1e-10]
    # Exclude features that cannot vary across image tiles.
    if len(eligible) < 2 or len(patches) < 4:
        return directions, usable
    raw = patches[:, np.asarray(columns)[eligible]]
    partial = partial_correlation(correlation(rank_normal(raw), shrinkage))
    valid = np.abs(partial) >= tolerance
    np.fill_diagonal(valid, False)
    directions[np.ix_(eligible, eligible)] = np.sign(partial)
    usable[np.ix_(eligible, eligible)] = valid
    return directions, usable
def conflicts(patches: np.ndarray, columns: list[int], directions: np.ndarray, weights: np.ndarray, names: list[str], shrinkage: float = 0.35, top: int = 8) -> dict:
    local, usable = local_signs(patches, columns, shrinkage)
    available = np.triu(usable, 1) & (weights > 0)
    mismatch = available & (local != directions)
    total_weight = float(weights[available].sum())
    conflict_weight = float(weights[mismatch].sum())
    pairs = [{"left": names[i], "right": names[j], "reference_sign": int(directions[i, j]), "local_sign": int(local[i, j]), "weight": float(weights[i, j])} for i, j in zip(*np.where(mismatch))]
    return {"literal_score": -conflict_weight, "conflict_ratio": conflict_weight / total_weight if total_weight > 0 else 0.0, "available_weight": total_weight, "available_pairs": int(available.sum()), "conflict_pairs": int(mismatch.sum()), "conflicts": sorted(pairs, key=lambda pair: pair["weight"], reverse=True)[:top]}
def poisson_binomial(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must lie in the unit interval")
    mass = np.array([1.0])
    for probability in probabilities:
        mass = np.convolve(mass, [1 - probability, probability])
    return mass
def count_quantile(probabilities: np.ndarray, level: float = 0.95) -> int:
    if not 0 < level < 1:
        raise ValueError("Quantile level must be between zero and one")
    return int(np.searchsorted(np.cumsum(poisson_binomial(probabilities)), level, side="left"))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
