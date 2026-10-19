# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import numpy as np
from scipy.stats import rankdata, ks_2samp
from scipy.special import logsumexp
from scipy.optimize import linprog

def joint_replay(bundle: dict, genuine: list[int], generated: list[int]) -> dict:
    from engine.storage import subset, validate_bundle
    validate_bundle(bundle)
    if not genuine or len(genuine) != len(generated) or set(genuine) & set(generated):
        raise ValueError("Joint replay requires balanced disjoint real and fake samples")
    for selected, label in ((genuine, 0), (generated, 1)):
        subset(bundle, selected)
        if any(bundle["rows"][index].get("label") != label for index in selected):
            raise ValueError("Joint replay sample roles must match observed labels")
    result = subset(bundle, np.arange(len(bundle["rows"])))
    for key in ("values", "observed", "patches"):
        result[key][generated] = bundle[key][genuine]
    for source, target in zip(genuine, generated):
        result["rows"][target] = dict(result["rows"][target], attack="feature_joint_replay", oracle_reference_id=bundle["rows"][source]["id"], patch_signs_valid_after_attack=True, attack_scope="Oracle feature-and-patch replay; no corresponding physical image is generated")
    return result

def quantile_match(generated: np.ndarray, genuine: np.ndarray) -> np.ndarray:
    generated, genuine = np.asarray(generated, dtype=float), np.asarray(genuine, dtype=float)
    if generated.ndim != 2 or genuine.ndim != 2 or generated.shape[1] < 1 or generated.shape[1] != genuine.shape[1] or min(len(generated), len(genuine)) < 2 or not np.isfinite(generated).all() or not np.isfinite(genuine).all():
        raise ValueError("MMA requires compatible nonempty feature matrices")
    output = np.empty_like(generated)
    for index in range(generated.shape[1]):
        ranks = rankdata(generated[:, index], method="average")
        positions = (ranks - 0.5) / len(generated)
        reference_positions = (np.arange(len(genuine)) + 0.5) / len(genuine)
        output[:, index] = np.interp(positions, reference_positions, np.sort(genuine[:, index]))
    return output

def marginal_report(first: np.ndarray, second: np.ndarray, names: list[str]) -> dict:
    tests = []
    for index, name in enumerate(names):
        result = ks_2samp(first[:, index], second[:, index])
        tests.append({"feature": name, "ks_statistic": float(result.statistic), "p_value": float(result.pvalue)})
    return {"features": tests, "largest_ks": max(test["ks_statistic"] for test in tests), "all_p_above_005": all(test["p_value"] > 0.05 for test in tests), "interpretation": "Failure to reject is not proof of equal marginals"}

def independent_shuffle(values: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.asarray(values).copy()
    for index in range(result.shape[1]):
        result[:, index] = rng.permutation(result[:, index])
    return result

def transport_inputs(first: np.ndarray, second: np.ndarray, cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second, cost = np.asarray(first, dtype=float), np.asarray(second, dtype=float), np.asarray(cost, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or cost.shape != (len(first), len(second)) or not np.isfinite(cost).all():
        raise ValueError("Invalid transport dimensions")
    if not np.isfinite(first).all() or not np.isfinite(second).all() or np.any(first <= 0) or np.any(second <= 0):
        raise ValueError("Transport marginals must be strictly positive")
    if not np.isclose(first.sum(), 1) or not np.isclose(second.sum(), 1):
        raise ValueError("Transport marginals must sum to one")
    return first, second, cost

def sinkhorn(first: np.ndarray, second: np.ndarray, cost: np.ndarray, epsilon: float = 0.1, tolerance: float = 1e-8, iterations: int = 5000) -> dict:
    first, second, cost = transport_inputs(first, second, cost)
    if not np.isfinite((epsilon, tolerance)).all() or epsilon <= 0 or tolerance <= 0 or isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Invalid Sinkhorn settings")
    kernel, left, right = -cost / epsilon, np.zeros(len(first)), np.zeros(len(second))
    residual = float("inf")
    for step in range(iterations):
        # Update transport scalings in the log domain.
        left = np.log(first) - logsumexp(kernel + right[None, :], axis=1)
        right = np.log(second) - logsumexp(kernel + left[:, None], axis=0)
        if step % 10 == 0 or step == iterations - 1:
            plan = np.exp(kernel + left[:, None] + right[None, :])
            residual = max(np.max(np.abs(plan.sum(axis=1) - first)), np.max(np.abs(plan.sum(axis=0) - second)))
            if residual <= tolerance:
                break
    return {"plan": plan, "marginal_error": float(residual), "converged": bool(residual <= tolerance), "iterations": step + 1, "transport_cost": float(np.sum(plan * cost)), "epsilon": epsilon}

def exact_transport(first: np.ndarray, second: np.ndarray, cost: np.ndarray) -> dict:
    first, second, cost = transport_inputs(first, second, cost)
    rows, columns = len(first), len(second)
    constraints = np.zeros((rows + columns, rows * columns))
    for index in range(rows):
        constraints[index, index * columns:(index + 1) * columns] = 1
    for index in range(columns):
        constraints[rows + index, index::columns] = 1
    fitted = linprog(cost.ravel(), A_eq=constraints[:-1], b_eq=np.r_[first, second][:-1], bounds=(0, None), method="highs")
    if not fitted.success:
        raise RuntimeError("Exact transport failed to converge")
    plan = fitted.x.reshape(rows, columns)
    return {"plan": plan, "transport_cost": float(fitted.fun), "marginal_error": float(max(np.max(abs(plan.sum(axis=1) - first)), np.max(abs(plan.sum(axis=0) - second))))}

def reachable(plan: np.ndarray, first: np.ndarray, second: np.ndarray, tolerance: float = 1e-7) -> dict:
    residual = np.asarray(plan) - np.outer(first, second)
    row_error, column_error = float(np.max(np.abs(residual.sum(axis=1)))), float(np.max(np.abs(residual.sum(axis=0))))
    return {"zero_row_error": row_error, "zero_column_error": column_error, "nonnegative": bool(np.min(plan) >= -tolerance), "feasible": bool(max(row_error, column_error) <= tolerance and np.min(plan) >= -tolerance), "residual": residual}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
