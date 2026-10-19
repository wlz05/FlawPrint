# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import numpy as np
from hashlib import sha256
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata
from engine.copula import matrix, Marginals, correlation, partial_correlation, rank_normal, score
from engine.attacks import quantile_match
from engine.metrics import binary
def pairing(reference: np.ndarray, query: np.ndarray, seed: int = 42, repetitions: int = 32, shrinkage: float = 0.1) -> dict:
    reference, query = matrix(reference), matrix(query)
    if reference.shape[1] != query.shape[1] or query.shape[1] < 2:
        raise ValueError("Pairing diagnostics require at least two common finite feature columns")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (seed, repetitions)) or seed < 0 or not 1 <= repetitions <= 4096:
        raise ValueError("Pairing diagnostics require a nonnegative seed and 1 to 4096 repetitions")
    if isinstance(shrinkage, bool) or not isinstance(shrinkage, (int, float, np.floating)) or not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("Pairing shrinkage must be a finite fraction")
    marginals = Marginals.fit(reference)
    fitted = correlation(marginals.transform(reference), shrinkage)
    precision = cho_solve(cho_factor(fitted, lower=True), np.eye(len(fitted)))
    operator = np.eye(len(fitted)) - precision
    diagonal = np.diag(operator)
    cross = operator - np.diag(diagonal)
    z = marginals.transform(query)
    original = score(z, fitted)
    diagonal_mean = float(np.mean(np.square(z) @ diagonal))
    second_moment = z.T @ z / len(z)
    pair_mean = float(np.sum(cross * second_moment))
    constant = float(-np.linalg.slogdet(fitted)[1])
    rng = np.random.default_rng(seed)
    row_order = rng.permutation(len(query))
    row_scores = score(marginals.transform(query[row_order]), fitted)
    row_error = float(np.max(np.abs(row_scores - original[row_order])))
    records = []
    for repetition in range(repetitions):
        # Rearrange query columns while keeping the fitted reference frozen.
        indices = np.column_stack([rng.permutation(len(query)) for _ in range(query.shape[1])])
        changed = np.take_along_axis(query, indices, axis=0)
        updated_z = marginals.transform(changed)
        updated = score(updated_z, fitted)
        updated_moment = updated_z.T @ updated_z / len(query)
        updated_diagonal = float(np.mean(np.square(updated_z) @ diagonal))
        updated_pair = float(np.sum(cross * updated_moment))
        mean_change = float(updated.mean() - original.mean())
        pair_change = updated_pair - pair_mean
        scale = max(1.0, float(np.max(np.abs(original))), float(np.max(np.abs(updated))))
        tolerance = 1e-10 * scale
        raw_error = float(np.max(np.abs(np.sort(changed, axis=0) - np.sort(query, axis=0))))
        pit_error = float(np.max(np.abs(np.sort(updated_z, axis=0) - np.sort(z, axis=0))))
        diagonal_error = abs(updated_diagonal - diagonal_mean)
        identity_error = abs(mean_change - pair_change)
        reconstruction_error = abs(float(updated.mean()) - (constant + updated_diagonal + updated_pair))
        records.append({"repetition": repetition, "permutation_sha256": sha256(indices.astype("<i8").tobytes()).hexdigest(), "maximum_sorted_marginal_error": raw_error, "maximum_sorted_pit_error": pit_error, "diagonal_mean_error": diagonal_error, "constant_change": 0.0, "pair_mean": updated_pair, "pair_mean_change": pair_change, "score_mean": float(updated.mean()), "score_mean_change": mean_change, "mean_identity_error": identity_error, "reconstruction_error": reconstruction_error, "maximum_second_moment_change": float(np.max(np.abs(updated_moment - second_moment))), "maximum_sorted_score_change": float(np.max(np.abs(np.sort(updated) - np.sort(original)))), "unchanged_row_fraction": float(np.mean(np.all(changed == query, axis=1))), "numerical_tolerance": tolerance, "identity_verified": raw_error == 0.0 and pit_error == 0.0 and max(diagonal_error, identity_error, reconstruction_error) <= tolerance})
    changes = np.array([item["score_mean_change"] for item in records])
    return {"version": "flawprint-pairing-v1", "seed": seed, "repetitions": repetitions, "reference_samples": len(reference), "query_samples": len(query), "dimensions": query.shape[1], "reference_sha256": sha256(reference.astype("<f8").tobytes()).hexdigest(), "query_sha256": sha256(query.astype("<f8").tobytes()).hexdigest(), "reference_fit": "Reference only; query values and labels do not fit marginals or correlation", "shrinkage": float(shrinkage), "original": {"score_mean": float(original.mean()), "diagonal_mean": diagonal_mean, "pair_mean": pair_mean, "constant": constant, "reconstruction_error": abs(float(original.mean()) - (constant + diagonal_mean + pair_mean))}, "common_row_reordering": {"maximum_aligned_score_error": row_error, "invariance_verified": row_error <= 1e-10 * max(1.0, float(np.max(np.abs(original))))}, "column_reorderings": records, "summary": {"all_identities_verified": all(item["identity_verified"] for item in records), "score_mean_change_average": float(changes.mean()), "score_mean_change_standard_deviation": float(changes.std(ddof=1)) if repetitions > 1 else None, "score_mean_change_minimum": float(changes.min()), "score_mean_change_maximum": float(changes.max())}, "scope": "Label-free finite complete-feature diagnostic; column permutations preserve empirical marginals and vary pairing, but do not imply exact finite-sample independence, physical image edits, news labels, or guaranteed score separation"}
def coordinate_map(values: np.ndarray, kind: str = "asinh") -> np.ndarray:
    values = matrix(values, 1)
    scales = np.linspace(0.7, 1.4, values.shape[1])
    offsets = np.linspace(-0.4, 0.4, values.shape[1])
    if kind == "affine":
        result = values * scales + offsets
    elif kind == "asinh":
        result = np.arcsinh(values * scales) + offsets
    elif kind == "cubic":
        result = scales * (values + 0.05 * values ** 3) + offsets
    else:
        raise ValueError("Unknown increasing coordinate transformation")
    if not np.isfinite(result).all():
        raise ValueError("Coordinate transformations must remain finite")
    return result
def order_report(before: np.ndarray, after: np.ndarray) -> dict:
    before, after = matrix(before), matrix(after)
    if before.shape != after.shape:
        raise ValueError("Order comparison requires matching dimensions")
    first = rankdata(before, method="average", axis=0)
    second = rankdata(after, method="average", axis=0)
    lost = [max(0, len(np.unique(before[:, index])) - len(np.unique(after[:, index]))) for index in range(before.shape[1])]
    maximum = float(np.max(np.abs(first - second)))
    return {"maximum_rank_displacement": maximum, "order_preserved": maximum == 0.0, "lost_unique_values": lost, "ties_before": [len(before) - len(np.unique(before[:, index])) for index in range(before.shape[1])], "ties_after": [len(after) - len(np.unique(after[:, index])) for index in range(after.shape[1])]}
def compare(reference: np.ndarray, query: np.ndarray, kind: str = "asinh", shrinkage: float = 0.1) -> dict:
    reference, query = matrix(reference), matrix(query, 1)
    if reference.shape[1] != query.shape[1]:
        raise ValueError("Invariance requires a common reference and query schema")
    changed_reference, changed_query = coordinate_map(reference, kind), coordinate_map(query, kind)
    first, second = Marginals.fit(reference), Marginals.fit(changed_reference)
    first_reference, second_reference = first.transform(reference), second.transform(changed_reference)
    first_query, second_query = first.transform(query), second.transform(changed_query)
    original, transformed = correlation(first_reference, shrinkage), correlation(second_reference, shrinkage)
    initial, updated = score(first_query, original), score(second_query, transformed)
    stale = score(first.transform(changed_query), original)
    error = float(np.max(np.abs(initial - updated)))
    combined = order_report(np.vstack((reference, query)), np.vstack((changed_reference, changed_query)))
    return {"transformation": kind, "reference_samples": len(reference), "query_samples": len(query), "dimensions": reference.shape[1], "order": combined, "maximum_reference_pit_error": float(np.max(np.abs(first_reference - second_reference))), "maximum_query_pit_error": float(np.max(np.abs(first_query - second_query))), "maximum_correlation_error": float(np.max(np.abs(original - transformed))), "maximum_partial_correlation_error": float(np.max(np.abs(partial_correlation(original) - partial_correlation(transformed)))), "maximum_score_error": error, "invariance_verified": combined["order_preserved"] and error <= 1e-10, "query_only_with_stale_reference": {"maximum_score_change": float(np.max(np.abs(initial - stale))), "changed_score_fraction": float(np.mean(np.abs(initial - stale) > 1e-10)), "scope": "Changing only queries is outside simultaneous reference and query invariance"}, "scope": "Midpoint empirical PIT with the same increasing map applied to reference and query; missing-value masks remain fixed"}
def histogram_risk(genuine: np.ndarray, generated: np.ndarray, bins: int = 32) -> np.ndarray:
    genuine, generated = matrix(genuine), matrix(generated)
    if genuine.shape[1] != generated.shape[1] or not isinstance(bins, int) or bins < 2:
        raise ValueError("Histogram diagnostics require compatible features and valid bins")
    values = np.vstack((genuine, generated))
    result = np.zeros(len(values))
    for index in range(values.shape[1]):
        edges = np.linspace(values[:, index].min(), values[:, index].max(), bins + 1)
        if edges[-1] == edges[0]:
            continue
        genuine_counts = np.histogram(genuine[:, index], bins=edges)[0] + 0.5
        generated_counts = np.histogram(generated[:, index], bins=edges)[0] + 0.5
        density_ratio = np.log(generated_counts / generated_counts.sum()) - np.log(genuine_counts / genuine_counts.sum())
        positions = np.clip(np.searchsorted(edges, values[:, index], side="right") - 1, 0, bins - 1)
        result += density_ratio[positions]
    return result
def verify(seed: int = 42, samples: int = 4000, dimensions: int = 4, repetitions: int = 32) -> dict:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (seed, samples, dimensions)) or seed < 0 or samples < 32 or dimensions < 2:
        raise ValueError("Invariance simulation requires a valid seed, sample count, and dimension")
    rng = np.random.default_rng(seed)
    def equicorrelation(rho: float) -> np.ndarray:
        return (1 - rho) * np.eye(dimensions) + rho * np.ones((dimensions, dimensions))
    genuine = rng.multivariate_normal(np.zeros(dimensions), equicorrelation(0.60), samples)
    generated = np.exp(0.45 * rng.multivariate_normal(np.zeros(dimensions), equicorrelation(0.10), samples))
    matched = quantile_match(generated, genuine)
    labels = np.repeat((0, 1), samples)
    reference = correlation(rank_normal(genuine), 0.1)
    before = -np.concatenate((score(rank_normal(genuine), reference), score(rank_normal(generated), reference)))
    after = -np.concatenate((score(rank_normal(genuine), reference), score(rank_normal(matched), reference)))
    initial_histogram, updated_histogram = histogram_risk(genuine, generated), histogram_risk(genuine, matched)
    mixed = genuine.copy()
    mixed[:, 1] += 0.8 * mixed[:, 0]
    quantized = np.round(generated, decimals=0)
    queries = rng.multivariate_normal(np.zeros(dimensions), equicorrelation(0.60), max(32, samples // 4))
    common = Marginals.fit(genuine)
    stale_before, stale_after = score(common.transform(generated), reference), score(common.transform(matched), reference)
    return {"version": "flawprint-invariance-v2", "seed": seed, "samples_per_class": samples, "dimensions": dimensions, "generative_settings": {"genuine_correlation": 0.60, "generated_correlation": 0.10, "genuine_marginal": "Standard normal", "generated_marginal": "Lognormal with log scale 0.45", "histogram_bins": 32, "histogram_pseudocount": 0.5, "reference_shrinkage": 0.1}, "fixed_reference_pairing": pairing(genuine, queries, seed, repetitions), "shared_coordinate_maps": [compare(genuine, queries, kind) for kind in ("affine", "asinh", "cubic")], "marginal_attack": {"order": order_report(generated, matched), "maximum_sorted_marginal_error": float(np.max(np.abs(np.sort(matched, axis=0) - np.sort(genuine, axis=0)))), "histogram_auc_before": binary(labels, initial_histogram)["auc"], "histogram_auc_after_refitting": binary(labels, updated_histogram)["auc"], "maximum_refitted_histogram_risk": float(np.max(np.abs(updated_histogram))), "classwise_copula_auc_before": binary(labels, before)["auc"], "classwise_copula_auc_after": binary(labels, after)["auc"], "maximum_classwise_copula_score_error": float(np.max(np.abs(before - after))), "fixed_common_profile_maximum_score_change": float(np.max(np.abs(stale_before - stale_after))), "scope": "Oracle classwise ranks and in-sample histograms demonstrate movable marginals; this is not held-out deployment performance"}, "tie_creation": dict(order_report(generated, quantized), scope="Rounding is many-to-one and can destroy ranks; existing ties are retained by midpoint PIT"), "cross_coordinate_mixing": {"order": order_report(genuine, mixed), "maximum_correlation_change": float(np.max(np.abs(correlation(rank_normal(mixed)) - reference))), "scope": "This example mixes coordinates and can change the copula; no claim is made that every non-coordinate map changes it"}, "limitations": ["Continuous copula orbit equivalence requires an explicitly defined invertible group and compatible marginal supports", "Empirical PIT is a step function and is not a strictly increasing bijection", "A Gaussian correlation matrix summarizes pairwise dependence and is not the full maximal invariant", "Class-specific attacks are not a common unlabeled domain transformation", "No real images, official baseline training, or SOTA validation are included"]}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
