# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import replace
import copy
import numpy as np
from engine.profile import DomainProfile, ProfileConfig
from engine.copula import Marginals, correlation, partial_correlation, separation
from engine.storage import validate_bundle
from engine.lineage import collect, merge, disjoint
from engine.results import prediction_rows
def pool(profiles: list[DomainProfile], domain: str = "pooled") -> DomainProfile:
    if not profiles:
        raise ValueError("Cold-start pooling requires historical profiles")
    for profile in profiles:
        profile.validate()
        if profile.quality.get("reference_selection"):
            raise ValueError("Selected references require an explicit joint selection protocol before pooling")
    identifiers = [identifier for profile in profiles for identifier in profile.identifiers + profile.quality.get("genuine_calibration_identifiers", [])]
    if len(set(identifiers)) != len(identifiers) or len({profile.domain for profile in profiles}) != len(profiles):
        raise ValueError("Historical profiles must have distinct domains and disjoint samples")
    histories = [merge(profile.history, profile.quality.get("genuine_calibration_history")) for profile in profiles]
    disjoint(*histories)
    common = sorted(set.intersection(*(set(profile.columns) for profile in profiles)))
    if len(common) < 2:
        raise ValueError("Historical profiles need two common features")
    mappings = [[profile.columns.index(column) for column in common] for profile in profiles]
    names = [profiles[0].names[index] for index in mappings[0]]
    if any([profile.names[index] for index in mapping] != names for profile, mapping in zip(profiles, mappings)):
        raise ValueError("Historical feature schemas must agree")
    reference = np.mean([profile.reference[np.ix_(mapping, mapping)] for profile, mapping in zip(profiles, mappings)], axis=0)
    partial = partial_correlation(reference)
    directions = np.sign(partial)
    stability = np.mean([np.sign(partial_correlation(profile.reference[np.ix_(mapping, mapping)])) == directions for profile, mapping in zip(profiles, mappings)], axis=0)
    weights = np.maximum(0, 2 * stability - 1) * np.abs(partial)
    np.fill_diagonal(weights, 0)
    raw = np.concatenate([profile.marginals.sorted_values[:, mapping] for profile, mapping in zip(profiles, mappings)])
    marginals = Marginals.fit(raw)
    quality = separation(reference, profiles[0].config.kappa)
    quality.update(samples=len(raw), historical_domains=[profile.domain for profile in profiles], cold_start=True, target_samples=0, shrinkage_weight=0.0, reference_scope="Historical pooled copula with source marginals; target transfer is an empirical assumption", evidence_history_complete=all(profile.history is not None and profile.quality.get("evidence_history_complete", True) and (not profile.quality.get("genuine_calibration_identifiers") or profile.quality.get("genuine_calibration_history") is not None) for profile in profiles))
    return DomainProfile(domain, common, names, np.median(raw, axis=0), marginals, reference, directions, weights, stability, identifiers, profiles[0].config, quality, merge(*histories))
def adapt(prior: DomainProfile, bundle: dict | None, domain: str, n0: float = 100.0) -> DomainProfile:
    prior.validate()
    if not np.isfinite(n0) or n0 <= 0:
        raise ValueError("Prior sample strength must be positive")
    result = copy.deepcopy(prior)
    result.domain = domain
    if bundle is None:
        return result
    validate_bundle(bundle)
    if len(bundle["values"]) == 0:
        return result
    if prior.quality.get("reference_selection"):
        raise ValueError("Reselect and refit a selected reference instead of appending unfiltered samples")
    prior.transform(bundle)
    if any(prior.overlap(bundle["rows"])) or any(row.get("split") == "test" for row in prediction_rows(bundle["rows"])):
        raise ValueError("Streaming adaptation must exclude history overlap and test samples")
    if any(row["domain"] != domain for row in bundle["rows"]):
        raise ValueError("Streaming adaptation requires one target domain")
    count = len(bundle["values"])
    result.history = merge(prior.history, collect(bundle["rows"]))
    complete = prior.history is not None and prior.quality.get("evidence_history_complete", True)
    clean = np.where(bundle["observed"][:, prior.columns], bundle["values"][:, prior.columns], prior.medians)
    if count < 4:
        result.identifiers += [row["id"] for row in bundle["rows"]]
        result.quality.update(target_samples=count, shrinkage_weight=0.0, cold_start=True, reference_scope="Fewer than four target samples; historical prior retained", evidence_history_complete=complete)
        return result
    empirical_marginals = Marginals.fit(clean)
    z = empirical_marginals.transform(clean)
    empirical = correlation(z, prior.config.shrinkage)
    weight = count / (count + n0)
    reference = (1 - weight) * prior.reference + weight * empirical
    # Convex pooling preserves positive definiteness.
    result.marginals = empirical_marginals
    result.medians = np.median(clean, axis=0)
    result.reference = reference
    result.directions = np.sign(partial_correlation(reference))
    result.weights = np.maximum(0, 2 * prior.stability - 1) * np.abs(partial_correlation(reference))
    np.fill_diagonal(result.weights, 0)
    result.identifiers += [row["id"] for row in bundle["rows"]]
    result.quality = separation(reference, prior.config.kappa)
    result.identifiers += prior.quality.get("genuine_calibration_identifiers", [])
    result.history = merge(result.history, prior.quality.get("genuine_calibration_history"))
    complete = complete and (not prior.quality.get("genuine_calibration_identifiers") or prior.quality.get("genuine_calibration_history") is not None)
    result.quality.update(target_samples=count, shrinkage_weight=weight, n0=n0, cold_start=count < prior.config.min_samples, reference_scope="F24 convex prior shrinkage; inherited sign stability and target empirical marginals", evidence_history_complete=complete)
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
