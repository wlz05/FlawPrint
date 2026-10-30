# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import dataclass, asdict
from numbers import Real
import numpy as np
from engine.copula import Marginals, correlation, score, separation, decompose, positive_matrix, moments
from engine.signs import bootstrap_weights, conflicts
from engine.storage import read_json, write_json, validate_bundle
from engine.conformal import validate_labels
from engine.lineage import collect, overlaps, validate as validate_history, disjoint, scope
from engine.results import prediction_rows

@dataclass(frozen=True)
class ProfileConfig:
    shrinkage: float = 0.1
    local_shrinkage: float = 0.35
    bootstrap: int = 100
    min_observed: float = 0.8
    min_samples: int = 20
    sign_tolerance: float = 0.03
    sign_weight: float = 1.0
    kappa: float = 2.0
    seed: int = 42

    def validate(self) -> None:
        settings = (self.shrinkage, self.local_shrinkage, self.min_observed, self.sign_tolerance, self.sign_weight, self.kappa)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in settings) or not np.isfinite(settings).all() or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (self.min_samples, self.bootstrap, self.seed)) or self.seed < 0:
            raise ValueError("Profile settings must be finite with valid integer counts")
        if not 0 < self.min_observed <= 1 or self.min_samples < 4 or self.bootstrap < 1:
            raise ValueError("Invalid profile sample settings")
        if not 0 <= self.shrinkage <= 1 or not 0 <= self.local_shrinkage <= 1 or self.sign_weight < 0 or self.sign_tolerance < 0 or self.kappa <= 0:
            raise ValueError("Invalid profile score settings")

@dataclass
class DomainProfile:
    domain: str
    columns: list[int]
    names: list[str]
    medians: np.ndarray
    marginals: Marginals
    reference: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    stability: np.ndarray
    identifiers: list[str]
    config: ProfileConfig
    quality: dict
    history: dict | None = None

    @classmethod
    def fit(cls, bundle: dict, domain: str, config: ProfileConfig, columns: list[int] | None = None) -> DomainProfile:
        config.validate()
        validate_bundle(bundle)
        roles = {row.get("split") for row in prediction_rows(bundle["rows"])}
        if len(roles) != 1 or not roles.issubset({"train", "profile"}):
            raise ValueError("Domain profiles require one train or profile partition and exclude validation, calibration, and test data")
        values, observed = bundle["values"], bundle["observed"]
        if len(values) < config.min_samples:
            raise ValueError("Insufficient unlabeled profile samples")
        if any(row.get("domain") != domain for row in bundle["rows"]):
            raise ValueError("A domain profile must use a single domain")
        candidates = list(range(values.shape[1])) if columns is None else list(columns)
        # Ignore labels when estimating the domain profile.
        if not candidates or any(isinstance(column, bool) or not isinstance(column, (int, np.integer)) for column in candidates) or len(set(candidates)) != len(candidates) or min(candidates) < 0 or max(candidates) >= values.shape[1]:
            raise ValueError("Invalid profile feature selection")
        # Estimate imputation values from the reference pool only.
        medians = np.array([np.median(values[observed[:, column], column]) if observed[:, column].any() else 0 for column in candidates])
        clean = np.where(observed[:, candidates], values[:, candidates], medians)
        retain = [index for index, column in enumerate(candidates) if observed[:, column].mean() >= config.min_observed and np.ptp(clean[:, index]) > 1e-10]
        if len(retain) < 2:
            raise ValueError("A copula profile requires two nonconstant observed features")
        selected = [candidates[index] for index in retain]
        clean, medians = clean[:, retain], medians[retain]
        marginals = Marginals.fit(clean)
        z = marginals.transform(clean)
        reference = correlation(z, config.shrinkage)
        directions, weights, stability = bootstrap_weights(z, reference, config.bootstrap, config.seed, config.shrinkage, config.sign_tolerance)
        quality = separation(reference, config.kappa)
        quality.update(samples=len(values), retained_features=len(selected), dropped_features=[str(bundle["names"][column]) for column in candidates if column not in selected], condition_number=float(np.linalg.cond(reference)), observed_fraction=observed[:, selected].mean(axis=0).tolist(), reference_scope="Unlabeled mixture profile; genuine-reference moments are conditional diagnostics", correlation_parameters=len(selected) * (len(selected) - 1) // 2, samples_per_dimension=len(values) / len(selected), unregularized_rank_bound=min(len(selected), len(values) - 1), estimation_scope="Shrinkage permits inversion; dimension and conditioning do not certify estimation accuracy or class direction")
        return cls(domain, selected, [str(bundle["names"][column]) for column in selected], medians, marginals, reference, directions, weights, stability, [str(row["id"]) for row in bundle["rows"]], config, quality, collect(bundle["rows"]))

    def overlap(self, rows: list[dict]) -> list[bool]:
        rows = prediction_rows(rows)
        history = overlaps(rows, self.history)
        calibration = overlaps(rows, self.quality.get("genuine_calibration_history"))
        identifiers = set(self.identifiers) | set(self.quality.get("genuine_calibration_identifiers", [])) | set(self.quality.get("selector_fit_identifiers", []))
        return [row["id"] in identifiers or first or second for row, first, second in zip(rows, history, calibration)]

    def transform(self, bundle: dict) -> np.ndarray:
        validate_bundle(bundle)
        if max(self.columns) >= bundle["values"].shape[1]:
            raise ValueError("Feature schema is smaller than the domain profile")
        names = [str(bundle["names"][column]) for column in self.columns]
        if names != self.names:
            raise ValueError("Feature schema does not match the domain profile")
        clean = np.where(bundle["observed"][:, self.columns], bundle["values"][:, self.columns], self.medians)
        return self.marginals.transform(clean)

    def predict(self, bundle: dict, explain: bool = False, signs: bool = True) -> list[dict]:
        if signs and any(row.get("patch_signs_valid_after_attack") is False for row in bundle["rows"]):
            raise ValueError("Disable patch signs for a feature-space MMA bundle")
        z = self.transform(bundle)
        scores = score(z, self.reference)
        scale = max(np.sqrt(self.quality["real_variance"]), np.sqrt(self.quality["independent_variance"]), 1e-6)
        output = []
        overlapping = self.overlap(bundle["rows"])
        complete = self.quality.get("evidence_history_complete", True) and (not self.quality.get("genuine_calibration_identifiers") or self.quality.get("genuine_calibration_history") is not None)
        for index, (row, value) in enumerate(zip(bundle["rows"], scores)):
            evidence = conflicts(bundle["patches"][index], self.columns, self.directions, self.weights, self.names, self.config.local_shrinkage) if signs else {"literal_score": 0.0, "conflict_ratio": 0.0, "available_pairs": 0, "conflict_pairs": 0, "conflicts": []}
            copula_risk = (self.quality["threshold"] - value) / scale
            physical_evidence = value + self.config.sign_weight * evidence["literal_score"]
            # Convert genuine evidence into a fake-positive risk.
            physical_risk = (self.quality["threshold"] - physical_evidence) / scale
            result = {"id": row["id"], "domain": row["domain"], "profile_domain": self.domain, "copula_score": float(value), "copula_risk": float(copula_risk), "sign_score": float(evidence["literal_score"]), "sign_conflict_ratio": float(evidence["conflict_ratio"]), "physical_evidence": float(physical_evidence), "physical_risk": float(physical_risk), "prediction": int(physical_risk >= 0), "available_sign_pairs": evidence["available_pairs"], "conflict_sign_pairs": evidence["conflict_pairs"], "observed_fraction": float(bundle["observed"][index, self.columns].mean()), "profile_overlap": overlapping[index], "profile_overlap_audit": scope(self.history, complete)}
            for key in ("label", "semantic_logit", "platform", "group", "split", "image", "content_sha256", "scenario", "physical_manipulated", "narrative_false"):
                if key in row:
                    result[key] = row[key]
            if explain:
                result["copula_explanation"] = decompose(z[index], self.reference, self.names)
                result["sign_explanation"] = evidence["conflicts"]
            output.append(result)
        return output

    def calibrate_genuine(self, bundle: dict, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1 or any("label" not in row for row in bundle["rows"]):
            raise ValueError("Genuine calibration requires labeled validation samples")
        labels = validate_labels(np.asarray([row["label"] for row in prediction_rows(bundle["rows"])]))
        selected = np.flatnonzero(labels == 0).tolist()
        if len(selected) < 4 or any(row.get("split") != "validation" for row in prediction_rows(bundle["rows"])):
            raise ValueError("Use four genuine validation samples and exclude test data")
        from engine.storage import subset
        scored = self.predict(subset(bundle, selected))
        if any(row["profile_overlap"] for row in scored):
            raise ValueError("Genuine calibration overlaps the fitted domain profile")
        values = np.asarray([row["physical_evidence"] for row in scored])
        self.quality.update(threshold=float(np.quantile(values, alpha, method="lower")), threshold_policy="Source genuine validation quantile on the target profile scale", genuine_calibration_samples=len(values), genuine_calibration_identifiers=[row["id"] for row in scored], genuine_calibration_history=collect(scored), transfer_scope="Error control requires transferable genuine copula and an independent evaluation set")

    def document(self) -> dict:
        self.validate()
        return {"version": "flawprint-profile-v1", "domain": self.domain, "columns": self.columns, "names": self.names, "medians": self.medians, "sorted_values": self.marginals.sorted_values, "reference": self.reference, "directions": self.directions, "weights": self.weights, "stability": self.stability, "identifiers": self.identifiers, "config": asdict(self.config), "quality": self.quality, "evidence_history": self.history}

    def validate(self) -> None:
        self.config.validate()
        validate_history(self.history)
        validate_history(self.quality.get("genuine_calibration_history"))
        disjoint(self.history, self.quality.get("genuine_calibration_history"))
        calibration = self.quality.get("genuine_calibration_identifiers", [])
        if not isinstance(calibration, list) or any(not isinstance(value, str) or not value.strip() for value in calibration) or len(set(calibration)) != len(calibration) or set(calibration) & set(self.identifiers):
            raise ValueError("Genuine calibration identifiers must be unique and independent")
        dimension = len(self.columns)
        if dimension < 2 or any(isinstance(column, bool) or not isinstance(column, (int, np.integer)) or column < 0 for column in self.columns) or len(set(self.columns)) != dimension:
            raise ValueError("Profile columns must be distinct nonnegative integers")
        if not isinstance(self.domain, str) or not self.domain.strip() or len(self.names) != dimension or any(not isinstance(name, str) or not name.strip() for name in self.names) or len(set(self.names)) != dimension:
            raise ValueError("Profile domain and feature names must be valid")
        if self.medians.shape != (dimension,) or not np.isfinite(self.medians).all() or self.marginals.sorted_values.shape[1] != dimension:
            raise ValueError("Profile marginals and medians have inconsistent dimensions")
        reference = positive_matrix(self.reference, True)
        if reference.shape != (dimension, dimension):
            raise ValueError("Profile reference dimensions must match its features")
        for array in (self.directions, self.weights, self.stability):
            if array.shape != reference.shape or not np.isfinite(array).all() or not np.allclose(array, array.T):
                raise ValueError("Profile sign matrices must be finite and symmetric")
        if not np.isin(self.directions, (-1, 0, 1)).all() or np.any(self.weights < 0) or np.any(self.weights > 1) or np.any(np.diag(self.weights) != 0) or np.any(self.stability < 0) or np.any(self.stability > 1):
            raise ValueError("Profile sign matrices contain invalid values")
        if not isinstance(self.identifiers, list) or not self.identifiers or any(not isinstance(identifier, str) or not identifier.strip() for identifier in self.identifiers) or len(set(self.identifiers)) != len(self.identifiers):
            raise ValueError("Profile sample identifiers must be unique and nonempty")
        if not np.isfinite(self.quality.get("threshold", np.nan)):
            raise ValueError("Profile thresholds must be finite")
        expected = moments(reference)
        for key in ("real_variance", "independent_variance"):
            if not np.isfinite(self.quality.get(key, np.nan)) or not np.isclose(self.quality[key], expected[key], rtol=1e-8, atol=1e-10):
                raise ValueError("Profile score scales disagree with the reference matrix")

    def save(self, path: str) -> None:
        write_json(path, self.document())

    @classmethod
    def load(cls, path: str, checksum: str | None = None) -> DomainProfile:
        value = read_json(path) if checksum is None else read_json(path, checksum)
        if not isinstance(value, dict) or value.get("version") != "flawprint-profile-v1":
            raise ValueError("Unsupported profile version")
        try:
            result = cls(value["domain"], value["columns"], value["names"], np.asarray(value["medians"], dtype=float), Marginals(np.asarray(value["sorted_values"], dtype=float)), np.asarray(value["reference"], dtype=float), np.asarray(value["directions"], dtype=float), np.asarray(value["weights"], dtype=float), np.asarray(value["stability"], dtype=float), value["identifiers"], ProfileConfig(**value["config"]), value["quality"], value.get("evidence_history"))
            result.validate()
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Malformed domain profile") from error
        return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
