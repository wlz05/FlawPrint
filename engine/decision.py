# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict
import numpy as np
from engine.profile import DomainProfile
from engine.traces import NAMES
from engine.dependence import availability
from engine.conformal import calibrate_rows, apply_calibration, validate_partitions, validate_labels
from engine.storage import validate_bundle
from engine.metrics import evaluate, binary
from engine.results import prediction_rows
from engine.lineage import collect, ensure_disjoint, scope

@dataclass(frozen=True)
class GatePolicy:
    min_observed: float = 0.8
    min_image_features: int = 2
    alpha: float = 0.05
    semantic_threshold: float = 0.0

    def validate(self) -> None:
        if not 0 < self.min_observed <= 1 or isinstance(self.min_image_features, bool) or not isinstance(self.min_image_features, (int, np.integer)) or self.min_image_features < 2 or not 0 < self.alpha < 0.5 or not np.isfinite(self.semantic_threshold):
            raise ValueError("Invalid deployment gate policy")

def physical_status(profile: DomainProfile | None, bundle: dict, policy: GatePolicy) -> list[dict]:
    policy.validate()
    validate_bundle(bundle)
    if profile is not None:
        profile.transform(bundle)
    positions = [index for index, name in enumerate(profile.names) if name in NAMES[:20]] if profile else []
    columns = [profile.columns[index] for index in positions] if profile else []
    screen = availability(profile.reference[np.ix_(positions, positions)], policy.alpha) if len(positions) >= policy.min_image_features else None
    result = []
    for index, row in enumerate(bundle["rows"]):
        observed = float(bundle["observed"][index, columns].mean()) if columns else 0.0
        reasons = []
        if not row.get("image") or row.get("has_image") is False:
            reasons.append("missing_image")
        if profile is None:
            reasons.append("missing_profile")
        if len(columns) < policy.min_image_features:
            reasons.append("insufficient_image_features")
        if observed < policy.min_observed:
            reasons.append("insufficient_observed_image_features")
        if screen is not None and not screen["power_screen"]:
            reasons.append("domain_power_screen_failed")
        result.append({"physical_available": not reasons, "physical_gate_reasons": reasons, "image_observed_fraction": observed, "image_profile_features": len(columns), "physical_power_screen": screen})
    return result

def route_rows(rows: list[dict], statuses: list[dict], policy: GatePolicy) -> list[dict]:
    policy.validate()
    if len(rows) != len(statuses):
        raise ValueError("Routing status count does not match predictions")
    output = []
    for row, status in zip(rows, statuses):
        result = dict(row, **status)
        physical = row.get("physical_risk")
        semantic = row.get("semantic_logit")
        physical_ready = status["physical_available"] and physical is not None and np.isfinite(physical)
        semantic_ready = semantic is not None and np.isfinite(semantic)
        if status["physical_available"] and not physical_ready:
            result["physical_available"] = False
            result["physical_gate_reasons"] = list(status["physical_gate_reasons"]) + ["missing_or_nonfinite_physical_score"]
        result["semantic_available"] = bool(semantic_ready)
        # Route through the channels with usable evidence.
        if physical_ready and semantic_ready:
            route, risk = "fusion", max(float(physical), float(semantic) - policy.semantic_threshold)
        elif semantic_ready:
            route, risk = "semantic", float(semantic) - policy.semantic_threshold
        elif physical_ready:
            route, risk = "physical", float(physical)
        else:
            route, risk = "abstain", None
        result.update(decision_route=route, fusion_risk=risk, prediction=int(risk >= 0) if risk is not None else None, abstain=risk is None, probability=None, probability_calibrated=False, prediction_set=None, decision_scope="Available evidence only; physical consistency does not establish factual truth", routing_policy=asdict(policy))
        output.append(result)
    return output

def deploy(bundle: dict, profile: DomainProfile | None, policy: GatePolicy, explain: bool = False, signs: bool = True, calibration: dict | None = None) -> list[dict]:
    statuses = physical_status(profile, bundle, policy)
    if profile is not None:
        if any(profile.overlap(bundle["rows"])):
            raise ValueError("Deployment examples overlap the fitted profile")
        rows = profile.predict(bundle, explain, signs)
    else:
        rows = [dict(row) for row in bundle["rows"]]
    for row, original in zip(rows, bundle["rows"]):
        for key in ("scenario", "physical_manipulated", "narrative_false"):
            if key in original:
                row[key] = original[key]
    routed = route_rows(rows, statuses, policy)
    return apply_routes(routed, calibration)

def calibrate_routes(validation: list[dict], calibration: list[dict], alpha: float = 0.1) -> dict:
    validation, calibration = prediction_rows(validation), prediction_rows(calibration)
    if not validation or not calibration:
        raise ValueError("Routed calibration requires both independent input partitions")
    validate_partitions(validation, calibration)
    if not 0 < alpha < 1 or any(row.get("split") == "test" or row.get("profile_overlap") for row in validation + calibration):
        raise ValueError("Routed calibration requires valid alpha and independent non-test samples")
    if any(row.get("decision_route") not in ("fusion", "semantic", "physical", "abstain") for row in validation + calibration):
        raise ValueError("Routed calibration contains an unknown evidence route")
    policy = validation[0]["routing_policy"]
    GatePolicy(**policy).validate()
    if any(row.get("routing_policy") != policy for row in validation + calibration):
        raise ValueError("Routed calibration policies must match")
    states, skipped = {}, {}
    for route in ("fusion", "semantic", "physical"):
        first = [row for row in validation if row.get("decision_route") == route]
        second = [row for row in calibration if row.get("decision_route") == route]
        if first and second and len(set(validate_labels(np.asarray([row.get("label", np.nan) for row in first])))) == 2:
            states[route] = calibrate_rows(first, second, alpha)
        else:
            skipped[route] = "Independent validation with both classes and labeled calibration are required"
    if not states:
        raise ValueError("No evidence route has independent validation and calibration samples")
    return {"version": "flawprint-route-calibration-v1", "routing_policy": policy, "routes": states, "skipped_routes": skipped, "validation_identifiers": [row["id"] for row in validation], "calibration_identifiers": [row["id"] for row in calibration], "evidence_history": collect(validation, calibration), "scope": "Route-conditional coverage requires exchangeable labeled samples within each route"}

def apply_routes(rows: list[dict], state: dict | None = None) -> list[dict]:
    rows = prediction_rows(rows)
    validate_partitions(rows)
    if state is not None:
        if state.get("version") != "flawprint-route-calibration-v1":
            raise ValueError("Deployment requires a route-specific calibration state")
        if any(row["routing_policy"] != state["routing_policy"] for row in rows):
            raise ValueError("Deployment and route calibration policies differ")
        GatePolicy(**state["routing_policy"]).validate()
        history = state["validation_identifiers"] + state["calibration_identifiers"]
        if len(set(history)) != len(history) or any(not isinstance(identifier, str) or not identifier.strip() for identifier in history) or set(state["routes"]) - {"fusion", "semantic", "physical"}:
            raise ValueError("Malformed routed calibration state")
        forbidden = set(history)
        if forbidden & {row["id"] for row in rows}:
            raise ValueError("Deployment data overlaps routed calibration")
        ensure_disjoint(rows, state.get("evidence_history"))
    output = []
    for row in rows:
        route = row["decision_route"]
        if route not in ("fusion", "semantic", "physical", "abstain"):
            raise ValueError("Unknown evidence route")
        route_state = state["routes"].get(route) if state else None
        if route == "abstain":
            result = dict(row, calibration_status="no_available_evidence")
        else:
            result = apply_calibration([row], route_state)[0]
            result["calibration_status"] = "route_calibrated" if route_state else "uncalibrated_route"
            if result["abstain"]:
                result["prediction"] = None
        result["route_overlap_audit"] = scope(state.get("evidence_history")) if state else "not_calibrated"
        output.append(result)
    return output

def deployment_report(rows: list[dict], stratify: bool = True) -> dict:
    rows = prediction_rows(rows)
    if not rows:
        raise ValueError("Deployment reporting requires predictions")
    accepted = [row for row in rows if not row["abstain"] and row.get("prediction") in (0, 1)]
    scored = [row for row in rows if row.get("fusion_risk") is not None]
    report = {"samples": len(rows), "scored_samples": len(scored), "accepted_samples": len(accepted), "decision_coverage": len(accepted) / len(rows), "physical_available_fraction": float(np.mean([row["physical_available"] for row in rows])), "routes": dict(Counter(row["decision_route"] for row in rows)), "gate_reasons": dict(Counter(reason for row in rows for reason in row["physical_gate_reasons"])), "calibration_status": dict(Counter(row["calibration_status"] for row in rows)), "scope": "Coverage denominator includes samples with no usable evidence"}
    if scored and all("label" in row for row in scored):
        diagnostic = [dict(row, prediction=int(row["fusion_risk"] >= 0)) for row in scored]
        report["scored_diagnostics"] = evaluate(diagnostic)
    if accepted and all("label" in row for row in accepted):
        report["accepted_metrics"] = binary(np.array([row["label"] for row in accepted]), np.array([row["fusion_risk"] for row in accepted]), predictions=np.array([row["prediction"] for row in accepted]))
        report["accepted_risk"] = 1 - report["accepted_metrics"]["accuracy"]
    else:
        report["accepted_metrics"], report["accepted_risk"] = None, None
    report["route_reports"] = {route: {"samples": len(part), "accepted": sum(not row["abstain"] for row in part), "labeled_samples": sum("label" in row for row in part)} for route in sorted(report["routes"]) for part in [[row for row in rows if row["decision_route"] == route]]}
    if stratify:
        scenarios = sorted({str(row.get("scenario", "unspecified")) for row in rows})
        if any(value != "unspecified" for value in scenarios):
            report["scenarios"] = {value: deployment_report([row for row in rows if str(row.get("scenario", "unspecified")) == value], False) for value in scenarios}
    return report

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
