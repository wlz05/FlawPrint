# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import math
import statistics
import numpy as np
from engine.data import boolean, identifier, missing, ordinal, portable_image, SPLITS, SPLIT_ALIASES
from engine.storage import parse_json

NUMBERS = ("label", "prediction", "probability", "fusion_risk", "physical_risk", "physical_evidence", "copula_risk", "copula_score", "semantic_logit", "sign_score", "sign_conflict_ratio", "observed_fraction", "potential_disabled_risk", "potential_risk_delta", "uniform_readout_risk", "query_readout_delta", "summary_readout_risk", "pair_readout_delta", "attention_contrast_rms", "attention_shift_tv", "pair_relation_fraction", "pair_contrast_rms", "pair_attention_shift_tv", "value_transport_rms", "pair_value_transport_rms", "pair_tangent_transport_rms", "pair_transport_remainder_rms", "pair_transport_remainder_bound")
BOOLEANS = ("abstain", "probability_calibrated", "profile_overlap", "physical_available", "has_image", "physical_trace_disabled", "patch_signs_valid_after_attack", "physical_manipulated", "narrative_false", "physical_reference_available", "embedding_protocol_verified")
STRUCTURES = {"prediction_set": list, "physical_gate_reasons": list, "physical_power_screen": dict, "routing_policy": dict, "copula_explanation": dict, "sign_explanation": list}

SEEDS = (42, 52, 3047, 3407)

def seed_summary(records, seeds=SEEDS):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("Seeds must be distinct unsigned 32-bit integers")
    if any(not isinstance(row, dict) or type(row.get("seed")) is not int or "metrics" not in row or not isinstance(row.get("protocol"), dict) for row in records):
        raise ValueError("Every seed result needs metrics and an experiment protocol")
    if len(records) != len(seeds) or {row["seed"] for row in records} != set(seeds):
        raise ValueError("Missing or duplicate seed results; partial averages are forbidden")
    if any(row["protocol"] != records[0]["protocol"] for row in records):
        raise ValueError("All seeds must use the same inputs and experiment settings")
    def flatten(value, prefix=""):
        result = {}
        if isinstance(value, dict):
            for key, item in value.items():
                result.update(flatten(item, prefix + ("." if prefix else "") + str(key)))
        elif value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
            if value is not None and not math.isfinite(value):
                raise ValueError("Seed metrics must be finite or explicitly undefined")
            result[prefix] = value
        return result
    byseed = {str(row["seed"]): row["metrics"] for row in records}
    flat = {seed: flatten(byseed[str(seed)]) for seed in seeds}
    metrics = {}
    for key in sorted(set().union(*(value.keys() for value in flat.values()))):
        values = [flat[seed].get(key) for seed in seeds]
        undefined = [seed for seed, value in zip(seeds, values) if value is None]
        metrics[key] = {"mean": statistics.mean(values) if not undefined else None, "sample_std": statistics.stdev(values) if not undefined and len(values) > 1 else None, "undefined_seeds": undefined}
    return {"seeds": seeds, "seed_count": len(seeds), "byseed": byseed, "metrics": metrics, "protocol": records[0]["protocol"], "aggregation": "Compute each seed's test metrics first, then take their equally weighted arithmetic mean."}

def numeric(value: object, key: str) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)) or not np.isscalar(value):
        raise ValueError("Numeric prediction fields require real scalar values: " + key)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid numeric prediction field: " + key) from error
    if not math.isfinite(result):
        raise ValueError("Prediction values must be finite: " + key)
    return result

def score_rows(records: list[dict], score_key: str) -> list[dict]:
    rows = prediction_rows(records)
    for row in rows:
        if score_key not in row or missing(row[score_key]):
            raise ValueError("Every scored input requires an observed score: " + score_key)
        row[score_key] = numeric(row[score_key], score_key)
    return rows

def prediction_rows(records: list[dict]) -> list[dict]:
    output = []
    for number, original in enumerate(records, 1):
        if not isinstance(original, dict):
            raise ValueError("Prediction rows must be objects")
        row = dict(original)
        if "id" in row:
            row["id"] = identifier(row["id"])
            if not row["id"]:
                raise ValueError("Prediction identifiers must be nonempty")
        for key in ("domain", "group", "platform", "content_sha256"):
            if key in row:
                value = identifier(row[key])
                if value:
                    row[key] = value
                else:
                    row.pop(key)
        if "image" in row:
            row["image"] = portable_image(row["image"])
        if "split" in row:
            if missing(row["split"]):
                row.pop("split")
            elif not isinstance(row["split"], str):
                raise ValueError("Prediction splits must be strings")
            else:
                value = row["split"].strip().casefold()
                row["split"] = SPLIT_ALIASES.get(value, value)
                if row["split"] not in (*SPLITS, "unspecified"):
                    raise ValueError("Unknown prediction split")
        for key, maximum in (("image_frame", 0), ("source_ordinal", 16908)):
            if key in row:
                if missing(row[key]):
                    row.pop(key)
                else:
                    row[key] = ordinal(row[key], maximum, key)
        for key in NUMBERS:
            if key not in row:
                continue
            if missing(row[key]):
                if key == "label":
                    row.pop(key)
                else:
                    row[key] = None
                continue
            value = numeric(row[key], key)
            if key in ("label", "prediction"):
                if value not in (0, 1):
                    raise ValueError("Prediction labels and decisions must be binary")
                value = int(value)
            if key in ("probability", "observed_fraction", "sign_conflict_ratio", "attention_shift_tv", "pair_relation_fraction", "pair_attention_shift_tv") and not 0 <= value <= 1:
                raise ValueError("Prediction probabilities and bounded diagnostics must lie in the unit interval")
            if key in ("attention_contrast_rms", "pair_contrast_rms", "value_transport_rms", "pair_value_transport_rms", "pair_tangent_transport_rms", "pair_transport_remainder_rms", "pair_transport_remainder_bound") and value < 0:
                raise ValueError("Attention contrast RMS must be nonnegative")
            row[key] = value
        for key in BOOLEANS:
            if key in row:
                if missing(row[key]):
                    row.pop(key)
                else:
                    row[key] = boolean(row[key])
        for key, kind in STRUCTURES.items():
            if key not in row:
                continue
            if missing(row[key]):
                row[key] = None
                continue
            value = parse_json(row[key]) if isinstance(row[key], str) else row[key]
            if not isinstance(value, kind):
                raise ValueError(f"Invalid prediction structure: {key}")
            if key == "prediction_set" and (any(isinstance(label, bool) or label not in (0, 1) for label in value) or len(set(value)) != len(value)):
                raise ValueError("Prediction sets require unique binary labels")
            if key == "physical_gate_reasons" and any(not isinstance(reason, str) for reason in value):
                raise ValueError("Gate reasons must be strings")
            row[key] = value
        output.append(row)
    return output

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
