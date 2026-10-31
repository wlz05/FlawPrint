# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import math
import numpy as np
from engine.data import boolean, identifier, missing, portable_image, SPLITS, SPLIT_ALIASES
from engine.storage import parse_json

NUMBERS = ("label", "prediction", "probability", "fusion_risk", "physical_risk", "physical_evidence", "copula_risk", "copula_score", "semantic_logit", "sign_score", "sign_conflict_ratio", "observed_fraction", "potential_disabled_risk", "potential_risk_delta", "uniform_readout_risk", "query_readout_delta", "summary_readout_risk", "pair_readout_delta", "attention_contrast_rms", "attention_shift_tv", "pair_relation_fraction", "pair_contrast_rms", "pair_attention_shift_tv", "value_transport_rms", "pair_value_transport_rms")
BOOLEANS = ("abstain", "probability_calibrated", "profile_overlap", "physical_available", "has_image", "patch_signs_valid_after_attack", "physical_manipulated", "narrative_false", "physical_reference_available", "embedding_protocol_verified")
STRUCTURES = {"prediction_set": list, "physical_gate_reasons": list, "physical_power_screen": dict, "routing_policy": dict, "copula_explanation": dict, "sign_explanation": list}

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
            if key in ("attention_contrast_rms", "pair_contrast_rms", "value_transport_rms", "pair_value_transport_rms") and value < 0:
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
