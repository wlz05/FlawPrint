# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import os
from engine.data import identifier, missing, portable_image
from engine.storage import resolved
FIELDS = ("group", "image", "content_sha256")
def identity(field: str, value: object) -> str | None:
    if missing(value):
        return None
    return os.path.normcase(str(resolved(portable_image(value)))) if field == "image" else identifier(value)
def collect(*partitions: list[dict]) -> dict:
    output = {field: {} for field in FIELDS}
    for rows in partitions:
        for row in rows:
            for field in FIELDS:
                value = row.get(field)
                key = identity(field, value)
                if key is not None:
                    output[field].setdefault(key, portable_image(value) if field == "image" else identifier(value))
    return {field: [values[key] for key in sorted(values)] for field, values in output.items()}
def validate(history: dict | None) -> None:
    if history is None:
        return
    if not isinstance(history, dict) or set(history) != set(FIELDS):
        raise ValueError("Evidence history requires groups, images, and content identifiers")
    for field, values in history.items():
        if not isinstance(values, list) or any(not isinstance(value, str) or missing(value) or value != value.strip() for value in values):
            raise ValueError("Evidence history values must be nonempty strings")
        keys = [identity(field, value) for value in values]
        if len(set(keys)) != len(keys):
            raise ValueError("Evidence history contains duplicate identities")
def overlaps(rows: list[dict], history: dict | None) -> list[bool]:
    validate(history)
    stored = {field: {identity(field, value) for value in history[field]} if history is not None else set() for field in FIELDS}
    return [any(identity(field, row.get(field)) in stored[field] for field in FIELDS) for row in rows]
def ensure_disjoint(rows: list[dict], history: dict | None) -> None:
    if any(overlaps(rows, history)):
        raise ValueError("Prediction data overlaps recorded groups, images, or content")
def merge(*histories: dict | None) -> dict:
    partitions = []
    for history in histories:
        validate(history)
        partitions.append([{field: value} for field in FIELDS for value in (history[field] if history is not None else [])])
    return collect(*partitions)
def disjoint(*histories: dict | None) -> None:
    previous = None
    for history in histories:
        validate(history)
        if history is not None:
            ensure_disjoint([{field: value} for field in FIELDS for value in history[field]], previous)
        previous = merge(previous, history)
def scope(history: dict | None, complete: bool = True) -> str:
    validate(history)
    return "identifiers_and_recorded_evidence" if history is not None and complete else "identifiers_and_partial_recorded_evidence" if history is not None else "identifiers_only"

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
