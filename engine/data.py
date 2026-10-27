# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import os
import unicodedata
import numpy as np
from engine.storage import relative_path, resolved, digest

SPLITS = ("train", "validation", "calibration", "profile", "test")
SPLIT_ALIASES = {"val": "validation", "valid": "validation", "dev": "validation", "training": "train", "testing": "test", "cal": "calibration"}

def missing(value: object) -> bool:
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return isinstance(value, str) and value.strip().casefold() in ("", "none", "null", "nan", "<na>")

def text_fingerprint(value: object) -> str | None:
    if value is None or type(value).__name__ in ("NAType", "NaTType") or isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    text = unicodedata.normalize("NFC", " ".join(str(value).split()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None

def image_fingerprints(rows: list[dict], required: bool = False) -> dict[str, str]:
    fingerprints = {}
    for row in rows:
        image = portable_image(row.get("image", ""))
        if not image:
            continue
        path = relative_path(image)
        canonical = os.path.normcase(str(resolved(path)))
        if canonical not in fingerprints:
            if not path.is_file():
                if required:
                    raise FileNotFoundError("Missing image: " + image)
                continue
            fingerprints[canonical] = digest(path)
        if row.get("content_sha256") and row["content_sha256"] != fingerprints[canonical]:
            raise ValueError("The manifest image checksum is stale")
    return fingerprints

def identifier(value: object) -> str:
    if missing(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or value != int(value) or abs(value) >= 2 ** 53:
            raise ValueError("Numeric identifiers must be exact safe integers; store large IDs as text")
        return str(int(value))
    if not isinstance(value, (str, int, np.integer)) or isinstance(value, bool):
        raise ValueError("Identifiers must be strings or integer values")
    return str(value).strip()

def stable_identifier(text: str, image: str, domain: str) -> str:
    content = json.dumps({"text": text, "image": image, "domain": domain}, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return "sample:" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]

def portable_image(value: object, root: str | Path = ".") -> str:
    if missing(value):
        return ""
    if not isinstance(value, str):
        raise ValueError("Each canonical image field must contain one relative path")
    value = value.strip().replace("\\", "/")
    if value.casefold().startswith(("http://", "https://", "file://")):
        raise ValueError("Download image URLs locally before conversion")
    image, base = relative_path(value), relative_path(str(root).replace("\\", "/"))
    if base.parts and image.parts[:len(base.parts)] != base.parts:
        image = base / image
    return Path(os.path.normpath(image)).as_posix()

def boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().casefold() in ("true", "false", "0", "1"):
        return value.strip().casefold() in ("true", "1")
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    raise ValueError("Boolean metadata must be true, false, zero, or one")

def normalize(rows: list[dict], image_root: str = ".", label_convention: str = "fake-one", domain_key: str = "domain") -> list[dict]:
    if label_convention not in ("fake-one", "real-one"):
        raise ValueError("Unknown label convention")
    root = relative_path(image_root)
    output, identifiers = [], set()
    for index, original in enumerate(rows):
        if not isinstance(original, dict):
            raise ValueError(f"Sample {index + 1} must be an object")
        row = dict(original)
        image = portable_image(row.get("image", row.get("image_path", "")), root)
        text = row.get("text", "")
        row["text"] = "" if text is None or type(text).__name__ in ("NAType", "NaTType") or isinstance(text, (float, np.floating)) and np.isnan(text) else str(text)
        row["domain"] = "unknown" if missing(row.get(domain_key)) else str(row[domain_key]).strip()
        sample_id = identifier(row.get("id")) or stable_identifier(row["text"], image, row["domain"])
        if sample_id in identifiers:
            raise ValueError("Sample identifiers must be unique and nonempty")
        identifiers.add(sample_id)
        row["image"] = image
        row["has_image"] = bool(image)
        row["id"] = sample_id
        row["platform"] = "unknown" if missing(row.get("platform")) else str(row["platform"]).strip()
        row["group"] = identifier(row.get("group", row.get("event_id"))) or sample_id
        split = "unspecified" if missing(row.get("split")) else str(row["split"]).strip().casefold()
        row["split"] = SPLIT_ALIASES.get(split, split)
        if row["split"] not in (*SPLITS, "unspecified"):
            raise ValueError(f"Invalid split for sample {sample_id}")
        label = row.get("label")
        if not missing(label):
            if isinstance(label, (bool, np.bool_, complex, np.complexfloating)) or not np.isscalar(label):
                raise ValueError("Labels must be scalar zero or one values, not Boolean metadata")
            if isinstance(label, str) and label.strip().casefold() in ("real", "true", "fake", "false"):
                row["label"] = int(label.strip().casefold() in ("fake", "false"))
            else:
                number = float(label)
                if number not in (0, 1):
                    raise ValueError("Binary labels must be zero or one")
                number = int(number)
                row["label"] = number if label_convention == "fake-one" else 1 - number
        else:
            row.pop("label", None)
        if row["split"] == "profile":
            row.pop("label", None)
        if not missing(row.get("semantic_logit")):
            if isinstance(row["semantic_logit"], (bool, np.bool_, complex, np.complexfloating)) or not np.isscalar(row["semantic_logit"]):
                raise ValueError("Semantic logits must be real scalar scores")
            row["semantic_logit"] = float(row["semantic_logit"])
            if not np.isfinite(row["semantic_logit"]):
                raise ValueError("Semantic logits must be finite")
        else:
            row.pop("semantic_logit", None)
        for key in ("physical_manipulated", "narrative_false", "patch_signs_valid_after_attack"):
            if key in row and not missing(row[key]):
                row[key] = boolean(row[key])
            else:
                row.pop(key, None)
        output.append(row)
    return output

def partition(rows: list[dict], target_domains: list[str], seed: int = 42, profile_fraction: float = 0.35) -> list[dict]:
    from engine.lineage import components
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("The split seed must be an unsigned integer")
    if not 0 < profile_fraction < 1:
        raise ValueError("Profile fraction must be between zero and one")
    targets = set(target_domains)
    available = {row["domain"] for row in rows}
    if not targets or not targets < available:
        raise ValueError("Select existing target domains and retain a source domain")
    rows = [dict(row) for row in rows]
    fingerprints = image_fingerprints(rows)
    for row in rows:
        if row.get("image"):
            key = os.path.normcase(str(resolved(portable_image(row["image"]))))
            if key in fingerprints:
                row["content_sha256"] = fingerprints[key]
    grouped = defaultdict(list)
    for group, row in zip(components(rows), rows):
        grouped[group].append(dict(row))
    assignments = {"source": [], "target": []}
    for group, members in sorted(grouped.items(), key=lambda item: min(str(row["id"]) for row in item[1])):
        sides = {row["domain"] in targets for row in members}
        if len(sides) != 1:
            raise ValueError("A duplicate group crosses source and target domains")
        assignments["target" if sides.pop() else "source"].append((group, members))
    rng = np.random.default_rng(seed)
    result = []
    for side, groups in assignments.items():
        rng.shuffle(groups)
        if len(groups) < (4 if side == "source" else 2):
            raise ValueError("Too few independent groups for the requested split")
        size = len(groups)
        for index, (_, members) in enumerate(groups):
            if side == "target":
                split = "profile" if index < max(1, min(size - 1, int(size * profile_fraction))) else "test"
            else:
                first = max(1, min(size - 3, int(size * 0.65)))
                second = max(first + 1, min(size - 2, int(size * 0.80)))
                third = max(second + 1, min(size - 1, int(size * 0.90)))
                split = "train" if index < first else "validation" if index < second else "calibration" if index < third else "test"
            for row in members:
                row["split"] = split
                if split == "profile":
                    row.pop("label", None)
                result.append(row)
    result = sorted(result, key=lambda row: row["id"])
    audit_splits(result)
    return result

def audit_splits(rows: list[dict], check_files: bool = False) -> dict:
    fingerprints = image_fingerprints(rows, check_files)
    groups, images, contents, texts = (defaultdict(set) for _ in range(4))
    counts, paths = defaultdict(int), {}
    for row in rows:
        split = row["split"]
        groups[str(row.get("group", row["id"]))].add(split)
        text_hash = text_fingerprint(row.get("text"))
        if text_hash is not None:
            texts[text_hash].add(split)
        if row.get("image"):
            canonical = os.path.normcase(str(resolved(portable_image(row["image"]))))
            images[canonical].add(split)
            paths.setdefault(canonical, portable_image(row["image"]))
            if canonical in fingerprints:
                contents[fingerprints[canonical]].add(split)
        counts[f"{row['domain']}:{split}"] += 1
        if row.get("content_sha256"):
            contents[str(row["content_sha256"])].add(split)
    leaks = {"groups": [key for key, splits in groups.items() if len(splits) > 1], "paths": [paths[key] for key, splits in images.items() if len(splits) > 1], "content": [key for key, splits in contents.items() if len(splits) > 1], "text": [key for key, splits in texts.items() if len(splits) > 1]}
    if any(leaks.values()):
        raise ValueError(f"Cross-split leakage detected: {leaks}")
    return {"counts": dict(counts), "samples": len(rows), "leakage": leaks, "label_convention": "zero-real-one-fake"}

def join_predictions(rows: list[dict], predictions: list[dict], score_key: str = "logit", real_positive: bool = False) -> list[dict]:
    lookup = {}
    for prediction in predictions:
        sample_id = identifier(prediction["id"])
        if not sample_id or sample_id in lookup:
            raise ValueError("Duplicate external prediction identifier")
        value = float(prediction[score_key])
        if not np.isfinite(value):
            raise ValueError("External scores must be finite")
        lookup[sample_id] = -value if real_positive else value
    if set(lookup) != {row["id"] for row in rows}:
        raise ValueError("External predictions must match manifest identifiers exactly")
    return [dict(row, semantic_logit=lookup[row["id"]]) for row in rows]

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
