# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
from PIL import Image
from engine.data import audit_splits, normalize
from engine.storage import digest, relative_path
from engine.traces import TraceConfig

def preflight(records: list[dict], check_images: bool = False, benchmark: bool = False, min_samples: int = 20, config: TraceConfig | None = None) -> dict:
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
        raise ValueError("The minimum profile size must be an integer of at least two")
    config = config or TraceConfig()
    config.validate()
    errors, warnings = [], []
    try:
        rows = normalize(records)
    except (ValueError, TypeError, OverflowError) as error:
        return {"passed": False, "errors": [str(error)], "warnings": [], "samples": len(records)}
    if not rows:
        errors.append("The manifest is empty")
    try:
        audit = audit_splits(rows)
    except ValueError as error:
        errors.append(str(error))
        audit = None
    counts, splits, groups, texts = defaultdict(Counter), defaultdict(list), defaultdict(set), defaultdict(set)
    for row in rows:
        counts[f"{row['domain']}:{row['split']}"]["samples"] += 1
        counts[f"{row['domain']}:{row['split']}"]["unlabeled" if "label" not in row else "fake" if row["label"] else "real"] += 1
        groups[f"{row['domain']}:{row['split']}"].add(row["group"])
        if row["text"].strip():
            texts[hashlib.sha256(row["text"].strip().encode("utf-8")).hexdigest()].add(row["split"])
        splits[row["split"]].append(row)
    for key in counts:
        counts[key]["groups"] = len(groups[key])
        if counts[key]["groups"] < counts[key]["samples"]:
            warnings.append(f"Partition {key} contains repeated groups; row count is not an independent sample count")
    repeated_text = sum(len(partitions) > 1 for partitions in texts.values())
    if repeated_text:
        warnings.append(f"Repeated text crosses splits under {repeated_text} hashes; review event groups and dataset scope")
    missing_images = sum(not row["image"] for row in rows)
    empty_text = sum(not row["text"].strip() for row in rows)
    semantic = sum("semantic_logit" in row for row in rows)
    if missing_images:
        warnings.append(f"{missing_images} samples are text-only and cannot provide physical image evidence")
    if empty_text:
        warnings.append(f"{empty_text} samples have empty text")
    if semantic != len(rows):
        warnings.append(f"External semantic scores cover {semantic} of {len(rows)} samples")
    if splits["unspecified"]:
        (errors if benchmark else warnings).append("Unspecified splits require an explicit split workflow before benchmarking")
    if benchmark:
        for split in ("train", "validation", "calibration", "test"):
            members = splits[split]
            if not members:
                errors.append(f"The benchmark requires a nonempty {split} partition")
            elif any("label" not in row for row in members):
                errors.append(f"Every {split} sample requires a canonical binary label")
            if split in ("train", "validation") and {row.get("label") for row in members} != {0, 1}:
                errors.append(f"The {split} partition requires both classes")
        domains = sorted({row["domain"] for row in rows})
        for domain in domains:
            training = [row for row in splits["train"] if row["domain"] == domain]
            profile = [row for row in splits["profile"] if row["domain"] == domain]
            if training and profile:
                errors.append(f"Domain {domain} mixes source training and target profile samples")
            if len(training or profile) < min_samples:
                errors.append(f"Domain {domain} has fewer than {min_samples} independent profile candidates")
        if missing_images:
            warnings.append("Text-only inputs do not establish reproduction of the physical image method")
    image_report, hashes, cache = [], defaultdict(set), {}
    if check_images:
        for row in rows:
            if not row["image"]:
                continue
            path = relative_path(row["image"])
            try:
                if path.as_posix() not in cache:
                    with Image.open(path) as image:
                        if getattr(image, "n_frames", 1) > 1:
                            raise ValueError("Animated or multipage inputs require an explicit still-image conversion")
                        width, height = image.size
                        image.load()
                        image_format = image.format
                    if min(width, height) < config.min_side:
                        raise ValueError("Image is smaller than min-side")
                    scale = min(1.0, config.max_side / max(width, height))
                    if int(min(width, height) * scale) < config.tile_grid * 8:
                        raise ValueError("Resized image cannot support the tile grid")
                    cache[path.as_posix()] = {"path": path.as_posix(), "width": width, "height": height, "format": image_format, "sha256": digest(path)}
                information = cache[path.as_posix()]
                if row.get("content_sha256") and row["content_sha256"] != information["sha256"]:
                    raise ValueError("The manifest image checksum is stale")
                hashes[information["sha256"]].add(row["split"])
                image_report.append(dict(information, id=row["id"]))
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                errors.append(f"Image for {row['id']}: {error}")
        leaking = [checksum for checksum, partitions in hashes.items() if len(partitions) > 1]
        if leaking:
            errors.append(f"Image bytes cross splits under {len(leaking)} content hashes")
    return {"version": "FlawPrint preflight 1", "passed": not errors, "errors": errors, "warnings": warnings, "samples": len(rows), "counts": {key: dict(value) for key, value in sorted(counts.items())}, "missing_images": missing_images, "empty_text": empty_text, "semantic_scores": semantic, "audit": audit, "image_check_requested": check_images, "images": image_report, "benchmark_check_requested": benchmark, "minimum_profile_size": min_samples, "label_contract": "zero-real-one-fake", "path_base": "process working directory"}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
