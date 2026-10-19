# -*- coding: utf-8 -*-
# Robustness: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image, ImageFilter
from engine.storage import relative_path, write_rows
def corrupt(rows: list[dict], output: str, operation: str, severity: float, seed: int = 42) -> list[dict]:
    if operation not in ("jpeg", "resize", "blur", "noise", "crop", "strip_metadata"):
        raise ValueError("Unknown corruption operation")
    if not np.isfinite(severity) or severity <= 0:
        raise ValueError("Severity must be positive")
    folder = relative_path(output)
    rng = np.random.default_rng(seed)
    results = []
    for row in rows:
        identifier = hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest()[:20]
        target = folder / "images" / f"{identifier}{'.jpg' if operation == 'jpeg' else '.png'}"
        if target.exists():
            raise FileExistsError("Corruption output already exists")
        with Image.open(relative_path(row["image"])) as opened:
            image = opened.convert("RGB")
        if operation == "resize":
            if not 0 < severity <= 1:
                raise ValueError("Resize severity must lie in the unit interval")
            image = image.resize(tuple(max(32, int(size * severity)) for size in image.size), Image.Resampling.LANCZOS)
        elif operation == "blur":
            image = image.filter(ImageFilter.GaussianBlur(severity))
        elif operation == "noise":
            pixels = np.asarray(image, dtype=float) / 255
            image = Image.fromarray(np.uint8(np.clip(pixels + rng.normal(0, severity, pixels.shape), 0, 1) * 255))
        elif operation == "crop":
            if not 0 < severity < 0.4:
                raise ValueError("Crop severity must be below 0.4")
            width, height = image.size
            image = image.crop((int(width * severity), int(height * severity), int(width * (1 - severity)), int(height * (1 - severity))))
        target.parent.mkdir(parents=True, exist_ok=True)
        if operation == "jpeg":
            if not 1 <= severity <= 95:
                raise ValueError("JPEG quality must be between one and 95")
            image.save(target, quality=int(severity), subsampling=2)
        else:
            image.save(target)
        item = dict(row, id=f"{row['id']}:{operation}:{severity:g}", image=target.as_posix(), original_id=row["id"], corruption=operation, severity=severity)
        results.append(item)
    write_rows(folder / "manifest.jsonl", results)
    return results
