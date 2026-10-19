# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from engine.traces import NAMES
from engine.storage import relative_path, write_rows, save_bundle
def fixture(seed: int = 42, per_split: int = 24) -> dict:
    if per_split < 12:
        raise ValueError("Synthetic splits require twelve examples")
    rng = np.random.default_rng(seed)
    rows, values, patches = [], [], []
    dimension = len(NAMES)
    reference = np.full((dimension, dimension), 0.3)
    np.fill_diagonal(reference, 1)
    for domain in ("source_alpha", "source_beta", "target_gamma"):
        splits = ("profile", "test") if domain.startswith("target") else ("train", "validation", "calibration", "test")
        for split in splits:
            for index in range(per_split):
                label = index % 2
                identifier = f"{domain}_{split}_{index:04d}"
                raw = rng.multivariate_normal(np.zeros(dimension), reference) if label == 0 else rng.normal(size=dimension)
                shift = 0 if domain == "source_alpha" else 2 if domain == "source_beta" else 5
                values.append(np.exp(raw / 3) + shift)
                tile_reference = reference if label == 0 else np.eye(dimension)
                patches.append(np.exp(rng.multivariate_normal(np.zeros(dimension), tile_reference, 16) / 3) + shift)
                row = {"id": identifier, "image": f"./synthetic/{identifier}.png", "text": "A verified report with context." if label == 0 else "An invented report with unsupported claims!", "domain": domain, "platform": "synthetic", "split": split, "group": identifier, "semantic_logit": float((2 * label - 1) * 0.6 + rng.normal(0, 1))}
                if split != "profile":
                    row["label"] = label
                rows.append(row)
    return {"values": np.asarray(values), "observed": np.ones((len(rows), dimension), dtype=bool), "patches": np.asarray(patches), "names": np.asarray(NAMES), "rows": rows}
def write_fixture(output: str, seed: int = 42, per_split: int = 24) -> dict:
    bundle = fixture(seed, per_split)
    save_bundle(output, bundle["values"], bundle["observed"], bundle["patches"], bundle["rows"], NAMES)
    return {"samples": len(bundle["rows"]), "features": len(NAMES), "output": output, "scope": "Synthetic feature fixture only"}
def image_demo(output: str = "./data/demo", seed: int = 42, per_split: int = 16) -> dict:
    bundle = fixture(seed, per_split)
    root = relative_path(output)
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((96, 96))
    manifest = []
    for index, row in enumerate(bundle["rows"]):
        identifier = row["id"]
        path = root / "images" / f"{identifier}.jpg"
        if path.exists():
            raise FileExistsError("Demo image output already exists")
        phase = rng.uniform(0, np.pi * 2)
        field = 0.5 + 0.2 * np.sin(xx / rng.uniform(7, 18) + phase) + 0.1 * np.cos(yy / rng.uniform(8, 20))
        pixels = np.stack([field, np.roll(field, index % 4, axis=1), np.roll(field, index % 3, axis=0)], axis=-1)
        noise = rng.uniform(0.01, 0.06)
        pixels += rng.normal(0, noise, pixels.shape)
        if index % 2:
            pixels = gaussian_filter(pixels, sigma=(0.6, 0.6, 0))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(np.clip(pixels, 0, 1) * 255)).save(path, quality=int(rng.integers(55, 96)))
        manifest.append(dict(row, image=path.as_posix()))
    write_rows(root / "manifest.jsonl", manifest)
    return {"manifest": (root / "manifest.jsonl").as_posix(), "samples": len(manifest), "scope": "Artificial images for integration checks only"}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
