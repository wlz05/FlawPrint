# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
import re
import numpy as np
from engine.data import normalize
from engine.storage import relative_path, digest, native

STREAMS = ("bert", "mae", "clip_text", "clip_image")

def checkpoint_digest(directory: str) -> str:
    root = relative_path(directory)
    if not root.is_dir():
        raise ValueError("A local encoder directory is missing")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("An encoder directory is empty")
    checksum = hashlib.sha256()
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Encoder files must remain inside their checkpoint directory")
        checksum.update(path.relative_to(root).as_posix().encode() + b"\0" + digest(path).encode())
    return checksum.hexdigest()

def validate(archive: dict) -> None:
    rows, features, observed = archive["rows"], archive["features"], archive["observed"]
    names = list(features)
    if names not in (list(STREAMS), list(STREAMS) + ["forensic"]):
        raise ValueError("Embedding streams must use the recorded canonical order")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Embedding identities must be unique and nonempty")
    if observed.dtype != np.bool_ or observed.shape != (len(rows), len(names)):
        raise ValueError("Embedding availability requires a Boolean sample by stream mask")
    for column, name in enumerate(names):
        values = features[name]
        if not np.issubdtype(values.dtype, np.floating) or values.ndim != 2 or values.shape[0] != len(rows) or not 0 < values.shape[1] <= 8192 or not np.isfinite(values).all():
            raise ValueError("Embedding arrays require finite aligned feature vectors")
        if np.any(values[~observed[:, column]] != 0):
            raise ValueError("Missing embedding streams must contain zero vectors")
    if not observed.any(axis=1).all():
        raise ValueError("Each sample requires at least one observed deep stream")
    if not isinstance(archive.get("provenance"), dict):
        raise ValueError("Embedding provenance is required")
    encoders = archive["provenance"].get("encoders", {})
    expected = {"bert", "mae", "clip"} | ({"forensic"} if "forensic" in names else set())
    if not isinstance(encoders, dict) or set(encoders) != expected or any(not isinstance(value, dict) or value.get("frozen") is not True or not re.fullmatch("[0-9a-f]{64}", str(value.get("sha256", ""))) for value in encoders.values()):
        raise ValueError("Record a frozen checkpoint hash for every encoder")

def save(path: str, archive: dict) -> None:
    validate(archive)
    target = relative_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"feature_" + name: values.astype(np.float32) for name, values in archive["features"].items()}
    metadata = {"version": "flawprint-embeddings-v1", "streams": list(archive["features"]), "rows": archive["rows"], "provenance": archive["provenance"]}
    with target.open("wb") as stream:
        np.savez_compressed(stream, **arrays, observed=archive["observed"], metadata=np.asarray(json.dumps(native(metadata), ensure_ascii=True, allow_nan=False)))

def load(path: str) -> dict:
    with np.load(relative_path(path), allow_pickle=False) as stream:
        metadata = json.loads(str(stream["metadata"].item()))
        if metadata.get("version") != "flawprint-embeddings-v1":
            raise ValueError("Unsupported embedding archive version")
        archive = {"features": {name: stream["feature_" + name].copy() for name in metadata["streams"]}, "observed": stream["observed"].copy(), "rows": metadata["rows"], "provenance": metadata["provenance"]}
    validate(archive)
    return archive

def align(bundle: dict, archive: dict) -> tuple[dict, np.ndarray]:
    validate(archive)
    mapping = {row["id"]: index for index, row in enumerate(archive["rows"])}
    if set(mapping) != {row["id"] for row in bundle["rows"]}:
        raise ValueError("Deep and physical inputs must contain the same identities")
    order = []
    for row in bundle["rows"]:
        index = mapping[row["id"]]
        other = archive["rows"][index]
        if any(row.get(key) != other.get(key) for key in ("text", "image", "group", "domain", "split")):
            raise ValueError("Cached embeddings disagree with the physical input metadata")
        if row.get("content_sha256") and row["content_sha256"] != other.get("content_sha256"):
            raise ValueError("Deep and physical features describe different image bytes")
        order.append(index)
    return {name: values[order] for name, values in archive["features"].items()}, archive["observed"][order]

def extract(rows: list[dict], directories: dict, batch_size: int = 64, device: str = "cuda:0", max_length: int = 197) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor, ViTMAEModel
    if isinstance(batch_size, bool) or batch_size < 1 or max_length < 2:
        raise ValueError("Encoder batch size and text length must be positive")
    if set(directories) not in ({"bert", "mae", "clip"}, {"bert", "mae", "clip", "forensic"}):
        raise ValueError("Supply BERT, MAE and CLIP directories with an optional forensic CLIP")
    rows = normalize(rows)
    before = {name: checkpoint_digest(path) for name, path in directories.items()}
    tokenizer = AutoTokenizer.from_pretrained(directories["bert"], local_files_only=True)
    bert = AutoModel.from_pretrained(directories["bert"], local_files_only=True).to(device).eval()
    mae = ViTMAEModel.from_pretrained(directories["mae"], local_files_only=True).to(device).eval()
    mae.config.mask_ratio = 0.0
    image_processor = AutoImageProcessor.from_pretrained(directories["mae"], local_files_only=True)
    clip = AutoModel.from_pretrained(directories["clip"], local_files_only=True).to(device).eval()
    if not all(hasattr(clip, name) for name in ("get_text_features", "get_image_features")):
        raise ValueError("The CLIP directory must contain a supported vision-language encoder")
    clip_processor = AutoProcessor.from_pretrained(directories["clip"], local_files_only=True)
    models = {"bert": bert, "mae": mae, "clip": clip}
    forensic, forensic_processor = None, None
    if "forensic" in directories:
        forensic = AutoModel.from_pretrained(directories["forensic"], local_files_only=True).to(device).eval()
        if not hasattr(forensic, "get_image_features"):
            raise ValueError("The forensic directory must contain a compatible CLIP image encoder")
        forensic_processor = AutoProcessor.from_pretrained(directories["forensic"], local_files_only=True)
        models["forensic"] = forensic
    for model in models.values():
        model.requires_grad_(False)
    names = list(STREAMS) + (["forensic"] if forensic is not None else [])
    output, masks, image_hashes = {name: [] for name in names}, [], {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images, texts, mask = [], [row["text"] for row in batch], []
        for row in batch:
            present = bool(row["image"])
            if present:
                path = relative_path(row["image"])
                image_hashes.setdefault(path.as_posix(), digest(path))
                if row.get("content_sha256") and row["content_sha256"] != image_hashes[path.as_posix()]:
                    raise ValueError("The manifest image checksum is stale")
                row["content_sha256"] = image_hashes[path.as_posix()]
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            else:
                images.append(Image.new("RGB", (224, 224)))
            mask.append([bool(row["text"].strip()), present, bool(row["text"].strip()), present] + ([present] if forensic is not None else []))
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        pixels = image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        patch_size = mae.config.patch_size
        patches = (pixels.shape[-2] // patch_size) * (pixels.shape[-1] // patch_size)
        # Keep every MAE patch in its original order.
        noise = torch.arange(patches, device=device, dtype=torch.float32).expand(len(batch), -1)
        limit = min(max_length, clip.config.text_config.max_position_embeddings)
        clip_inputs = clip_processor(text=texts, images=images, padding=True, truncation=True, max_length=limit, return_tensors="pt")
        accepted = inspect.signature(clip.get_text_features).parameters
        clip_text = {key: value.to(device) for key, value in clip_inputs.items() if key in accepted and key != "pixel_values"}
        with torch.inference_mode():
            encoded = {"bert": bert(**inputs).last_hidden_state[:, 0], "mae": mae(pixel_values=pixels, noise=noise).last_hidden_state[:, 0], "clip_text": clip.get_text_features(**clip_text), "clip_image": clip.get_image_features(pixel_values=clip_inputs["pixel_values"].to(device))}
            if forensic is not None:
                forensic_pixels = forensic_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
                encoded["forensic"] = forensic.get_image_features(pixel_values=forensic_pixels)
        availability = np.asarray(mask, dtype=bool)
        for column, name in enumerate(names):
            value = getattr(encoded[name], "pooler_output", encoded[name])
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError("The encoder must return pooled feature tensors")
            values = value.detach().float().cpu().numpy()
            values[~availability[:, column]] = 0
            output[name].append(values)
        masks.append(availability)
    if not rows:
        raise ValueError("Embedding extraction requires nonempty input")
    if any(checkpoint_digest(directories[name]) != fingerprint for name, fingerprint in before.items()) or any(digest(path) != fingerprint for path, fingerprint in image_hashes.items()):
        raise ValueError("An encoder or image changed during extraction")
    provenance = {"encoders": {name: {"directory": directory, "sha256": before[name], "frozen": True, "model_type": models[name].config.model_type} for name, directory in directories.items()}, "device": device, "max_text_length": max_length, "mae_mask_ratio": 0.0, "image_sha256": image_hashes, "transformers_version": version("transformers"), "torch_version": str(torch.__version__)}
    result = {"features": {name: np.concatenate(values) for name, values in output.items()}, "observed": np.concatenate(masks), "rows": rows, "provenance": provenance}
    validate(result)
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
