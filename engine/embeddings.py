# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path, PureWindowsPath
import re
from zipfile import BadZipFile
from zlib import error as CompressionError
import numpy as np
from engine.data import normalize, missing
from engine.storage import relative_path, digest, native, parse_json, atomic_stream

STREAMS = ("bert", "mae", "clip_text", "clip_image")

def runtime(device: str) -> str:
    import torch
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; install a CUDA-enabled PyTorch build or use --device cpu")
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise ValueError("The requested CUDA device index is unavailable")
    return str(selected)

def context(requested: int, *limits: int | None) -> int:
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 2:
        raise ValueError("Text length must be an integer of at least two tokens")
    available = [value for value in limits if value is not None]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in available):
        raise ValueError("Encoder context limits must be positive token counts")
    return min([requested] + available)

def protocol(archive: dict) -> dict:
    provenance = archive["provenance"]
    keys = ("max_text_length", "mae_mask_ratio", "text_context", "readout")
    settings = {key: provenance[key] for key in keys if key in provenance}
    return {"version": "flawprint-feature-protocol-v1", "settings": settings, "complete": len(settings) == len(keys)}

def checkpoint_digest(directory: str) -> str:
    root = relative_path(directory)
    if not root.is_dir():
        raise ValueError("A local encoder directory is missing")
    if root.is_symlink() or getattr(root, "is_junction", lambda: False)():
        raise ValueError("Copy complete encoder files into a local directory instead of linking a cache")
    entries = list(root.rglob("*"))
    if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)() for path in entries):
        raise ValueError("Copy complete encoder files into a local directory instead of linking a cache")
    # Preserve Windows checkpoint hashes across operating systems.
    files = sorted((path for path in entries if path.is_file()), key=lambda path: (PureWindowsPath(path.relative_to(root).as_posix()), path.relative_to(root).as_posix()))
    if not files:
        raise ValueError("An encoder directory is empty")
    checksum = hashlib.sha256()
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Encoder files must remain inside their checkpoint directory")
        checksum.update(path.relative_to(root).as_posix().encode() + b"\0" + digest(path).encode())
    return checksum.hexdigest()

def validate(archive: dict) -> None:
    if not isinstance(archive, dict) or not all(key in archive for key in ("rows", "features", "observed")):
        raise ValueError("An embedding archive requires rows, features and availability")
    rows, features, observed = archive["rows"], archive["features"], archive["observed"]
    if not isinstance(features, dict) or not isinstance(rows, list) or not isinstance(observed, np.ndarray):
        raise ValueError("Embedding arrays and metadata have invalid container types")
    names = list(features)
    if names not in (list(STREAMS), list(STREAMS) + ["forensic"]):
        raise ValueError("Embedding streams must use the recorded canonical order")
    if not rows or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip() for row in rows) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Embedding identities must be unique and nonempty")
    if observed.dtype != np.bool_ or observed.shape != (len(rows), len(names)):
        raise ValueError("Embedding availability requires a Boolean sample by stream mask")
    for column, name in enumerate(names):
        values = features[name]
        if not isinstance(values, np.ndarray) or not np.issubdtype(values.dtype, np.floating) or values.ndim != 2 or values.shape[0] != len(rows) or not 0 < values.shape[1] <= 8192 or not np.isfinite(values).all():
            raise ValueError("Embedding arrays require finite aligned feature vectors")
        if np.any(np.abs(values) > np.finfo(np.float32).max):
            raise ValueError("Embedding values must be representable as float32")
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
    settings = protocol(archive)["settings"]
    if "max_text_length" in settings:
        context(settings["max_text_length"])
    if "text_context" in settings:
        lengths = settings["text_context"]
        if not isinstance(lengths, dict) or set(lengths) != {"bert", "clip"}:
            raise ValueError("Record the effective BERT and CLIP context limits")
        for length in lengths.values():
            context(length)
            if "max_text_length" in settings and length > settings["max_text_length"]:
                raise ValueError("Effective context cannot exceed requested text length")
    if "mae_mask_ratio" in settings and (isinstance(settings["mae_mask_ratio"], bool) or not isinstance(settings["mae_mask_ratio"], (int, float)) or settings["mae_mask_ratio"] != 0):
        raise ValueError("Cached MAE features require every visible patch")
    if "readout" in settings and (not isinstance(settings["readout"], dict) or set(settings["readout"]) != set(names) or any(not isinstance(value, str) or not value.strip() for value in settings["readout"].values())):
        raise ValueError("Record the vector readout for every embedding stream")

def save(path: str, archive: dict) -> None:
    validate(archive)
    target = relative_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"feature_" + name: values.astype(np.float32) for name, values in archive["features"].items()}
    metadata = {"version": "flawprint-embeddings-v1", "streams": list(archive["features"]), "rows": archive["rows"], "provenance": archive["provenance"]}
    with atomic_stream(target, binary=True) as stream:
        np.savez_compressed(stream, **arrays, observed=archive["observed"], metadata=np.asarray(json.dumps(native(metadata), ensure_ascii=True, allow_nan=False)))

def load(path: str) -> dict:
    try:
        with relative_path(path).open("rb") as source, np.load(source, allow_pickle=False) as stream:
            serialized = stream["metadata"].item()
            if isinstance(serialized, bytes):
                serialized = serialized.decode("utf-8")
            if not isinstance(serialized, str):
                raise ValueError("Embedding metadata must contain a JSON string")
            metadata = parse_json(serialized)
            if not isinstance(metadata, dict) or metadata.get("version") != "flawprint-embeddings-v1":
                raise ValueError("Unsupported embedding archive version")
            if metadata.get("streams") not in (list(STREAMS), list(STREAMS) + ["forensic"]):
                raise ValueError("Embedding metadata requires canonical stream identities")
            archive = {"features": {name: stream["feature_" + name].copy() for name in metadata["streams"]}, "observed": stream["observed"].copy(), "rows": metadata["rows"], "provenance": metadata["provenance"]}
        validate(archive)
    except (BadZipFile, CompressionError, EOFError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid embedding archive: " + str(error)) from error
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
        checksums = [None if missing(item.get("content_sha256")) else item["content_sha256"] for item in (row, other)]
        if any(value is not None for value in checksums) and (any(not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None for value in checksums) or checksums[0] != checksums[1]):
            raise ValueError("Deep and physical features describe different image bytes or lack matching SHA256 fingerprints")
        order.append(index)
    return {name: values[order] for name, values in archive["features"].items()}, archive["observed"][order]

def extract(rows: list[dict], directories: dict, batch_size: int = 64, device: str = "cuda:0", max_length: int = 128) -> dict:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Encoder batch size and text length must be positive")
    context(max_length)
    if set(directories) not in ({"bert", "mae", "clip"}, {"bert", "mae", "clip", "forensic"}):
        raise ValueError("Supply BERT, MAE and CLIP directories with an optional forensic CLIP")
    directories = {name: relative_path(path).as_posix() for name, path in directories.items()}
    rows = normalize(rows)
    if not rows:
        raise ValueError("Embedding extraction requires nonempty input")
    if any(not row["text"].strip() and not row["image"] for row in rows):
        raise ValueError("Embedding extraction requires text or an image for every sample")
    before = {name: checkpoint_digest(path) for name, path in directories.items()}
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor, ViTMAEModel
    device = runtime(device)
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
    limits = {"bert": context(max_length, getattr(bert.config, "max_position_embeddings", None), getattr(tokenizer, "model_max_length", None)), "clip": context(max_length, clip.config.text_config.max_position_embeddings, getattr(clip_processor.tokenizer, "model_max_length", None))}
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
                    if getattr(image, "n_frames", 1) > 1:
                        raise ValueError("Animated or multipage inputs require an explicit still-image conversion")
                    images.append(image.convert("RGB"))
            else:
                images.append(Image.new("RGB", (224, 224)))
            mask.append([bool(row["text"].strip()), present, bool(row["text"].strip()), present] + ([present] if forensic is not None else []))
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=limits["bert"], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        pixels = image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        patch_size = mae.config.patch_size
        patches = (pixels.shape[-2] // patch_size) * (pixels.shape[-1] // patch_size)
        # Keep every MAE patch in its original order.
        noise = torch.arange(patches, device=device, dtype=torch.float32).expand(len(batch), -1)
        clip_inputs = clip_processor(text=texts, images=images, padding=True, truncation=True, max_length=limits["clip"], return_tensors="pt")
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
    if any(checkpoint_digest(directories[name]) != fingerprint for name, fingerprint in before.items()) or any(digest(path) != fingerprint for path, fingerprint in image_hashes.items()):
        raise ValueError("An encoder or image changed during extraction")
    provenance = {"encoders": {name: {"directory": directory, "sha256": before[name], "frozen": True, "model_type": models[name].config.model_type} for name, directory in directories.items()}, "device": device, "max_text_length": max_length, "mae_mask_ratio": 0.0, "image_sha256": image_hashes, "transformers_version": version("transformers"), "torch_version": str(torch.__version__)}
    provenance.update(text_context=limits, readout={name: "class_token" if name in ("bert", "mae") else "projected_vector" for name in names})
    result = {"features": {name: np.concatenate(values) for name, values in output.items()}, "observed": np.concatenate(masks), "rows": rows, "provenance": provenance}
    validate(result)
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
