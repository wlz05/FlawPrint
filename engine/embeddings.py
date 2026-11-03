# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import hashlib
import inspect
import io
import json
from importlib.metadata import version
from pathlib import Path, PureWindowsPath
from pickle import UnpicklingError
import re
from zipfile import BadZipFile
from zlib import error as CompressionError
import numpy as np
from engine.data import normalize, missing, row_namespace, portable_image
from engine.storage import relative_path, digest, native, parse_json, atomic_stream, validate_bundle

STREAMS = ("bert", "mae", "clip_text", "clip_image")
TEXT_PATTERN = '[\uff0c\u3002 :,.\uff1b|-\u201c\u201d\u2014\u2014_/nbsp+&;@\u3001\u300a\u300b\uff5e\uff08\uff09())#O\uff01\uff1a\u3010\u3011]'

def pool_summary(hidden, observed):
    import torch
    if not isinstance(hidden, torch.Tensor) or not isinstance(observed, torch.Tensor) or hidden.ndim != 3 or min(hidden.shape[1:]) < 1 or not hidden.is_floating_point() or observed.shape != hidden.shape[:2] or observed.dtype != torch.bool or observed.device != hidden.device:
        raise ValueError("Context pooling requires aligned hidden states and Boolean masks")
    values = hidden.masked_fill(~observed[:, :, None], 0)
    mean = values.sum(1) / observed.sum(1)[:, None].clamp_min(1)
    maximum = hidden.masked_fill(~observed[:, :, None], -torch.inf).amax(1)
    maximum = maximum.masked_fill(~observed.any(1)[:, None], 0)
    result = torch.cat((hidden[:, 0], mean, maximum), dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("Context pooling requires finite observed hidden states")
    return result

def clean_text(text: str, policy: str = "identity") -> str:
    if policy == "identity":
        return text
    if policy != "eann":
        raise ValueError("Unknown semantic text preprocessing policy")
    # Match the released EANN string cleaning rule.
    return re.sub(TEXT_PATTERN, "", text).strip().lower()

def processing_hash(policy: str) -> str:
    clean_text("example", policy)
    definition = "eann-clean-str-sst-v1|" + TEXT_PATTERN + "|strip|lower" if policy == "eann" else "identity"
    return hashlib.sha256(definition.encode("utf-8")).hexdigest()

def image_processing_hash(policy: str) -> str:
    if policy not in ("model", "ddin"):
        raise ValueError("Unknown MAE image preprocessing policy")
    rule = "pil-bilinear-short256-center224-rgb-float32-imagenet-v1" if policy == "ddin" else "pinned-model-image-processor"
    return hashlib.sha256(rule.encode()).hexdigest()

def mae_pixels(images):
    import torch
    from PIL import Image
    if not isinstance(images, list) or not images:
        raise ValueError("MAE preprocessing requires a nonempty image list")
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    scale = torch.tensor([.229, .224, .225])[:, None, None]
    output = []
    for image in images:
        if not isinstance(image, Image.Image) or min(image.size) < 1:
            raise ValueError("MAE preprocessing requires decoded PIL images")
        width, height = image.size
        size = (256, int(256 * height / width)) if width <= height else (int(256 * width / height), 256)
        resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        left, top = (int(round((value - 224) / 2.0)) for value in resized.size)
        # Match the DDIN MAE resize, center crop, and normalization.
        cropped = resized.crop((left, top, left + 224, top + 224))
        value = torch.from_numpy(np.array(cropped, dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div_(255)
        output.append(value.sub_(mean).div_(scale))
    return torch.stack(output)

def word_mask(inputs, tokenizer):
    mask = inputs["attention_mask"].bool().clone()
    for token in tokenizer.all_special_ids:
        mask &= inputs["input_ids"] != token
    return mask

def load_pooler(directory: str, device: str = "cpu"):
    import torch
    from engine.neural import TextPool
    root = relative_path(directory)
    config, parameters = root / "pooler.json", root / "pooler.pt"
    if not config.exists() and not parameters.exists():
        return None
    if not config.is_file() or not parameters.is_file():
        raise ValueError("Local text pooling requires its configuration and parameters")
    settings = parse_json(config.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict) or set(settings) != {"version", "hidden_size", "width"} or type(settings["version"]) is not int or settings["version"] != 1:
        raise ValueError("Unknown local text pooling configuration")
    sizes = (settings["hidden_size"], settings["width"])
    if any(type(value) is not int or value < 1 for value in sizes) or sizes[0] + 3 * sizes[1] > 8192:
        raise ValueError("Local text pooling dimensions exceed the feature contract")
    model = TextPool(*sizes)
    try:
        stored = torch.load(io.BytesIO(parameters.read_bytes()), map_location="cpu", weights_only=True)
    except (EOFError, UnpicklingError, RuntimeError, ValueError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Cannot decode local text pooling parameters") from error
    expected = model.state_dict()
    if not isinstance(stored, dict) or set(stored) != set(expected):
        raise ValueError("Local text pooling parameter names do not match")
    for name, value in stored.items():
        target = expected[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided or value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all():
            raise ValueError("Invalid local text pooling parameter: " + name)
    # Validate the complete state before allocating accelerator memory.
    model.load_state_dict(stored)
    return model.to(device).eval().requires_grad_(False)

def cliptext(model, inputs):
    if model.config.model_type == "chinese_clip":
        # Chinese CLIP projects its unpooled class token.
        hidden = model.text_model(**inputs, return_dict=True).last_hidden_state
        return model.text_projection(hidden[:, 0])
    return model.get_text_features(**inputs)

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
    settings.update({key: provenance[key] for key in ("text_processing", "text_processing_sha256", "image_processing", "image_processing_sha256") if key in provenance})
    return {"version": "flawprint-feature-protocol-v1", "settings": settings, "complete": all(key in settings for key in keys)}

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
    if "image_sha256" in archive["provenance"]:
        fingerprints = archive["provenance"]["image_sha256"]
        if not isinstance(fingerprints, dict):
            raise ValueError("Embedding image provenance requires a path-to-SHA256 mapping")
        images = {}
        for path, checksum in fingerprints.items():
            if not isinstance(path, str) or not path.strip() or not isinstance(checksum, str) or not re.fullmatch("[a-f0-9]{64}", checksum):
                raise ValueError("Embedding image provenance has an invalid path or SHA256")
            canonical = portable_image(path)
            if not canonical or canonical in images and images[canonical] != checksum:
                raise ValueError("Embedding image provenance contains conflicting path aliases")
            images[canonical] = checksum
        # Bind row metadata to the image bytes recorded during encoding.
        for row in rows:
            image = portable_image(row.get("image", ""))
            checksum = row.get("content_sha256")
            if image and (image not in images or checksum != images[image]):
                raise ValueError("Embedding rows disagree with their recorded image provenance")
            if not image and not missing(checksum):
                raise ValueError("Text-only embedding rows cannot declare an image fingerprint")
    encoders = archive["provenance"].get("encoders", {})
    expected = {"bert", "mae", "clip"} | ({"forensic"} if "forensic" in names else set())
    if not isinstance(encoders, dict) or set(encoders) != expected or any(not isinstance(value, dict) or value.get("frozen") is not True or not re.fullmatch("[0-9a-f]{64}", str(value.get("sha256", ""))) for value in encoders.values()):
        raise ValueError("Record a frozen checkpoint hash for every encoder")
    settings = protocol(archive)["settings"]
    if "image_processing" in settings or "image_processing_sha256" in settings:
        if not {"image_processing", "image_processing_sha256"}.issubset(settings) or settings["image_processing_sha256"] != image_processing_hash(settings["image_processing"]):
            raise ValueError("MAE preprocessing and its fingerprint must agree")
    if "text_processing" in settings or "text_processing_sha256" in settings:
        if settings.get("text_processing_sha256") != processing_hash(settings.get("text_processing")):
            raise ValueError("The text preprocessing definition differs from its recorded hash")
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
    validate_bundle(bundle)
    validate(archive)
    # Compare canonical metadata without rewriting either input cache.
    physical, cached = normalize(bundle["rows"]), normalize(archive["rows"])
    for original, canonical in ((bundle["rows"], physical), (archive["rows"], cached)):
        if any(first["id"] != second["id"] for first, second in zip(original, canonical)):
            raise ValueError("Cached identifiers must already use their canonical spelling")
    mapping = {row["id"]: index for index, row in enumerate(cached)}
    if set(mapping) != {row["id"] for row in physical}:
        raise ValueError("Deep and physical inputs must contain the same identities")
    order = []
    for row in physical:
        index = mapping[row["id"]]
        other = cached[index]
        if any(row.get(key) != other.get(key) for key in ("text", "image", "group", "domain", "split", "image_frame")):
            raise ValueError("Cached embeddings disagree with the physical input metadata")
        if row_namespace(row) != row_namespace(other):
            raise ValueError("Cached embeddings disagree with the benchmark contract")
        checksums = [None if missing(item.get("content_sha256")) else item["content_sha256"] for item in (row, other)]
        if any(value is not None for value in checksums) and (any(not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None for value in checksums) or checksums[0] != checksums[1]):
            raise ValueError("Deep and physical features describe different image bytes or lack matching SHA256 fingerprints")
        order.append(index)
    return {name: values[order] for name, values in archive["features"].items()}, archive["observed"][order]

def extract(rows: list[dict], directories: dict, batch_size: int = 64, device: str = "cuda:0", max_length: int = 128, text_processing: str = "identity", pooling: str = "cls", image_processing: str = "model") -> dict:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Encoder batch size and text length must be positive")
    context(max_length)
    image_processing_hash(image_processing)
    if pooling not in ("cls", "summary"):
        raise ValueError("Unknown contextual vector pooling policy")
    if set(directories) not in ({"bert", "mae", "clip"}, {"bert", "mae", "clip", "forensic"}):
        raise ValueError("Supply BERT, MAE and CLIP directories with an optional forensic CLIP")
    directories = {name: relative_path(path).as_posix() for name, path in directories.items()}
    rows = normalize(rows)
    if not rows:
        raise ValueError("Embedding extraction requires nonempty input")
    if any(not row["text"].strip() and not row["image"] for row in rows):
        raise ValueError("Embedding extraction requires text or an image for every sample")
    processed = [clean_text(row["text"], text_processing) for row in rows]
    if any(not text.strip() and not row["image"] for row, text in zip(rows, processed)):
        raise ValueError("Text preprocessing removed the sole observed input")
    before = {name: checkpoint_digest(path) for name, path in directories.items()}
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor, ViTMAEModel
    device = runtime(device)
    tokenizer = AutoTokenizer.from_pretrained(directories["bert"], local_files_only=True)
    bert = AutoModel.from_pretrained(directories["bert"], local_files_only=True).to(device).eval()
    pooler = load_pooler(directories["bert"], device)
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
        images, texts, mask = [], processed[start:start + len(batch)], []
        for row, text in zip(batch, texts):
            present = bool(row["image"])
            if present:
                path = relative_path(row["image"])
                image_hashes.setdefault(path.as_posix(), digest(path))
                if row.get("content_sha256") and row["content_sha256"] != image_hashes[path.as_posix()]:
                    raise ValueError("The manifest image checksum is stale")
                row["content_sha256"] = image_hashes[path.as_posix()]
                with Image.open(path) as image:
                    if getattr(image, "n_frames", 1) > 1 and row.get("image_frame") != 0:
                        raise ValueError("Animated or multipage inputs require an explicit still-image conversion")
                    image.seek(0)
                    images.append(image.convert("RGB"))
            else:
                images.append(Image.new("RGB", (224, 224)))
            mask.append([bool(text.strip()), present, bool(text.strip()), present] + ([present] if forensic is not None else []))
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=limits["bert"], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        pixels = mae_pixels(images).to(device) if image_processing == "ddin" else image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        patch_size = mae.config.patch_size
        patches = (pixels.shape[-2] // patch_size) * (pixels.shape[-1] // patch_size)
        # Keep every MAE patch in its original order.
        noise = torch.arange(patches, device=device, dtype=torch.float32).expand(len(batch), -1)
        clip_inputs = clip_processor(text=texts, images=images, padding=True, truncation=True, max_length=limits["clip"], return_tensors="pt")
        accepted = inspect.signature(clip.get_text_features).parameters
        clip_text = {key: value.to(device) for key, value in clip_inputs.items() if key in accepted and key != "pixel_values"}
        with torch.inference_mode():
            bert_hidden = bert(**inputs).last_hidden_state
            mae_hidden = mae(pixel_values=pixels, noise=noise).last_hidden_state
            encoded = {"bert": bert_hidden[:, 0], "mae": mae_hidden[:, 0], "clip_text": cliptext(clip, clip_text), "clip_image": clip.get_image_features(pixel_values=clip_inputs["pixel_values"].to(device))}
            if pooling == "summary":
                words = inputs["attention_mask"].bool().clone()
                for token in tokenizer.all_special_ids:
                    words &= inputs["input_ids"] != token
                patch_mask = torch.ones(mae_hidden.shape[:2], dtype=torch.bool, device=device)
                patch_mask[:, 0] = False
                # Keep word and patch summaries beside their class tokens.
                encoded["bert"] = pool_summary(bert_hidden, words)
                encoded["mae"] = pool_summary(mae_hidden, patch_mask)
            if pooler is not None:
                encoded["bert"] = pooler(bert_hidden, word_mask(inputs, tokenizer))
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
        if start // batch_size % 10 == 0 or start + len(batch) == len(rows):
            print(json.dumps({"stage": "encoding", "completed": start + len(batch), "total": len(rows)}), flush=True)
    if any(checkpoint_digest(directories[name]) != fingerprint for name, fingerprint in before.items()) or any(digest(path) != fingerprint for path, fingerprint in image_hashes.items()):
        raise ValueError("An encoder or image changed during extraction")
    provenance = {"encoders": {name: {"directory": directory, "sha256": before[name], "frozen": True, "model_type": models[name].config.model_type} for name, directory in directories.items()}, "device": device, "max_text_length": max_length, "mae_mask_ratio": 0.0, "image_sha256": image_hashes, "transformers_version": version("transformers"), "torch_version": str(torch.__version__)}
    if image_processing != "model":
        provenance.update(image_processing=image_processing, image_processing_sha256=image_processing_hash(image_processing), pillow_version=version("Pillow"))
    if text_processing != "identity":
        provenance.update(text_processing=text_processing, text_processing_sha256=processing_hash(text_processing))
    provenance.update(text_context=limits, readout={name: "class_token" if name in ("bert", "mae") else "projected_vector" for name in names})
    if pooling == "summary":
        provenance["readout"].update(bert="class_mean_max", mae="class_mean_max")
    if pooler is not None:
        provenance["readout"]["bert"] = "class_local_cnn"
    result = {"features": {name: np.concatenate(values) for name, values in output.items()}, "observed": np.concatenate(masks), "rows": rows, "provenance": provenance}
    validate(result)
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
