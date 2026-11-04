# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
from pickle import UnpicklingError
import platform
import random
import re
import tempfile
import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from engine import embeddings
from engine.conformal import calibrate_rows, apply_calibration, validate_labels
from engine.data import audit_splits, SPLITS, SPLIT_ALIASES
from engine.lineage import collect, ensure_disjoint, components
from engine.metrics import binary
from engine.neural import NeuralConfig, ResidualAttention, BIASED
from engine.profile import DomainProfile, ProfileConfig
from engine.provenance import source_digest
from engine.storage import relative_path, resolved, subset, read_json, write_json, validate_bundle, digest, native, atomic_stream, protect_outputs
from engine.traces import NAMES

def agreement(features: dict, observed: np.ndarray) -> np.ndarray:
    # Reuse stable shared-space normalization for cosine agreement.
    return pairing(features, observed)[:, -2:]

def pairing(features: dict, observed: np.ndarray) -> np.ndarray:
    if not isinstance(features, dict) or not {"clip_text", "clip_image"}.issubset(features):
        raise ValueError("Pairing requires both CLIP feature streams")
    names = list(features)
    first, second = (np.asarray(features[name], dtype=np.float64) for name in ("clip_text", "clip_image"))
    if first.ndim != 2 or first.shape != second.shape or first.shape[1] < 1 or not isinstance(observed, np.ndarray) or observed.dtype != np.bool_ or observed.shape != (len(first), len(names)):
        raise ValueError("Pairing requires aligned CLIP dimensions and Boolean masks")
    flags = [observed[:, names.index(name)] for name in ("clip_text", "clip_image")]
    vectors, available = [], []
    for values, present in zip((first, second), flags):
        values = np.where(present[:, None], values, 0)
        if not np.isfinite(values).all():
            raise ValueError("Observed CLIP pairing features must be finite")
        scale = np.abs(values).max(-1, keepdims=True)
        scaled = values / np.maximum(scale, np.finfo(np.float64).tiny)
        norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
        vectors.append(scaled / np.maximum(norm, np.finfo(np.float64).tiny))
        available.append(present & (scale[:, 0] > 0))
    valid = available[0] & available[1]
    # Retain shared-space pair structure before separate stream transforms.
    product = vectors[0] * vectors[1]
    difference = np.abs(vectors[0] - vectors[1])
    payload = np.column_stack((product, difference, np.clip(product.sum(-1), -1, 1)))
    payload[~valid] = 0
    return np.column_stack((payload, valid)).astype(np.float32)

def pairing_tensor(text: torch.Tensor, image: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if not isinstance(text, torch.Tensor) or not isinstance(image, torch.Tensor) or not isinstance(observed, torch.Tensor) or text.ndim != 2 or text.shape[1] < 1 or image.shape != text.shape or not text.is_floating_point() or image.dtype != text.dtype or image.device != text.device or observed.dtype != torch.bool or observed.shape != (len(text), 2) or observed.device != text.device:
        raise ValueError("Live pairing requires aligned CLIP tensors and Boolean masks")
    vectors, available = [], []
    for column, values in enumerate((text, image)):
        values = values.double().masked_fill(~observed[:, column, None], 0)
        if not torch.isfinite(values).all():
            raise ValueError("Observed live CLIP features must be finite")
        scale = values.abs().amax(-1, keepdim=True)
        scaled = values / scale.clamp_min(torch.finfo(values.dtype).tiny)
        norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        vectors.append(scaled / norm.clamp_min(torch.finfo(values.dtype).tiny))
        available.append(observed[:, column] & (scale[:, 0] > 0))
    valid = available[0] & available[1]
    product = vectors[0] * vectors[1]
    payload = torch.cat((product, (vectors[0] - vectors[1]).abs(), product.sum(-1, keepdim=True).clamp(-1, 1)), dim=-1)
    payload = payload.masked_fill(~valid[:, None], 0)
    return torch.cat((payload, valid[:, None].to(payload.dtype)), dim=-1).to(text.dtype)

def groups(rows: list[dict]) -> np.ndarray:
    return np.asarray(components(rows))

def reference_folds(rows: list[dict], folds: int, seed: int, balanced: bool = False, min_samples: int = 20):
    identifiers = groups(rows)
    if not balanced:
        return list(GroupKFold(folds).split(rows, groups=identifiers))
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Reference balancing requires training rows only")
    labels = validate_labels(np.asarray([row.get("label") for row in rows]))
    unique, inverse = np.unique(identifiers, return_inverse=True)
    if folds == 2 and 2 <= len(unique) <= 16:
        sizes = np.bincount(inverse)
        positives = np.bincount(inverse, weights=labels)
        total, positive = len(rows), labels.sum()
        minimum = max(min_samples, int(np.ceil(total * .2)))
        best, selected = None, None
        # Preserve the source mixture without splitting related news groups.
        for bits in range((1 << (len(unique) - 1)) - 1):
            assignment = np.asarray([True] + [bool(bits & (1 << index)) for index in range(len(unique) - 1)])
            count = int(sizes[assignment].sum())
            if min(count, total - count) < minimum:
                continue
            first = positives[assignment].sum()
            drift = max(abs(first / count - positive / total), abs((positive - first) / (total - count) - positive / total))
            objective = (float(drift), abs(2 * count - total))
            if best is None or objective < best:
                best, selected = objective, assignment[inverse]
        if selected is not None:
            first, second = np.flatnonzero(selected), np.flatnonzero(~selected)
            return [(first, second), (second, first)]
    return list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(rows, labels, groups=identifiers))

def relation(profile: DomainProfile, bundle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if any(profile.overlap(bundle["rows"])):
        raise ValueError("A neural query overlaps its fitted physical reference")
    transformed = profile.transform(bundle)
    precision = np.linalg.solve(profile.reference, np.eye(len(profile.reference)))
    # Preserve the pair contributions of the copula quadratic form.
    local = (np.eye(len(precision)) - precision)[None] * transformed[:, :, None] * transformed[:, None, :]
    samples, dimension = len(bundle["rows"]), len(bundle["names"])
    ranks, graph = np.zeros((samples, dimension)), np.zeros((samples, dimension, dimension))
    observed = np.zeros((samples, dimension), dtype=bool)
    columns = np.asarray(profile.columns)
    ranks[:, columns] = transformed
    observed[:, columns] = bundle["observed"][:, columns]
    for index, first in enumerate(columns):
        graph[:, first, columns] = local[:, index]
    graph *= observed[:, :, None] * observed[:, None, :]
    image_columns = [index for index, name in enumerate(bundle["names"]) if str(name) in profile.names and str(name) in NAMES[:20]]
    ready = np.asarray([bool(row.get("image")) for row in bundle["rows"]]) & (observed[:, image_columns].sum(1) >= 2)
    graph[~ready] = 0
    ranks[~observed] = 0
    return ranks.astype(np.float32), graph.astype(np.float32), observed

def canonical_bundle(bundle: dict) -> dict:
    validate_bundle(bundle)
    return dict(bundle, rows=embeddings.canonical_rows(bundle["rows"]))

def prepare(bundle: dict, archive: dict, config: ProfileConfig, folds: int, use_reference: bool = True, use_alignment: bool = False, balanced: bool = False, use_pairing: bool = False) -> tuple[dict, dict]:
    bundle = canonical_bundle(bundle)
    audit_splits(bundle["rows"])
    features, observed = embeddings.align(bundle, archive)
    aligned = {"alignment": agreement(features, observed)} if use_alignment else {}
    if use_pairing:
        aligned["pairing"] = pairing(features, observed)
    if not use_reference:
        samples, dimension = len(bundle["rows"]), len(bundle["names"])
        return {**aligned, "features": features, "observed": observed, "ranks": np.zeros((samples, dimension), dtype=np.float32), "graph": np.zeros((samples, dimension, dimension), dtype=np.float32), "trace_observed": np.zeros((samples, dimension), dtype=bool)}, {}
    profiles = {}

    def fit_reference(indices: list[int], domain: str) -> DomainProfile | None:
        if len(indices) < config.min_samples:
            return None
        try:
            return DomainProfile.fit(subset(bundle, indices), domain, config)
        except ValueError as error:
            if str(error) == "A copula profile requires two nonconstant observed features":
                return None
            raise

    for domain in sorted({row["domain"] for row in bundle["rows"]}):
        source = [index for index, row in enumerate(bundle["rows"]) if row["domain"] == domain and row["split"] == "train"]
        target = [index for index, row in enumerate(bundle["rows"]) if row["domain"] == domain and row["split"] == "profile"]
        if source and target:
            raise ValueError("A target reference domain cannot also supply source training rows")
        reference = fit_reference(source or target, domain)
        if reference is not None:
            profiles[domain] = reference
    samples, dimension = len(bundle["rows"]), len(bundle["names"])
    ranks, graph, trace_mask = np.zeros((samples, dimension), dtype=np.float32), np.zeros((samples, dimension, dimension), dtype=np.float32), np.zeros((samples, dimension), dtype=bool)

    def assign(indices: list[int], profile: DomainProfile) -> None:
        first, second, third = relation(profile, subset(bundle, indices))
        ranks[indices], graph[indices], trace_mask[indices] = first, second, third

    for domain in sorted({row["domain"] for row in bundle["rows"]}):
        development = [index for index, row in enumerate(bundle["rows"]) if row["domain"] == domain and row["split"] in ("validation", "calibration", "test")]
        if development and domain in profiles:
            assign(development, profiles[domain])
        training = [index for index, row in enumerate(bundle["rows"]) if row["domain"] == domain and row["split"] == "train"]
        if not training:
            continue
        labels = groups([bundle["rows"][index] for index in training])
        if len(np.unique(labels)) < folds:
            continue
        # Fit each training reference without its query groups.
        for fit, query in reference_folds([bundle["rows"][index] for index in training], folds, config.seed, balanced, config.min_samples):
            fitting_indices = [training[index] for index in fit]
            querying = [training[index] for index in query]
            reference = fit_reference(fitting_indices, domain)
            if reference is not None:
                assign(querying, reference)
    packet = {"features": features, "observed": observed, "ranks": ranks, "graph": graph, "trace_observed": trace_mask}
    packet.update(aligned)
    return packet, profiles

def standardize(packet: dict, training: list[int], state: dict | None = None) -> tuple[dict, dict]:
    if state is not None and (not isinstance(state, dict) or set(state) != set(packet["features"])):
        raise ValueError("The source feature standardizer must match every encoded stream")
    learned = {} if state is None else state
    output = dict(packet, features={})
    for column, (name, values) in enumerate(packet["features"].items()):
        if state is None:
            indices = [index for index in training if packet["observed"][index, column]]
            if not indices:
                raise ValueError("Each encoded stream needs observed source training examples")
            selected = values[indices].astype(np.float64)
            learned[name] = {"center": selected.mean(0).tolist(), "scale": np.maximum(selected.std(0), 1e-6).tolist()}
        settings = learned[name]
        if not isinstance(settings, dict) or set(settings) != {"center", "scale"}:
            raise ValueError("The source feature standardizer requires a center and scale")
        center, scale = np.asarray(settings["center"]), np.asarray(settings["scale"])
        if center.dtype.kind not in "fiu" or scale.dtype.kind not in "fiu" or center.shape != (values.shape[1],) or scale.shape != center.shape or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("The source feature standardizer is invalid")
        normalized = (values.astype(np.float64) - center) / scale
        normalized[~packet["observed"][:, column]] = 0
        if not np.isfinite(normalized).all() or np.any(np.abs(normalized) > np.finfo(np.float32).max):
            raise ValueError("Standardized embeddings exceed the finite float32 range")
        normalized = normalized.astype(np.float32)
        output["features"][name] = normalized
    return output, learned

def batch(packet: dict, indices: list[int], device: str) -> dict:
    output = {name: torch.as_tensor(packet[name][indices], device=device) for name in ("observed", "ranks", "graph", "trace_observed")}
    output["features"] = {name: torch.as_tensor(values[indices], device=device) for name, values in packet["features"].items()}
    if "alignment" in packet:
        output["alignment"] = torch.as_tensor(packet["alignment"][indices], device=device)
    if "pairing" in packet:
        output["pairing"] = torch.as_tensor(packet["pairing"][indices], device=device)
    return output

def logits(model: ResidualAttention, packet: dict, indices: list[int], device: str, size: int, diagnostics: dict | None = None) -> np.ndarray:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Inference batch size must be a positive integer")
    model.eval()
    values, measurements = [], {}
    with torch.inference_mode():
        for start in range(0, len(indices), size):
            selected = indices[start:start + size]
            output = model(**batch(packet, selected, device), explain=diagnostics is not None)
            if not isinstance(output, dict) or not isinstance(output.get("logit"), torch.Tensor):
                raise ValueError("Neural inference requires one logit tensor per batch")
            # Check sample alignment before combining inference batches.
            risk = output["logit"].detach().cpu().numpy()
            if risk.shape != (len(selected),) or risk.dtype.kind != "f" or not np.isfinite(risk).all():
                raise ValueError("Neural inference requires one finite risk per selected sample")
            values.append(risk)
            if diagnostics is not None:
                for name in ("attention_contrast_rms", "attention_shift_tv", "pair_relation_fraction", "pair_contrast_rms", "pair_attention_shift_tv", "value_transport_rms", "pair_value_transport_rms", "pair_tangent_transport_rms", "pair_transport_remainder_bound"):
                    if name in output:
                        if not isinstance(output[name], torch.Tensor):
                            raise ValueError("Neural diagnostics require tensors: " + name)
                        value = output[name].detach().cpu().numpy()
                        if value.shape != risk.shape or value.dtype.kind != "f" or not np.isfinite(value).all() or np.any(value < 0) or name in ("attention_shift_tv", "pair_relation_fraction", "pair_attention_shift_tv") and np.any(value > 1):
                            raise ValueError("Neural diagnostics require finite aligned values: " + name)
                        measurements.setdefault(name, []).append(value)
    if diagnostics is not None:
        combined = {name: np.concatenate(parts) for name, parts in measurements.items()}
        if any(len(value) != len(indices) for value in combined.values()):
            raise ValueError("Neural diagnostics must cover every selected sample")
        diagnostics.update(combined)
    return np.concatenate(values) if values else np.empty(0)

def scored(rows: list[dict], indices: list[int], risks: np.ndarray) -> list[dict]:
    risks = np.asarray(risks)
    if risks.shape != (len(indices),) or risks.dtype.kind not in "fiu" or not np.isfinite(risks).all():
        raise ValueError("Scoring requires one finite risk per selected sample")
    return [dict(rows[index], fusion_risk=float(risk), prediction=int(risk >= 0), method="flawprint_residual_attention") for index, risk in zip(indices, risks)]

def signature(archive: dict) -> dict:
    return {name: {"sha256": value.get("sha256"), "frozen": value.get("frozen")} for name, value in archive["provenance"].get("encoders", {}).items()}

def checkpoint(metadata: dict) -> NeuralConfig:
    required = {"version", "weights_sha256", "config", "dimensions", "feature_order", "trace_names", "normalizers", "encoder_signature", "calibration", "fit_identifiers", "fit_history", "profiles", "profile_sha256"}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError("Neural checkpoint metadata is missing required fields")
    if metadata["version"] not in ("flawprint-neural-v1", "flawprint-neural-v2"):
        raise ValueError("The neural checkpoint version is unsupported")
    if metadata["version"] == "flawprint-neural-v2" and metadata.get("relation_operator") != "content-conditioned-reference-v2":
        raise ValueError("The neural checkpoint relation operator is unsupported")
    mappings = ("config", "dimensions", "normalizers", "encoder_signature", "fit_history", "profiles", "profile_sha256")
    if any(not isinstance(metadata[name], dict) for name in mappings) or metadata["calibration"] is not None and not isinstance(metadata["calibration"], dict) or "embedding_protocol" in metadata and not isinstance(metadata["embedding_protocol"], dict):
        raise ValueError("Neural checkpoint settings must contain objects")
    order, dimensions, names = (metadata[name] for name in ("feature_order", "dimensions", "trace_names"))
    if order not in (list(embeddings.STREAMS), list(embeddings.STREAMS) + ["forensic"]) or set(dimensions) != set(order) or any(isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 8192 for value in dimensions.values()):
        raise ValueError("Neural checkpoint streams or dimensions are invalid")
    if set(metadata["normalizers"]) != set(order):
        raise ValueError("Neural checkpoint normalizers must cover every stream")
    for name in order:
        normalizer = metadata["normalizers"][name]
        if not isinstance(normalizer, dict) or set(normalizer) != {"center", "scale"}:
            raise ValueError("Neural checkpoint normalizers require a center and scale")
        center, scale = (np.asarray(normalizer[key]) for key in ("center", "scale"))
        if (center.shape != (dimensions[name],) or scale.shape != center.shape
                or center.dtype.kind not in "fiu" or scale.dtype.kind not in "fiu"
                or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError("Neural checkpoint normalizer dimensions or values are invalid")
    if not isinstance(names, list) or len(names) < 2 or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Neural checkpoint trace names must be unique nonempty strings")
    identifiers = metadata["fit_identifiers"]
    if not isinstance(identifiers, list) or not identifiers or any(not isinstance(value, str) or not value.strip() for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Neural checkpoint fitting identifiers must be unique nonempty strings")
    profiles, hashes = metadata["profiles"], metadata["profile_sha256"]
    if set(profiles) != set(hashes) or any(not isinstance(domain, str) or not domain.strip() or not isinstance(path, str) or not path.strip() for domain, path in profiles.items()):
        raise ValueError("Neural checkpoint reference paths and hashes must agree")
    if any(not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None for value in [metadata["weights_sha256"], *hashes.values()]):
        raise ValueError("Neural checkpoint hashes must be lowercase SHA256 strings")
    try:
        config = NeuralConfig(**metadata["config"])
    except TypeError as error:
        raise ValueError("Neural checkpoint configuration fields are invalid") from error
    config.validate()
    ensure_disjoint([], metadata["fit_history"])
    apply_calibration([], metadata["calibration"])
    return config

def parameters(model: ResidualAttention, path: Path, checksum: str) -> None:
    path = relative_path(path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != checksum:
        raise ValueError("The neural checkpoint weights changed while loading")
    try:
        # Validate stored tensors before moving the model to the GPU.
        stored = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except (EOFError, UnpicklingError, RuntimeError, ValueError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Cannot decode neural checkpoint weights") from error
    if digest(path) != checksum:
        raise ValueError("The neural checkpoint weights changed while loading")
    validate_parameters(model, stored)
    model.load_state_dict(stored)

def validate_parameters(model: ResidualAttention, stored: dict) -> None:
    expected = model.state_dict()
    if not isinstance(stored, dict) or set(stored) != set(expected):
        raise ValueError("Neural checkpoint parameter names do not match the model")
    for name, value in stored.items():
        target = expected[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError("Neural checkpoint parameter shape or type is invalid: " + name)
        if not torch.isfinite(value).all():
            raise ValueError("Neural checkpoint parameters must be finite: " + name)

def training_protocol(bundle: dict, archive: dict, config: NeuralConfig, profile_config: ProfileConfig, device: str, implementation: str) -> dict:
    from engine.experiment import bundle_digest
    bundle = canonical_bundle(bundle)
    features, observed = embeddings.align(bundle, archive)
    metadata = {"rows": sorted(embeddings.canonical_rows(archive["rows"]), key=lambda row: row["id"]), "encoders": signature(archive), "protocol": embeddings.protocol(archive)}
    checksum = hashlib.sha256(json.dumps(native(metadata), sort_keys=True, ensure_ascii=True, allow_nan=False).encode())
    # Bind resume identity to aligned vectors rather than archive row order.
    for name, values in [("observed", observed), *features.items()]:
        array = np.ascontiguousarray(values)
        checksum.update(name.encode() + str(array.shape).encode() + array.dtype.str.encode())
        checksum.update(array.tobytes())
    runtime = {"torch": str(torch.__version__), "numpy": np.__version__, "device_type": torch.device(device).type, "default_dtype": str(torch.get_default_dtype()), "matmul_precision": torch.get_float32_matmul_precision(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    return native({"source_sha256": implementation, "physical_sha256": bundle_digest(bundle), "embedding_sha256": checksum.hexdigest(), "config": asdict(config), "profile_config": asdict(profile_config), "runtime": runtime})

def read_training(path: Path, checksum: str, protocol: dict, model: ResidualAttention, optimizer, rng) -> dict:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != checksum:
        raise ValueError("The training state changed while loading")
    try:
        stored = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except (EOFError, UnpicklingError, RuntimeError, ValueError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Cannot decode the training state") from error
    required = {"version", "protocol", "epoch", "model", "optimizer", "best", "best_score", "selected_epoch", "waiting", "history", "rng", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"}
    if not isinstance(stored, dict) or set(stored) != required or stored["version"] != "flawprint-training-v1":
        raise ValueError("The training state format is invalid")
    if stored["protocol"] != protocol:
        raise ValueError("The training state protocol differs from the current run")
    epoch, selected, waiting = (stored[name] for name in ("epoch", "selected_epoch", "waiting"))
    if any(type(value) is not int for value in (epoch, selected, waiting)) or not 1 <= selected <= epoch <= protocol["config"]["epochs"] or not 0 <= waiting <= protocol["config"]["patience"]:
        raise ValueError("The training state epoch counters are invalid")
    history = stored["history"]
    if not isinstance(history, list) or len(history) != epoch:
        raise ValueError("The training state history is incomplete")
    for index, row in enumerate(history, 1):
        if not isinstance(row, dict) or type(row.get("epoch")) is not int or row["epoch"] != index:
            raise ValueError("The training state history has inconsistent epochs")
        if type(row.get("image_supervised_samples")) is not int or row["image_supervised_samples"] < 0:
            raise ValueError("The training state history has invalid supervision counts")
        for name in ("loss", "validation_macro_f1", "validation_accuracy", "validation_auc"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or name != "loss" and value > 1:
                raise ValueError("The training state history contains invalid metrics")
    scores = [row["validation_macro_f1"] for row in history]
    if isinstance(stored["best_score"], bool) or not isinstance(stored["best_score"], (int, float)) or stored["best_score"] != max(scores) or selected != scores.index(max(scores)) + 1 or waiting != epoch - selected:
        raise ValueError("The training state selection history is inconsistent")
    validate_parameters(model, stored["model"])
    validate_parameters(model, stored["best"])
    previous, expected = stored["optimizer"], optimizer.state_dict()
    if not isinstance(previous, dict) or set(previous) != {"state", "param_groups"} or not isinstance(previous["state"], dict):
        raise ValueError("The training state optimizer is invalid")
    try:
        actual_groups = json.dumps(native(previous["param_groups"]), sort_keys=True, allow_nan=False)
        expected_groups = json.dumps(native(expected["param_groups"]), sort_keys=True, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("The training state optimizer settings are invalid") from error
    if actual_groups != expected_groups:
        raise ValueError("The training state optimizer settings changed")
    identifiers = {key: parameter for saved, group in zip(expected["param_groups"], optimizer.param_groups) for key, parameter in zip(saved["params"], group["params"])}
    if set(previous["state"]) - set(identifiers):
        raise ValueError("The training state optimizer has unknown parameters")
    for key, fields in previous["state"].items():
        if not isinstance(fields, dict) or set(fields) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("The training state optimizer tensors are incomplete")
        for name, value in fields.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.device.type != "cpu" or value.layout != torch.strided or not torch.isfinite(value).all():
                raise ValueError("The training state optimizer tensors are invalid")
            if name == "step":
                if value.numel() != 1 or value.item() < 0 or value.item() != int(value.item()):
                    raise ValueError("The training state optimizer step is invalid")
            elif value.shape != identifiers[key].shape or value.dtype != identifiers[key].dtype:
                raise ValueError("The training state optimizer tensor shape or type changed")
            elif name == "exp_avg_sq" and torch.any(value < 0):
                raise ValueError("The training state optimizer variance is negative")
    numpy_rng = stored["numpy_rng"]
    if not isinstance(numpy_rng, (tuple, list)) or len(numpy_rng) != 5:
        raise ValueError("The training state NumPy random state is invalid")
    algorithm, keys, position, gaussian, cached = numpy_rng
    if algorithm != "MT19937" or not isinstance(keys, torch.Tensor) or keys.device.type != "cpu" or keys.layout != torch.strided or keys.dtype != torch.int64 or keys.shape != (624,) or torch.any(keys < 0) or torch.any(keys >= 2 ** 32) or type(position) is not int or not 0 <= position <= 624 or type(gaussian) is not int or gaussian not in (0, 1) or isinstance(cached, bool) or not isinstance(cached, (int, float)) or not math.isfinite(cached):
        raise ValueError("The training state NumPy random state is invalid")
    cuda_rng = stored["cuda_rng"]
    if protocol["runtime"]["device_type"] == "cuda":
        if not isinstance(cuda_rng, torch.Tensor) or cuda_rng.device.type != "cpu" or cuda_rng.dtype != torch.uint8 or cuda_rng.ndim != 1:
            raise ValueError("The training state CUDA random state is invalid")
    elif cuda_rng is not None:
        raise ValueError("A CPU training state cannot contain CUDA random state")
    # Check random states without mutating the running model or generators.
    try:
        candidate = copy.deepcopy(rng)
        candidate.bit_generator.state = stored["rng"]
        random.Random(0).setstate(stored["python_rng"])
        np.random.RandomState(0).set_state((algorithm, keys.cpu().numpy().astype(np.uint32), position, gaussian, cached))
        torch.Generator(device="cpu").set_state(stored["torch_rng"])
        if cuda_rng is not None:
            torch.Generator(device=next(model.parameters()).device).set_state(cuda_rng)
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        raise ValueError("The training state contains invalid random generators") from error
    if digest(path) != checksum:
        raise ValueError("The training state changed while loading")
    try:
        model.load_state_dict(stored["model"])
        optimizer.load_state_dict(previous)
        rng.bit_generator.state = stored["rng"]
        random.setstate(stored["python_rng"])
        np.random.set_state((algorithm, keys.cpu().numpy().astype(np.uint32), position, gaussian, cached))
        torch.set_rng_state(stored["torch_rng"])
        if stored["cuda_rng"] is not None:
            torch.cuda.set_rng_state(stored["cuda_rng"], next(model.parameters()).device)
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        raise ValueError("The training state cannot restore optimizer or random states") from error
    return stored

def artifacts(root: Path, metadata: dict, checksum: str) -> dict[Path, str]:
    # Require the complete saved reference set after a model transfer.
    files = {root / "model.json": checksum, root / "parameters.pt": metadata["weights_sha256"]}
    for domain, filename in metadata["profiles"].items():
        path = root / relative_path(filename)
        if path in files:
            raise ValueError("Neural checkpoint artifact paths must be distinct")
        files[path] = metadata["profile_sha256"][domain]
    paths = [path.resolve() for path in files]
    if len(set(paths)) != len(paths):
        raise ValueError("Neural checkpoint artifact paths must be distinct")
    if any(not path.is_file() or not path.resolve().is_relative_to(root.resolve()) for path in files):
        raise ValueError("The neural checkpoint is incomplete or has files outside its directory")
    if any(digest(path) != expected for path, expected in files.items()):
        raise ValueError("The neural checkpoint artifact hash is invalid")
    return files

def publish(destination: Path, parameters: dict, profiles: dict, metadata: dict, provenance: dict | None = None) -> None:
    if "source_sha256" in metadata and source_digest() != metadata["source_sha256"]:
        raise ValueError("The source implementation changed before checkpoint publication")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".flawprint-") as temporary:
        stage = relative_path(os.path.relpath(temporary))
        paths, hashes = {}, {}
        for domain, profile in profiles.items():
            filename = hashlib.sha256(domain.encode()).hexdigest() + ".json"
            path = stage / "profiles" / filename
            profile.save(path.as_posix())
            paths[domain], hashes[domain] = "profiles/" + filename, digest(path)
        weights = stage / "parameters.pt"
        torch.save(parameters, weights)
        metadata = dict(metadata, profiles=paths, profile_sha256=hashes, weights_sha256=digest(weights))
        write_json(stage / "model.json", metadata)
        if provenance is not None:
            write_json(stage / "provenance.json", provenance)
            if any(digest(item["path"]) != item["sha256"] for item in provenance["inputs"]):
                raise ValueError("An input changed during training; the checkpoint was not published")
        if "source_sha256" in metadata and source_digest() != metadata["source_sha256"]:
            raise ValueError("The source implementation changed before checkpoint publication")
        prepared = destination.exists()
        if prepared:
            if not destination.is_dir() or any(destination.iterdir()):
                raise ValueError("The model output became occupied during training")
        try:
            if prepared:
                destination.rmdir()
            # Publish the complete checkpoint without mixing separate runs.
            stage.rename(destination)
        except BaseException:
            if prepared and not destination.exists():
                destination.mkdir(exist_ok=True)
            raise

def fit(bundle: dict, archive: dict, output: str, config: NeuralConfig, profile_config: ProfileConfig, device: str = "cuda:0", provenance: dict | None = None, state_path: str | None = None, resume: bool = False) -> dict:
    if not isinstance(config, NeuralConfig) or not isinstance(profile_config, ProfileConfig):
        raise ValueError("Training requires neural and reference configurations")
    config, profile_config = copy.deepcopy(config), copy.deepcopy(profile_config)
    config.validate()
    profile_config.validate()
    if type(resume) is not bool or resume and state_path is None:
        raise ValueError("Resume requires a relative training state path")
    bundle = canonical_bundle(bundle)
    implementation = source_digest()
    if provenance is not None:
        if not isinstance(provenance, dict) or not isinstance(provenance.get("inputs"), list) or any(not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str) or re.fullmatch("[0-9a-f]{64}", item["sha256"]) is None for item in provenance["inputs"]):
            raise ValueError("Training provenance requires input paths and SHA256 strings")
        provenance = copy.deepcopy(provenance)
        if "source_sha256" in provenance and provenance["source_sha256"] != implementation:
            raise ValueError("Training provenance describes a different source implementation")
        if any(digest(item["path"]) != item["sha256"] for item in provenance["inputs"]):
            raise ValueError("A training input changed before fitting")
    destination = relative_path(output)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Choose a new or empty model output directory")
    training_file = relative_path(state_path) if state_path is not None else None
    state_checksum = None
    if training_file is not None:
        validate_bundle(bundle)
        if training_file.suffix.lower() not in (".pt", ".pth"):
            raise ValueError("A training state file must use the pt or pth extension")
        if resolved(training_file).is_relative_to(resolved(destination)) or resolved(destination).is_relative_to(resolved(training_file)):
            raise ValueError("The training state must remain outside the model directory")
        inputs = [item["path"] for item in provenance["inputs"]] if provenance is not None else []
        inputs.extend(row["image"] for row in bundle["rows"] if row.get("image"))
        protect_outputs([training_file], inputs)
        if resume:
            if not training_file.is_file():
                raise ValueError("A complete training state file is required for resume")
            state_checksum = digest(training_file)
        elif training_file.exists():
            raise ValueError("Choose a fresh training state file or enable resume")
    device = embeddings.runtime(device)
    protocol = training_protocol(bundle, archive, config, profile_config, device, implementation) if training_file is not None else None
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    packet, profiles = prepare(bundle, archive, profile_config, config.folds, config.mode != "deep", config.aligned, config.balanced_reference, config.paired)
    rows = bundle["rows"]
    partitions = {role: [index for index, row in enumerate(rows) if row["split"] == role] for role in ("train", "validation", "calibration")}
    if any(not partitions[role] for role in ("train", "validation")):
        raise ValueError("Neural fitting requires train and validation partitions")
    source_domains = {rows[index]["domain"] for index in partitions["train"]}
    if any(rows[index]["domain"] not in source_domains for role in ("validation", "calibration") for index in partitions[role]):
        raise ValueError("Neural model selection and calibration require source domains")
    labels = {}
    for role, indices in partitions.items():
        if not indices:
            continue
        labels[role] = validate_labels(np.asarray([rows[index].get("label") for index in indices]))
        if role != "calibration" and len(np.unique(labels[role])) != 2:
            raise ValueError("Source training and validation require both news classes")
    packet, normalizers = standardize(packet, partitions["train"])
    dimensions = {name: int(values.shape[1]) for name, values in packet["features"].items()}
    model = ResidualAttention(dimensions, len(bundle["names"]), config).to(device)
    optimizer = torch.optim.AdamW((value for value in model.parameters() if value.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
    rng, history, best, best_score, patience, selected_epoch = np.random.default_rng(config.seed), [], None, -1.0, 0, 0
    start_epoch = 0
    if resume:
        saved = read_training(training_file, state_checksum, protocol, model, optimizer, rng)
        start_epoch, history, best, best_score, patience, selected_epoch = (saved[name] for name in ("epoch", "history", "best", "best_score", "waiting", "selected_epoch"))
        print(json.dumps({"stage": "resuming", "seed": config.seed, "completed_epochs": start_epoch}), flush=True)
    for epoch in range(start_epoch, config.epochs):
        if patience >= config.patience:
            break
        model.train()
        order = rng.permutation(partitions["train"]).tolist()
        total, auxiliary_count = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            indices = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            predicted = model(**batch(packet, indices, device))
            target = torch.as_tensor([rows[index]["label"] for index in indices], dtype=torch.float32, device=device)
            smoothed = target * (1 - config.label_smoothing) + config.label_smoothing / 2
            loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted["logit"], smoothed)
            available = [position for position, index in enumerate(indices) if isinstance(rows[index].get("physical_manipulated"), (bool, np.bool_)) and rows[index].get("image")]
            if available and config.auxiliary_weight:
                # Image supervision uses manipulation labels rather than news labels.
                image_target = torch.as_tensor([float(rows[indices[position]]["physical_manipulated"]) for position in available], device=device)
                loss = loss + config.auxiliary_weight * torch.nn.functional.binary_cross_entropy_with_logits(predicted["image_logit"][available], image_target)
                auxiliary_count += len(available)
            if not torch.isfinite(loss):
                raise ValueError("Neural training produced a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        risks = logits(model, packet, partitions["validation"], device, config.batch_size)
        validation = binary(labels["validation"], risks)
        history.append({"epoch": epoch + 1, "loss": total / len(order), "validation_macro_f1": validation["macro_f1"], "validation_accuracy": validation["accuracy"], "validation_auc": validation["auc"], "image_supervised_samples": auxiliary_count})
        if validation["macro_f1"] > best_score:
            best_score, patience = validation["macro_f1"], 0
            selected_epoch = epoch + 1
            best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
        if training_file is not None:
            if source_digest() != implementation:
                raise ValueError("The source implementation changed during training")
            if (digest(training_file) if training_file.exists() else None) != state_checksum:
                raise ValueError("The training state changed before publication")
            optimizer_state = optimizer.state_dict()
            optimizer_state["param_groups"] = native(optimizer_state["param_groups"])
            algorithm, keys, position, gaussian, cached = np.random.get_state()
            numpy_rng = (algorithm, torch.as_tensor(keys.astype(np.int64), dtype=torch.int64, device="cpu"), int(position), int(gaussian), float(cached))
            snapshot = {"version": "flawprint-training-v1", "protocol": protocol, "epoch": epoch + 1, "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, "optimizer": optimizer_state, "best": best, "best_score": float(best_score), "selected_epoch": selected_epoch, "waiting": patience, "history": native(history), "rng": rng.bit_generator.state, "python_rng": random.getstate(), "numpy_rng": numpy_rng, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(torch.device(device)) if torch.device(device).type == "cuda" else None}
            # Commit complete epochs without replacing the previous state on failure.
            with atomic_stream(training_file, binary=True) as stream:
                torch.save(snapshot, stream)
            state_checksum = digest(training_file)
        print(json.dumps({"stage": "training", "seed": config.seed, **history[-1]}, allow_nan=False), flush=True)
        if patience >= config.patience:
            break
    model.load_state_dict(best)
    validation = scored(rows, partitions["validation"], logits(model, packet, partitions["validation"], device, config.batch_size))
    calibration = scored(rows, partitions["calibration"], logits(model, packet, partitions["calibration"], device, config.batch_size))
    state = calibrate_rows(validation, calibration) if calibration else None
    fit_rows = [rows[index] for role in ("train", "validation", "calibration") for index in partitions[role]]
    if source_digest() != implementation:
        raise ValueError("The source implementation changed during training")
    metadata = {"version": "flawprint-neural-v2", "relation_operator": "content-conditioned-reference-v2", "config": asdict(config), "dimensions": dimensions, "feature_order": list(dimensions), "trace_names": bundle["names"].tolist(), "normalizers": normalizers, "encoder_signature": signature(archive), "calibration": state, "fit_identifiers": [row["id"] for row in fit_rows], "fit_history": collect(fit_rows), "source_sha256": implementation, "selection_metric": "source validation macro F1", "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad), "history": history, "best_validation_macro_f1": best_score, "backbones": "Frozen local feature encoders; trainable projectors, attention and heads", "target_labels_used_for_fitting": False}
    if training_file is not None:
        if digest(training_file) != state_checksum:
            raise ValueError("The training state changed before model publication")
        metadata["training_state"] = {"path": training_file.as_posix(), "sha256": state_checksum, "resumed_from_epoch": start_epoch, "completed_epochs": len(history)}
    metadata["environment"] = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "device": device, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == "cuda" else None}
    metadata["embedding_protocol"] = embeddings.protocol(archive)
    metadata["partition_counts"] = {role: sum(row["split"] == role for row in rows) for role in ("train", "validation", "calibration", "profile", "test")}
    metadata["partition_policy"] = "Preserve supplied sample identities and partition roles"
    policies = {row.get("benchmark_protocol", "strict-content") for row in rows}
    metadata["benchmark_protocols"] = sorted(policies)
    if policies == {"finefake-paper-3407"}:
        metadata["paper_content_overlap"] = {name: len(values) for name, values in audit_splits(rows)["leakage"].items()}
    metadata["reference_partition_policy"] = "Source group partitions with class proportion preservation" if config.balanced_reference else "GroupKFold source partitions"
    metadata["selected_epoch"] = selected_epoch
    metadata["calibration_protocol"] = "independent_source_partitions" if state is not None else "uncalibrated"
    publish(destination, best, profiles, metadata, provenance)
    image_samples = sum(isinstance(rows[index].get("physical_manipulated"), (bool, np.bool_)) and bool(rows[index].get("image")) for index in partitions["train"]) if config.auxiliary_weight else 0
    return {"output": output, "epochs": len(history), "selected_epoch": selected_epoch, "best_validation_macro_f1": best_score, "trainable_parameters": metadata["trainable_parameters"], "image_supervised_samples": image_samples, "calibration_samples": len(partitions["calibration"])}

def predict(bundle: dict, archive: dict, directory: str, split: str = "test", device: str = "cuda:0", reference_domain: str | None = None, explain: bool = False) -> list[dict]:
    if reference_domain is not None and (not isinstance(reference_domain, str) or not reference_domain.strip()):
        raise ValueError("An explicit reference domain must be a nonempty name")
    if not isinstance(split, str):
        raise ValueError("The prediction split must be a canonical role or supported alias")
    split = split.strip().casefold()
    split = SPLIT_ALIASES.get(split, split)
    if split not in SPLITS:
        raise ValueError("The prediction split must be a canonical role or supported alias")
    bundle = canonical_bundle(bundle)
    embeddings.validate(archive)
    device = embeddings.runtime(device)
    root = relative_path(directory)
    document = root / "model.json"
    if not document.is_file() or not document.resolve().is_relative_to(root.resolve()):
        raise ValueError("The neural checkpoint metadata is missing or outside its directory")
    checksum = digest(document)
    metadata = read_json(document, checksum)
    if digest(document) != checksum:
        raise ValueError("The neural checkpoint metadata changed while loading")
    config = checkpoint(metadata)
    files = artifacts(root, metadata, checksum)
    if bundle["names"].tolist() != metadata["trace_names"] or signature(archive) != metadata["encoder_signature"]:
        raise ValueError("The query encoder or physical feature schema changed")
    if "embedding_protocol" in metadata and embeddings.protocol(archive) != metadata["embedding_protocol"]:
        raise ValueError("The query embedding preprocessing protocol changed")
    features, observed = embeddings.align(bundle, archive)
    if list(features) != metadata["feature_order"] or {name: value.shape[1] for name, value in features.items()} != metadata["dimensions"]:
        raise ValueError("The query embedding dimensions or stream order changed")
    indices = [index for index, row in enumerate(bundle["rows"]) if row["split"] == split]
    if not indices:
        raise ValueError("The neural query selection is empty")
    rows = [bundle["rows"][index] for index in indices]
    if set(metadata["fit_identifiers"]) & {row["id"] for row in rows}:
        raise ValueError("Neural evaluation identities overlap fitting or calibration")
    expected_protocols = set(metadata.get("benchmark_protocols", ["strict-content"]))
    if {row.get("benchmark_protocol", "strict-content") for row in rows} != expected_protocols:
        raise ValueError("Prediction rows must use the checkpoint benchmark protocol")
    ensure_disjoint(rows, metadata["fit_history"])
    references = {}
    # Validate the complete reference set before any model execution.
    for reference, filename in metadata["profiles"].items():
        path = root / relative_path(filename)
        reference_checksum = metadata["profile_sha256"][reference]
        profile = DomainProfile.load(path.as_posix(), reference_checksum)
        if digest(path) != reference_checksum:
            raise ValueError("The saved physical reference changed while loading")
        if profile.domain != reference:
            raise ValueError("The saved physical reference domain does not match its checkpoint key")
        if (max(profile.columns) >= len(metadata["trace_names"])
                or [metadata["trace_names"][column] for column in profile.columns] != profile.names):
            raise ValueError("The saved physical reference feature schema differs from the checkpoint")
        references[reference] = profile
    selected = subset(bundle, indices)
    packet = {"features": {name: value[indices] for name, value in features.items()}, "observed": observed[indices], "ranks": np.zeros_like(selected["values"], dtype=np.float32), "graph": np.zeros((len(rows), len(bundle["names"]), len(bundle["names"])), dtype=np.float32), "trace_observed": np.zeros_like(selected["observed"])}
    if config.aligned:
        packet["alignment"] = agreement(packet["features"], packet["observed"])
    if config.paired:
        packet["pairing"] = pairing(packet["features"], packet["observed"])
    for domain in sorted({row["domain"] for row in rows}):
        reference = reference_domain or domain
        if reference not in metadata["profiles"]:
            if reference_domain is not None:
                raise ValueError("The explicitly requested reference is not saved")
            continue
        profile = references[reference]
        selected_indices = [index for index, row in enumerate(rows) if row["domain"] == domain]
        ranks, graph, mask = relation(profile, subset(selected, selected_indices))
        packet["ranks"][selected_indices], packet["graph"][selected_indices], packet["trace_observed"][selected_indices] = ranks, graph, mask
    packet, _ = standardize(packet, [], metadata["normalizers"])
    if metadata["version"] == "flawprint-neural-v1" and config.mode == "coupled":
        config = replace(config, mode="broadcast")
    dimensions = {name: metadata["dimensions"][name] for name in metadata["feature_order"]}
    # Preserve caller randomness while restoring the saved float32 model.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = ResidualAttention(dimensions, len(metadata["trace_names"]), config).float()
    parameters(model, root / "parameters.pt", metadata["weights_sha256"])
    model.to(device)
    measurements = {} if explain else None
    risks = logits(model, packet, list(range(len(rows))), device, config.batch_size, measurements)
    disabled, uniform = risks, risks if config.mode == "uniform" else None
    summarized = risks if config.mode == "summary" else None
    if explain:
        try:
            if config.mode in BIASED:
                model.config = replace(config, mode="unbiased")
                disabled = logits(model, packet, list(range(len(rows))), device, config.batch_size)
            if config.mode == "coupled":
                # Change only the responsibility rule at the fixed checkpoint.
                model.config = replace(config, mode="uniform")
                uniform = logits(model, packet, list(range(len(rows))), device, config.batch_size)
                # Retain the same query rule while removing unsummarized pair information.
                model.config = replace(config, mode="summary")
                summarized = logits(model, packet, list(range(len(rows))), device, config.batch_size)
        finally:
            model.config = config
    predictions = apply_calibration(scored(rows, list(range(len(rows))), risks), metadata["calibration"])
    for index, row in enumerate(predictions):
        row["model_mode"] = config.mode
        row["checkpoint_version"] = metadata["version"]
        row["checkpoint_sha256"] = checksum
        row["relation_operator"] = metadata.get("relation_operator", "broadcast-reference-v1")
        row["embedding_protocol_verified"] = bool(metadata.get("embedding_protocol", {}).get("complete"))
        row["reference_domain"] = reference_domain or row["domain"]
        row["physical_reference_available"] = row["reference_domain"] in metadata["profiles"]
        if explain:
            row["potential_disabled_risk"] = float(disabled[index])
            row["potential_risk_delta"] = float(risks[index]) - float(disabled[index])
            row["uniform_readout_risk"] = float(uniform[index]) if uniform is not None else None
            row["query_readout_delta"] = float(risks[index]) - float(uniform[index]) if uniform is not None else None
            row["summary_readout_risk"] = float(summarized[index]) if summarized is not None else None
            row["pair_readout_delta"] = float(risks[index]) - float(summarized[index]) if summarized is not None else None
            for name in ("attention_contrast_rms", "attention_shift_tv", "pair_relation_fraction", "pair_contrast_rms", "pair_attention_shift_tv", "value_transport_rms", "pair_value_transport_rms", "pair_tangent_transport_rms", "pair_transport_remainder_bound"):
                row[name] = float(measurements[name][index]) if name in measurements else None
            row["intervention_scope"] = "Fixed checkpoint and nodes; local attention contrasts and potential-off, uniform, and summary readouts"
    # Bind every returned prediction to one complete checkpoint snapshot.
    if any(not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected for path, expected in files.items()):
        raise ValueError("The neural checkpoint artifacts changed during inference")
    return predictions

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
