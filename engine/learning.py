# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
from pickle import UnpicklingError
import platform
import random
import re
import tempfile
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from engine import embeddings
from engine.conformal import calibrate_rows, apply_calibration, validate_labels
from engine.data import audit_splits
from engine.lineage import collect, ensure_disjoint, components
from engine.metrics import binary
from engine.neural import NeuralConfig, ResidualAttention, BIASED
from engine.profile import DomainProfile, ProfileConfig
from engine.provenance import source_digest
from engine.storage import relative_path, subset, read_json, write_json, validate_bundle, digest
from engine.traces import NAMES

def groups(rows: list[dict]) -> np.ndarray:
    return np.asarray(components(rows))

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

def prepare(bundle: dict, archive: dict, config: ProfileConfig, folds: int, use_reference: bool = True) -> tuple[dict, dict]:
    validate_bundle(bundle)
    audit_splits(bundle["rows"])
    features, observed = embeddings.align(bundle, archive)
    if not use_reference:
        samples, dimension = len(bundle["rows"]), len(bundle["names"])
        return {"features": features, "observed": observed, "ranks": np.zeros((samples, dimension), dtype=np.float32), "graph": np.zeros((samples, dimension, dimension), dtype=np.float32), "trace_observed": np.zeros((samples, dimension), dtype=bool)}, {}
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
        for fit, query in GroupKFold(folds).split(training, groups=labels):
            fitting_indices = [training[index] for index in fit]
            querying = [training[index] for index in query]
            reference = fit_reference(fitting_indices, domain)
            if reference is not None:
                assign(querying, reference)
    packet = {"features": features, "observed": observed, "ranks": ranks, "graph": graph, "trace_observed": trace_mask}
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
                for name in ("attention_contrast_rms", "attention_shift_tv", "pair_relation_fraction", "pair_contrast_rms", "pair_attention_shift_tv"):
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
    try:
        # Validate stored tensors before moving the model to the GPU.
        stored = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, UnpicklingError, RuntimeError, ValueError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Cannot decode neural checkpoint weights") from error
    if digest(path) != checksum:
        raise ValueError("The neural checkpoint weights changed while loading")
    expected = model.state_dict()
    if not isinstance(stored, dict) or set(stored) != set(expected):
        raise ValueError("Neural checkpoint parameter names do not match the model")
    for name, value in stored.items():
        target = expected[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError("Neural checkpoint parameter shape or type is invalid: " + name)
        if not torch.isfinite(value).all():
            raise ValueError("Neural checkpoint parameters must be finite: " + name)
    model.load_state_dict(stored)

def publish(destination: Path, parameters: dict, profiles: dict, metadata: dict, provenance: dict | None = None) -> None:
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
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise ValueError("The model output became occupied during training")
            destination.rmdir()
        # Publish the complete checkpoint without mixing separate runs.
        stage.rename(destination)

def fit(bundle: dict, archive: dict, output: str, config: NeuralConfig, profile_config: ProfileConfig, device: str = "cuda:0", provenance: dict | None = None) -> dict:
    config.validate()
    profile_config.validate()
    if provenance is not None:
        if not isinstance(provenance, dict) or not isinstance(provenance.get("inputs"), list) or any(not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str) or re.fullmatch("[0-9a-f]{64}", item["sha256"]) is None for item in provenance["inputs"]):
            raise ValueError("Training provenance requires input paths and SHA256 strings")
        provenance = copy.deepcopy(provenance)
        if any(digest(item["path"]) != item["sha256"] for item in provenance["inputs"]):
            raise ValueError("A training input changed before fitting")
    destination = relative_path(output)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Choose a new or empty model output directory")
    device = embeddings.runtime(device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    packet, profiles = prepare(bundle, archive, profile_config, config.folds, config.mode != "deep")
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
    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(partitions["train"]).tolist()
        total, auxiliary_count = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            indices = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            predicted = model(**batch(packet, indices, device))
            target = torch.as_tensor([rows[index]["label"] for index in indices], dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted["logit"], target)
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
        history.append({"epoch": epoch + 1, "loss": total / len(order), "validation_macro_f1": validation["macro_f1"], "image_supervised_samples": auxiliary_count})
        if validation["macro_f1"] > best_score:
            best_score, patience = validation["macro_f1"], 0
            selected_epoch = epoch + 1
            best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
        if patience >= config.patience:
            break
    model.load_state_dict(best)
    validation = scored(rows, partitions["validation"], logits(model, packet, partitions["validation"], device, config.batch_size))
    calibration = scored(rows, partitions["calibration"], logits(model, packet, partitions["calibration"], device, config.batch_size))
    state = calibrate_rows(validation, calibration) if calibration else None
    fit_rows = [rows[index] for role in ("train", "validation", "calibration") for index in partitions[role]]
    metadata = {"version": "flawprint-neural-v2", "relation_operator": "content-conditioned-reference-v2", "config": asdict(config), "dimensions": dimensions, "feature_order": list(dimensions), "trace_names": bundle["names"].tolist(), "normalizers": normalizers, "encoder_signature": signature(archive), "calibration": state, "fit_identifiers": [row["id"] for row in fit_rows], "fit_history": collect(fit_rows), "source_sha256": source_digest(), "selection_metric": "source validation macro F1", "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad), "history": history, "best_validation_macro_f1": best_score, "backbones": "Frozen local feature encoders; trainable projectors, attention and heads", "target_labels_used_for_fitting": False}
    metadata["environment"] = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "device": device, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == "cuda" else None}
    metadata["embedding_protocol"] = embeddings.protocol(archive)
    metadata["partition_counts"] = {role: sum(row["split"] == role for row in rows) for role in ("train", "validation", "calibration", "profile", "test")}
    metadata["partition_policy"] = "Preserve supplied sample identities and partition roles"
    metadata["selected_epoch"] = selected_epoch
    metadata["calibration_protocol"] = "independent_source_partitions" if state is not None else "uncalibrated"
    publish(destination, best, profiles, metadata, provenance)
    image_samples = sum(isinstance(rows[index].get("physical_manipulated"), (bool, np.bool_)) and bool(rows[index].get("image")) for index in partitions["train"]) if config.auxiliary_weight else 0
    return {"output": output, "epochs": len(history), "selected_epoch": selected_epoch, "best_validation_macro_f1": best_score, "trainable_parameters": metadata["trainable_parameters"], "image_supervised_samples": image_samples, "calibration_samples": len(partitions["calibration"])}

def predict(bundle: dict, archive: dict, directory: str, split: str = "test", device: str = "cuda:0", reference_domain: str | None = None, explain: bool = False) -> list[dict]:
    validate_bundle(bundle)
    embeddings.validate(archive)
    device = embeddings.runtime(device)
    root = relative_path(directory)
    metadata = read_json(root / "model.json")
    config = checkpoint(metadata)
    if digest(root / "parameters.pt") != metadata["weights_sha256"]:
        raise ValueError("The neural checkpoint weight hash is invalid")
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
    ensure_disjoint(rows, metadata["fit_history"])
    selected = subset(bundle, indices)
    packet = {"features": {name: value[indices] for name, value in features.items()}, "observed": observed[indices], "ranks": np.zeros_like(selected["values"], dtype=np.float32), "graph": np.zeros((len(rows), len(bundle["names"]), len(bundle["names"])), dtype=np.float32), "trace_observed": np.zeros_like(selected["observed"])}
    for domain in sorted({row["domain"] for row in rows}):
        reference = reference_domain or domain
        if reference not in metadata["profiles"]:
            if reference_domain is not None:
                raise ValueError("The explicitly requested reference is not saved")
            continue
        path = root / relative_path(metadata["profiles"][reference])
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != metadata["profile_sha256"][reference]:
            raise ValueError("The saved physical reference changed")
        profile = DomainProfile.load(path.as_posix())
        selected_indices = [index for index, row in enumerate(rows) if row["domain"] == domain]
        ranks, graph, mask = relation(profile, subset(selected, selected_indices))
        packet["ranks"][selected_indices], packet["graph"][selected_indices], packet["trace_observed"][selected_indices] = ranks, graph, mask
    packet, _ = standardize(packet, [], metadata["normalizers"])
    if metadata["version"] == "flawprint-neural-v1" and config.mode == "coupled":
        config = replace(config, mode="broadcast")
    dimensions = {name: metadata["dimensions"][name] for name in metadata["feature_order"]}
    model = ResidualAttention(dimensions, len(metadata["trace_names"]), config)
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
            for name in ("attention_contrast_rms", "attention_shift_tv", "pair_relation_fraction", "pair_contrast_rms", "pair_attention_shift_tv"):
                row[name] = float(measurements[name][index]) if name in measurements else None
            row["intervention_scope"] = "Fixed checkpoint and nodes; local attention contrasts and potential-off, uniform, and summary readouts"
    return predictions

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
