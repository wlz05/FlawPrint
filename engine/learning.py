# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import platform
import random
import tempfile
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from engine import embeddings
from engine.conformal import calibrate_rows, apply_calibration, validate_labels
from engine.data import audit_splits
from engine.lineage import collect, ensure_disjoint, identity, FIELDS
from engine.metrics import binary
from engine.neural import NeuralConfig, ResidualAttention, BIASED
from engine.profile import DomainProfile, ProfileConfig
from engine.provenance import source_digest
from engine.storage import relative_path, subset, read_json, write_json, validate_bundle, digest
from engine.traces import NAMES

def groups(rows: list[dict]) -> np.ndarray:
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for field in FIELDS:
            value = identity(field, row.get(field))
            if value is not None:
                key = (field, value)
                if key in seen:
                    parents[find(index)] = find(seen[key])
                seen[key] = index
    return np.asarray([find(index) for index in range(len(rows))])

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
    learned = {} if state is None else state
    output = dict(packet, features={})
    for column, (name, values) in enumerate(packet["features"].items()):
        if state is None:
            indices = [index for index in training if packet["observed"][index, column]]
            if not indices:
                raise ValueError("Each encoded stream needs observed source training examples")
            selected = values[indices].astype(np.float64)
            learned[name] = {"center": selected.mean(0).tolist(), "scale": np.maximum(selected.std(0), 1e-6).tolist()}
        center, scale = np.asarray(learned[name]["center"]), np.asarray(learned[name]["scale"])
        if center.shape != (values.shape[1],) or scale.shape != center.shape or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
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

def logits(model: ResidualAttention, packet: dict, indices: list[int], device: str, size: int) -> np.ndarray:
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(indices), size):
            values.append(model(**batch(packet, indices[start:start + size], device))["logit"].detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0)

def scored(rows: list[dict], indices: list[int], risks: np.ndarray) -> list[dict]:
    return [dict(rows[index], fusion_risk=float(risk), prediction=int(risk >= 0), method="flawprint_residual_attention") for index, risk in zip(indices, risks)]

def signature(archive: dict) -> dict:
    return {name: {"sha256": value.get("sha256"), "frozen": value.get("frozen")} for name, value in archive["provenance"].get("encoders", {}).items()}

def publish(destination: Path, parameters: dict, profiles: dict, metadata: dict) -> None:
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
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise ValueError("The model output became occupied during training")
            destination.rmdir()
        # Publish the complete checkpoint without mixing separate runs.
        stage.rename(destination)

def fit(bundle: dict, archive: dict, output: str, config: NeuralConfig, profile_config: ProfileConfig, device: str = "cuda:0") -> dict:
    config.validate()
    profile_config.validate()
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
    if any(not values for values in partitions.values()):
        raise ValueError("Neural fitting requires train, validation and calibration partitions")
    source_domains = {rows[index]["domain"] for index in partitions["train"]}
    if any(rows[index]["domain"] not in source_domains for role in ("validation", "calibration") for index in partitions[role]):
        raise ValueError("Neural model selection and calibration require source domains")
    labels = {}
    for role, indices in partitions.items():
        labels[role] = validate_labels(np.asarray([rows[index].get("label") for index in indices]))
        if role != "calibration" and len(np.unique(labels[role])) != 2:
            raise ValueError("Source training and validation require both news classes")
    packet, normalizers = standardize(packet, partitions["train"])
    dimensions = {name: int(values.shape[1]) for name, values in packet["features"].items()}
    model = ResidualAttention(dimensions, len(bundle["names"]), config).to(device)
    optimizer = torch.optim.AdamW((value for value in model.parameters() if value.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
    rng, history, best, best_score, patience = np.random.default_rng(config.seed), [], None, -1.0, 0
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
            best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
        if patience >= config.patience:
            break
    model.load_state_dict(best)
    validation = scored(rows, partitions["validation"], logits(model, packet, partitions["validation"], device, config.batch_size))
    calibration = scored(rows, partitions["calibration"], logits(model, packet, partitions["calibration"], device, config.batch_size))
    state = calibrate_rows(validation, calibration)
    fit_rows = [rows[index] for role in ("train", "validation", "calibration") for index in partitions[role]]
    metadata = {"version": "flawprint-neural-v2", "relation_operator": "content-conditioned-reference-v2", "config": asdict(config), "dimensions": dimensions, "feature_order": list(dimensions), "trace_names": bundle["names"].tolist(), "normalizers": normalizers, "encoder_signature": signature(archive), "calibration": state, "fit_identifiers": [row["id"] for row in fit_rows], "fit_history": collect(fit_rows), "source_sha256": source_digest(), "selection_metric": "source validation macro F1", "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad), "history": history, "best_validation_macro_f1": best_score, "backbones": "Frozen local feature encoders; trainable projectors, attention and heads", "target_labels_used_for_fitting": False}
    metadata["environment"] = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "device": device, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == "cuda" else None}
    metadata["embedding_protocol"] = embeddings.protocol(archive)
    publish(destination, best, profiles, metadata)
    image_samples = sum(isinstance(rows[index].get("physical_manipulated"), (bool, np.bool_)) and bool(rows[index].get("image")) for index in partitions["train"]) if config.auxiliary_weight else 0
    return {"output": output, "epochs": len(history), "best_validation_macro_f1": best_score, "trainable_parameters": metadata["trainable_parameters"], "image_supervised_samples": image_samples}

def predict(bundle: dict, archive: dict, directory: str, split: str = "test", device: str = "cuda:0", reference_domain: str | None = None, explain: bool = False) -> list[dict]:
    validate_bundle(bundle)
    embeddings.validate(archive)
    device = embeddings.runtime(device)
    root = relative_path(directory)
    metadata = read_json(root / "model.json")
    if metadata.get("version") not in ("flawprint-neural-v1", "flawprint-neural-v2") or digest(root / "parameters.pt") != metadata.get("weights_sha256"):
        raise ValueError("The neural checkpoint version or weight hash is invalid")
    if metadata["version"] == "flawprint-neural-v2" and metadata.get("relation_operator") != "content-conditioned-reference-v2":
        raise ValueError("The neural checkpoint relation operator is unsupported")
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
    config = NeuralConfig(**metadata["config"])
    if metadata["version"] == "flawprint-neural-v1" and config.mode == "coupled":
        config = replace(config, mode="broadcast")
    dimensions = {name: metadata["dimensions"][name] for name in metadata["feature_order"]}
    model = ResidualAttention(dimensions, len(metadata["trace_names"]), config).to(device)
    model.load_state_dict(torch.load(root / "parameters.pt", map_location=device, weights_only=True))
    risks = logits(model, packet, list(range(len(rows))), device, config.batch_size)
    disabled = risks
    if explain and config.mode in BIASED:
        model.config = replace(config, mode="unbiased")
        disabled = logits(model, packet, list(range(len(rows))), device, config.batch_size)
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
            row["potential_risk_delta"] = float(risks[index] - disabled[index])
            row["intervention_scope"] = "Fixed checkpoint and relation tokens; attention potential disabled"
    return predictions

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
