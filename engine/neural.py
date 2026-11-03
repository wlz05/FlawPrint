# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import dataclass
import copy
import math
from numbers import Real
import torch
from torch import nn

MODES = ("coupled", "unbiased", "independent", "deep", "concat", "uniform", "broadcast", "diagonal", "shuffled", "summary")
BIASED = ("coupled", "uniform", "broadcast", "diagonal", "shuffled", "summary")

def summarize(relation: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if not isinstance(relation, torch.Tensor) or not isinstance(observed, torch.Tensor) or relation.ndim != 3 or relation.shape[-1] < 1 or relation.shape[-1] != relation.shape[-2] or not relation.is_floating_point() or observed.dtype != torch.bool or observed.device != relation.device or observed.shape != relation.shape[:2]:
        raise ValueError("Summary reconstruction requires square relations and aligned Boolean masks")
    pairs = observed[:, :, None] & observed[:, None, :]
    values = relation.masked_fill(~pairs, 0)
    if not torch.isfinite(values).all() or not torch.allclose(values, values.transpose(-1, -2)):
        raise ValueError("Summary reconstruction requires finite symmetric relations")
    diagonal = values.diagonal(dim1=-2, dim2=-1)
    residual = values.sum(-1) - diagonal
    count = observed.sum(-1).to(relation.dtype)[:, None, None]
    # Recover the minimum-norm relation fixed by its diagonal and row sums.
    off_diagonal = (residual[:, :, None] + residual[:, None, :] - residual.sum(-1)[:, None, None] / (count - 1).clamp_min(1)) / (count - 2).clamp_min(1)
    active = pairs & ~torch.eye(relation.shape[-1], device=relation.device, dtype=torch.bool)[None]
    reconstructed = torch.diag_embed(diagonal) + off_diagonal.masked_fill(~active, 0)
    result = torch.where(count <= 2, values, reconstructed)
    if not torch.isfinite(result).all():
        raise ValueError("Summary reconstruction overflowed; rescale the relation")
    return result

def scramble(relation: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(relation.shape[-1], device=relation.device).expand(len(relation), -1).clone()
    for index, mask in enumerate(observed):
        active = torch.where(mask)[0]
        indices[index, active] = active.roll(1)
    # Reassign relation identities while preserving the active spectrum.
    return relation.gather(1, indices[:, :, None].expand_as(relation)).gather(2, indices[:, None, :].expand_as(relation))

def contrast(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor) or not isinstance(observed, torch.Tensor) or values.ndim != 4 or not values.is_floating_point() or values.shape[-2] != values.shape[-1] or observed.shape != (values.shape[0], values.shape[-1]) or observed.dtype != torch.bool or observed.device != values.device:
        raise ValueError("Attention contrasts require square head matrices and a Boolean node mask")
    keys = observed[:, None, None, :]
    count = observed.sum(-1)[:, None, None, None].clamp_min(1)
    # Remove the row offset that cancels in softmax.
    center = values.masked_fill(~keys, 0).sum(-1, keepdim=True) / count
    return (values - center).masked_fill(~(keys & observed[:, None, :, None]), 0)

def transport(current: torch.Tensor, reference: torch.Tensor, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if not all(isinstance(value, torch.Tensor) for value in (current, reference, values, observed)) or current.ndim != 4 or current.shape[-1] < 1 or current.shape[-2] != current.shape[-1] or reference.shape != current.shape or values.ndim != 4 or values.shape[:3] != current.shape[:3] or values.shape[-1] < 1:
        raise ValueError("Value transport requires aligned attention and value tensors")
    if not current.is_floating_point() or reference.dtype != current.dtype or values.dtype != current.dtype or reference.device != current.device or values.device != current.device or observed.device != current.device or observed.dtype != torch.bool or observed.shape != (current.shape[0], current.shape[-1]):
        raise ValueError("Value transport requires shared floating types and Boolean node masks")
    pairs = observed[:, None, :, None] & observed[:, None, None, :]
    current, reference = current.masked_fill(~pairs, 0), reference.masked_fill(~pairs, 0)
    values = values.masked_fill(~observed[:, None, :, None], 0)
    expected = observed[:, None].expand(current.shape[:3]).to(current.dtype)
    if not all(torch.isfinite(value).all() for value in (current, reference, values)) or any(torch.any(value < 0) or not torch.allclose(value.sum(-1), expected, atol=1e-5, rtol=1e-4) for value in (current, reference)):
        raise ValueError("Value transport requires finite values and normalized attention")
    # Remove a shared value offset before measuring evidence redistribution.
    first = observed.to(torch.int64).argmax(-1)
    anchor = values.gather(2, first[:, None, None, None].expand(values.shape[0], values.shape[1], 1, values.shape[-1]))
    result = (current - reference) @ (values - anchor)
    if not torch.isfinite(result).all():
        raise ValueError("Value transport overflowed; rescale the values")
    return result.masked_fill(~observed[:, None, :, None], 0)

def sensitivity(reference: torch.Tensor, direction: torch.Tensor, values: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    transport(reference, reference, values, observed)
    if not isinstance(direction, torch.Tensor) or direction.shape != reference.shape or direction.dtype != reference.dtype or direction.device != reference.device:
        raise ValueError("Transport sensitivity requires aligned floating logit changes")
    pairs = observed[:, None, :, None] & observed[:, None, None, :]
    direction = direction.masked_fill(~pairs, 0)
    reference = reference.masked_fill(~pairs, 0)
    values = values.masked_fill(~observed[:, None, :, None], 0)
    if not torch.isfinite(direction).all():
        raise ValueError("Observed logit changes must be finite")
    first = observed.to(torch.int64).argmax(-1)
    anchor = values.gather(2, first[:, None, None, None].expand(values.shape[0], values.shape[1], 1, values.shape[-1]))
    origin = direction.gather(-1, first[:, None, None, None].expand(*direction.shape[:-1], 1))
    # Center logits and values before computing their weighted covariance.
    centered = (direction - origin).masked_fill(~pairs, 0)
    centered = centered - (reference * centered).sum(-1, keepdim=True)
    tangent = (reference * centered) @ (values - anchor)
    distances = torch.linalg.vector_norm(values[:, :, :, None] - values[:, :, None, :], dim=-1)
    diameter = distances.masked_fill(~pairs, 0).amax(dim=(-1, -2))
    maximum = direction.masked_fill(~pairs, -torch.inf).amax(-1)
    minimum = direction.masked_fill(~pairs, torch.inf).amin(-1)
    span = (maximum - minimum).masked_fill(~observed[:, None], 0)
    bound = diameter[:, :, None] * span.square() / 8
    tangent = tangent.masked_fill(~observed[:, None, :, None], 0)
    if not torch.isfinite(tangent).all() or not torch.isfinite(bound).all():
        raise ValueError("Transport sensitivity overflowed; rescale logits and values")
    return tangent, bound

def potential(logits: torch.Tensor, observed: torch.Tensor, relation: torch.Tensor, mode: str, unit: bool = False) -> torch.Tensor:
    if not all(isinstance(value, torch.Tensor) for value in (logits, observed, relation)) or logits.ndim != 4 or logits.shape[-1] != logits.shape[-2] or not logits.is_floating_point() or relation.ndim != 3 or relation.dtype != logits.dtype or relation.device != logits.device or observed.dtype != torch.bool or observed.device != logits.device:
        raise ValueError("Attention potentials require square floating matrices and a Boolean node mask")
    if type(unit) is not bool:
        raise ValueError("Coordinate normalization must be Boolean")
    batch, heads, length, _ = logits.shape
    traces = relation.shape[-1]
    prefix = length - traces
    if prefix < 1 or traces < 1 or relation.shape != (batch, traces, traces) or observed.shape != (batch, length):
        raise ValueError("Relation coordinates must match the attention sequence")
    mask = observed[:, prefix:]
    relation = relation.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0)
    if mode == "broadcast":
        values = relation.sum(-1) / math.sqrt(traces)
        output = logits.new_zeros((batch, 1, length, length))
        output[:, :, prefix:, prefix:] = relation[:, None]
        output[:, :, :prefix, prefix:] = values[:, None, None, :]
        output[:, :, prefix:, :prefix] = values[:, None, :, None]
        return output.expand(-1, heads, -1, -1)
    if mode == "summary":
        relation = summarize(relation, mask)
    elif mode == "diagonal":
        relation = torch.diag_embed(relation.diagonal(dim1=-2, dim2=-1))
    elif mode == "shuffled":
        relation = scramble(relation, mask)
    elif mode not in ("coupled", "uniform"):
        raise ValueError("The attention potential requires a biased comparison mode")
    # Map each content query onto its available trace anchors.
    affinity = logits[:, :, :prefix, prefix:].masked_fill(~observed[:, None, :prefix, None], 0)
    if mode == "uniform":
        weights = mask.to(logits.dtype) / mask.sum(-1, keepdim=True).clamp_min(1)
        weights = weights[:, None, None, :].expand(batch, heads, prefix, -1)
    else:
        weights = torch.softmax(affinity.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min), dim=-1)
    weights = weights.masked_fill(~(mask[:, None, None, :] & observed[:, None, :prefix, None]), 0)
    if unit:
        # Preserve relation magnitude across diffuse anchor assignments.
        norm = torch.linalg.vector_norm(weights, dim=-1, keepdim=True)
        weights = weights / norm.clamp_min(torch.finfo(weights.dtype).tiny)
    anchors = torch.eye(traces, device=logits.device, dtype=logits.dtype)[None, None] * mask[:, None, :, None]
    coordinates = torch.cat((weights, anchors.expand(batch, heads, -1, -1)), dim=2)
    return (coordinates @ relation[:, None]) @ coordinates.transpose(-1, -2)

@dataclass
class NeuralConfig:
    dimension: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    bias_limit: float = 2.0
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    patience: int = 5
    seed: int = 42
    folds: int = 2
    auxiliary_weight: float = 0.1
    mode: str = "coupled"
    trace_dropout: float = 0.0
    label_smoothing: float = 0.0
    aligned: bool = False
    content_query: bool = False
    balanced_reference: bool = False
    field_only: bool = False
    unit_coordinates: bool = False
    paired: bool = False

    def validate(self) -> None:
        integers = (self.dimension, self.heads, self.layers, self.epochs, self.batch_size, self.patience, self.seed, self.folds)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers) or min(integers[:-2]) < 1 or not 0 <= self.seed < 2 ** 32 or self.folds < 2:
            raise ValueError("Neural sizes, epochs, folds and seed must be valid integers")
        scalars = (self.dropout, self.bias_limit, self.learning_rate, self.weight_decay, self.auxiliary_weight, self.trace_dropout, self.label_smoothing)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in scalars):
            raise ValueError("Neural numeric settings must be finite real numbers")
        if self.dimension % self.heads or not 0 <= self.dropout < 1 or not 0 < self.bias_limit <= 8:
            raise ValueError("Invalid attention dimensions, dropout or potential bound")
        if not all(math.isfinite(value) for value in (self.learning_rate, self.weight_decay, self.auxiliary_weight)) or self.learning_rate <= 0 or min(self.weight_decay, self.auxiliary_weight) < 0:
            raise ValueError("Invalid optimizer or auxiliary-loss settings")
        if not 0 <= self.trace_dropout <= 1 or not 0 <= self.label_smoothing < 1:
            raise ValueError("Invalid trace dropout or label smoothing")
        if type(self.aligned) is not bool:
            raise ValueError("The alignment option must be Boolean")
        if type(self.content_query) is not bool:
            raise ValueError("The content query option must be Boolean")
        if type(self.balanced_reference) is not bool:
            raise ValueError("The reference balancing option must be Boolean")
        if type(self.field_only) is not bool or self.field_only and self.mode == "concat":
            raise ValueError("Field routing requires an attention comparison mode")
        if type(self.unit_coordinates) is not bool or self.unit_coordinates and self.mode == "concat":
            raise ValueError("Coordinate normalization requires an attention mode")
        if type(self.paired) is not bool:
            raise ValueError("Shared-space pairing must be Boolean")
        if self.mode not in MODES:
            raise ValueError("Unknown neural comparison mode")

class ResidualBlock(nn.Module):
    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.heads, self.width = config.heads, config.dimension // config.heads
        self.norm = nn.LayerNorm(config.dimension)
        self.qkv = nn.Linear(config.dimension, 3 * config.dimension)
        self.output = nn.Linear(config.dimension, config.dimension)
        self.feed = nn.Sequential(nn.LayerNorm(config.dimension), nn.Linear(config.dimension, 4 * config.dimension), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(4 * config.dimension, config.dimension))
        self.dropout = nn.Dropout(config.dropout)
        self.strength = nn.Parameter(torch.tensor(-2.0))
        self.limit = config.bias_limit
        self.field_only = config.field_only
        self.unit_coordinates = config.unit_coordinates

    def forward(self, tokens: torch.Tensor, observed: torch.Tensor, relation: torch.Tensor, mode: str, diagnostics: list | None = None) -> torch.Tensor:
        batch, length, dimension = tokens.shape
        # Discard unavailable values before normalization and matrix products.
        tokens = tokens.masked_fill(~observed[:, :, None], 0)
        trace_mask = observed[:, length - relation.shape[-1]:]
        relation = relation.masked_fill(~(trace_mask[:, :, None] & trace_mask[:, None, :]), 0)
        qkv = self.qkv(self.norm(tokens)).reshape(batch, length, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = query @ key.transpose(-1, -2) / math.sqrt(self.width)
        original, bias = logits, None
        if mode in BIASED:
            strength = self.limit * torch.sigmoid(self.strength)
            # Condition every interaction on content-selected reference relations.
            bias = strength * potential(logits, observed, relation, mode, unit=self.unit_coordinates)
            logits = logits + bias
        selected = observed.clone()
        if self.field_only:
            # Use physical anchors to route semantic values only.
            selected[:, length - relation.shape[-1]:] = False
        weights = torch.softmax(logits.masked_fill(~selected[:, None, None, :], -torch.inf), dim=-1)
        if diagnostics is not None:
            prefix = length - relation.shape[-1]
            queries = observed[:, :prefix]
            centered = contrast(bias, selected) if bias is not None else torch.zeros_like(logits)
            pairs = (queries.sum(-1) * selected.sum(-1) * self.heads).clamp_min(1)
            rms = (centered[:, :, :prefix].square().sum((1, 2, 3)) / pairs).sqrt()
            plain = torch.softmax(original.masked_fill(~selected[:, None, None, :], -torch.inf), dim=-1)
            transmitted = transport(weights, plain, value, selected).transpose(1, 2).reshape(batch, length, dimension)
            transmitted = torch.nn.functional.linear(transmitted, self.output.weight)
            updates = queries.sum(-1).clamp_min(1) * dimension
            value_rms = (transmitted[:, :prefix].square().sum((1, 2)) / updates).sqrt()
            shifts = (weights[:, :, :prefix] - plain[:, :, :prefix]).abs().sum(-1) * queries[:, None]
            variation = shifts.sum((1, 2)) / (2 * self.heads * queries.sum(-1).clamp_min(1))
            measured = {"attention_contrast_rms": rms, "attention_shift_tv": variation, "value_transport_rms": value_rms}
            if mode == "coupled":
                summary = summarize(relation, observed[:, prefix:])
                # Remove only pair information while keeping the current layer state fixed.
                reduced = strength * potential(original, observed, summary, "coupled", unit=self.unit_coordinates)
                difference = contrast(bias - reduced, selected)[:, :, :prefix]
                alternative = torch.softmax((original + reduced).masked_fill(~selected[:, None, None, :], -torch.inf), dim=-1)
                pair_update = transport(weights, alternative, value, selected).transpose(1, 2).reshape(batch, length, dimension)
                pair_update = torch.nn.functional.linear(pair_update, self.output.weight)
                tangent, remainder = sensitivity(alternative, bias - reduced, value, selected)
                tangent = tangent.transpose(1, 2).reshape(batch, length, dimension)
                tangent = torch.nn.functional.linear(tangent, self.output.weight)
                # Bound the projected remainder with each head's projection norm.
                projection = torch.linalg.vector_norm(self.output.weight.reshape(dimension, self.heads, self.width), dim=(0, 2))
                remainder = (remainder * projection[None, :, None]).sum(1)
                change = (weights[:, :, :prefix] - alternative[:, :, :prefix]).abs().sum(-1) * queries[:, None]
                energy = relation.square().sum((1, 2))
                fraction = (relation - summary).square().sum((1, 2)) / energy.clamp_min(torch.finfo(relation.dtype).tiny)
                measured.update(pair_relation_fraction=fraction.clamp(0, 1), pair_contrast_rms=(difference.square().sum((1, 2, 3)) / pairs).sqrt(), pair_attention_shift_tv=change.sum((1, 2)) / (2 * self.heads * queries.sum(-1).clamp_min(1)))
                measured["pair_value_transport_rms"] = (pair_update[:, :prefix].square().sum((1, 2)) / updates).sqrt()
                measured["pair_tangent_transport_rms"] = (tangent[:, :prefix].square().sum((1, 2)) / updates).sqrt()
                measured["pair_transport_remainder_bound"] = ((remainder[:, :prefix].square() * queries).sum(1) / updates).sqrt()
            diagnostics.append(measured)
        mixed = (weights @ value).transpose(1, 2).reshape(batch, length, dimension)
        tokens = tokens + self.dropout(self.output(mixed))
        return (tokens + self.dropout(self.feed(tokens))).masked_fill(~observed[:, :, None], 0)

class LinearAdapter(nn.Module):
    def __init__(self, original: nn.Linear, rank: int):
        super().__init__()
        if not isinstance(original, nn.Linear) or type(rank) is not int or not 1 <= rank <= min(original.in_features, original.out_features):
            raise ValueError("Linear adaptation requires a valid layer and rank")
        self.original = original.requires_grad_(False)
        self.down = nn.Linear(original.in_features, rank, bias=False, device=original.weight.device, dtype=original.weight.dtype)
        self.up = nn.Linear(rank, original.out_features, bias=False, device=original.weight.device, dtype=original.weight.dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, values):
        # Form the same effective weight used by the exported encoder.
        return nn.functional.linear(values, self.original.weight + self.up.weight @ self.down.weight, self.original.bias)

    def merged(self):
        result = copy.deepcopy(self.original)
        with torch.no_grad():
            result.weight.add_(self.up.weight @ self.down.weight)
        return result

def adapt_linears(model: nn.Module, layers: int, rank: int) -> list[str]:
    if type(layers) is not int or layers < 1 or type(rank) is not int or rank < 1:
        raise ValueError("Encoder adaptation requires positive layer counts and rank")
    candidates = []
    for branch in ("text_model", "vision_model"):
        encoder = getattr(getattr(model, branch, None), "encoder", None)
        blocks = getattr(encoder, "layers", getattr(encoder, "layer", None))
        if blocks is None or len(blocks) < layers:
            raise ValueError("The requested CLIP transformer layers are unavailable")
        for index in range(len(blocks) - layers, len(blocks)):
            for path, module in list(blocks[index].named_modules()):
                if path.rsplit(".", 1)[-1] in ("q_proj", "v_proj", "query", "value") and isinstance(module, nn.Linear):
                    if rank > min(module.in_features, module.out_features):
                        raise ValueError("The adapter rank exceeds its projection dimensions")
                    candidates.append((branch, index, blocks[index], path, module))
    if len(candidates) != 4 * layers:
        raise ValueError("Each adapted CLIP block must expose query and value projections")
    model.requires_grad_(False)
    names = []
    for branch, index, block, path, module in candidates:
        parent, _, name = path.rpartition(".")
        block.get_submodule(parent).add_module(name, LinearAdapter(module, rank))
        container = "layers" if hasattr(getattr(model, branch).encoder, "layers") else "layer"
        names.append(branch + ".encoder." + container + "." + str(index) + "." + path)
    return names

def merge_linears(model: nn.Module) -> int:
    candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, LinearAdapter)]
    for path, module in candidates:
        parent, _, name = path.rpartition(".")
        model.get_submodule(parent).add_module(name, module.merged())
    return len(candidates)

class TextPool(nn.Module):
    def __init__(self, hidden_size: int, width: int = 128):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (hidden_size, width)):
            raise ValueError("Text pooling dimensions must be positive integers")
        self.hidden_size, self.width = hidden_size, width
        self.kernels = (1, 3, 5)
        self.output_size = hidden_size + width * len(self.kernels)
        self.norm = nn.LayerNorm(hidden_size)
        self.filters = nn.ModuleList([nn.Conv1d(hidden_size, width, kernel) for kernel in self.kernels])

    def forward(self, hidden: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor) or not isinstance(observed, torch.Tensor) or hidden.ndim != 3 or hidden.shape[1] < 1 or hidden.shape[-1] != self.hidden_size or not hidden.is_floating_point() or observed.shape != hidden.shape[:2] or observed.dtype != torch.bool or observed.device != hidden.device:
            raise ValueError("Text pooling requires aligned hidden states and word masks")
        values = hidden.masked_fill(~observed[:, :, None], 0)
        if not torch.isfinite(values).all() or not torch.isfinite(hidden[:, 0]).all():
            raise ValueError("Observed text states must be finite")
        values = self.norm(values).masked_fill(~observed[:, :, None], 0).transpose(1, 2)
        output = [hidden[:, 0]]
        for kernel, convolution in zip(self.kernels, self.filters):
            padding = max(0, kernel - values.shape[-1])
            sequence = nn.functional.pad(values, (0, padding))
            mask = nn.functional.pad(observed, (0, padding), value=False)
            valid = mask.unfold(-1, kernel, 1).all(-1)
            # Select local evidence from complete observed word windows.
            response = nn.functional.gelu(convolution(sequence)).masked_fill(~valid[:, None], -torch.inf)
            maximum = response.amax(-1).masked_fill(~valid.any(-1)[:, None], 0)
            output.append(maximum)
        result = torch.cat(output, dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError("Local text pooling produced nonfinite features")
        return result

class ResidualAttention(nn.Module):
    def __init__(self, dimensions: dict[str, int], traces: int, config: NeuralConfig):
        super().__init__()
        config.validate()
        if isinstance(traces, bool) or not isinstance(traces, int) or traces < 2 or not isinstance(dimensions, dict) or not dimensions or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in dimensions.values()):
            raise ValueError("Attention requires valid encoder and trace dimensions")
        self.config, self.names, self.traces = config, list(dimensions), traces
        self.projectors = nn.ModuleDict({name: nn.Sequential(nn.LayerNorm(size), nn.Linear(size, config.dimension)) for name, size in dimensions.items()})
        self.trace_projection = nn.Linear(3, config.dimension)
        if config.aligned and not {"clip_text", "clip_image"}.issubset(dimensions):
            raise ValueError("Alignment requires both CLIP streams")
        self.alignment_projection = nn.Linear(1, config.dimension) if config.aligned else None
        if config.paired and (not {"clip_text", "clip_image"}.issubset(dimensions) or dimensions["clip_text"] != dimensions["clip_image"]):
            raise ValueError("Pairing requires equal-dimensional CLIP streams")
        pair_size = 2 * dimensions.get("clip_text", 0) + 1
        self.pairing_projection = nn.Sequential(nn.LayerNorm(pair_size), nn.Linear(pair_size, config.dimension)) if config.paired else None
        self.identities = nn.Parameter(torch.randn(1, 1 + len(dimensions) + traces + int(config.aligned) + int(config.paired), config.dimension) * 0.02)
        self.blocks = nn.ModuleList([ResidualBlock(config) for _ in range(config.layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(config.dimension), nn.Linear(config.dimension, 1))
        self.concatenation = nn.Sequential(nn.Linear((len(dimensions) + traces + int(config.aligned) + int(config.paired)) * config.dimension, config.dimension), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.dimension, 1))
        self.image_head = nn.Sequential(nn.LayerNorm(config.dimension), nn.Linear(config.dimension, 1))
        # Exclude inactive comparison heads from the optimizer budget.
        self.concatenation.requires_grad_(config.mode == "concat")
        self.blocks.requires_grad_(config.mode != "concat")
        self.classifier.requires_grad_(config.mode != "concat")
        self.identities.requires_grad_(config.mode != "concat")
        self.trace_projection.requires_grad_(config.mode != "deep")
        self.image_head.requires_grad_(config.auxiliary_weight > 0)
        for block in self.blocks:
            block.strength.requires_grad_(config.mode in BIASED)

    def forward(self, features: dict[str, torch.Tensor], observed: torch.Tensor, ranks: torch.Tensor, graph: torch.Tensor, trace_observed: torch.Tensor, explain: bool = False, alignment: torch.Tensor | None = None, pairing: torch.Tensor | None = None) -> dict:
        deep = torch.stack([self.projectors[name](features[name].masked_fill(~observed[:, column, None], 0)) for column, name in enumerate(self.names)], dim=1)
        if self.config.mode == "deep":
            trace_observed = torch.zeros_like(trace_observed)
        elif self.training and self.config.trace_dropout:
            # Train semantic decisions without relying on every physical channel.
            keep = torch.rand((len(trace_observed), 1), device=trace_observed.device) >= self.config.trace_dropout
            trace_observed = trace_observed & keep
        graph = graph.masked_fill(~(trace_observed[:, :, None] & trace_observed[:, None, :]), 0)
        ranks = ranks.masked_fill(~trace_observed, 0)
        if self.config.mode == "independent":
            graph = torch.zeros_like(graph)
        base = graph / (1 + torch.linalg.matrix_norm(graph, keepdim=True))
        contributions = base.sum(-1) / math.sqrt(self.traces)
        trace_values = torch.stack((ranks, base.diagonal(dim1=-2, dim2=-1), contributions), dim=-1)
        trace_tokens = self.trace_projection(trace_values)
        deep = deep.masked_fill(~observed[:, :, None], 0)
        trace_tokens = trace_tokens.masked_fill(~trace_observed[:, :, None], 0)
        image_columns = [index for index, name in enumerate(self.names) if name in ("mae", "clip_image", "forensic")]
        image_mask = observed[:, image_columns]
        image = (deep[:, image_columns] * image_mask[:, :, None]).sum(1) / image_mask.sum(1, keepdim=True).clamp_min(1)
        image_logit = self.image_head(image).squeeze(-1)
        if self.config.aligned:
            if alignment is None or alignment.shape != (len(deep), 2) or not torch.isfinite(alignment).all() or torch.any(alignment[:, 0].abs() > 1) or torch.any((alignment[:, 1] != 0) & (alignment[:, 1] != 1)):
                raise ValueError("Alignment requires finite similarities and availability masks")
            present = alignment[:, 1].bool()
            allowed = observed[:, self.names.index("clip_text")] & observed[:, self.names.index("clip_image")]
            if torch.any(present & ~allowed):
                raise ValueError("Alignment cannot use a missing CLIP input")
            # Let pretrained agreement participate in the same content selection.
            token = self.alignment_projection(alignment[:, :1]).masked_fill(~present[:, None], 0)
            deep = torch.cat((deep, token[:, None]), dim=1)
            observed = torch.cat((observed, present[:, None]), dim=1)
        if self.config.paired:
            expected = (len(deep), 2 * features["clip_text"].shape[1] + 2)
            if not isinstance(pairing, torch.Tensor) or pairing.shape != expected or not pairing.is_floating_point() or not torch.isfinite(pairing).all() or torch.any((pairing[:, -1] != 0) & (pairing[:, -1] != 1)):
                raise ValueError("Pairing requires finite shared-space features and availability")
            present = pairing[:, -1].bool()
            allowed = observed[:, self.names.index("clip_text")] & observed[:, self.names.index("clip_image")]
            if torch.any(present & ~allowed):
                raise ValueError("Pairing cannot use missing text or image inputs")
            # Keep pair-specific evidence available during content selection.
            token = self.pairing_projection(pairing[:, :-1]).masked_fill(~present[:, None], 0)
            deep = torch.cat((deep, token[:, None]), dim=1)
            observed = torch.cat((observed, present[:, None]), dim=1)
        if self.config.mode == "concat":
            news = self.concatenation(torch.cat((deep.flatten(1), trace_tokens.flatten(1)), dim=-1)).squeeze(-1)
            return {"logit": news, "image_logit": image_logit}
        batch = ranks.shape[0]
        query = torch.zeros_like(deep[:, :1])
        if self.config.content_query:
            # Initialize evidence selection from the observed news content.
            query = deep.sum(1, keepdim=True) / observed.sum(1)[:, None, None].clamp_min(1)
        tokens = torch.cat((query, deep, trace_tokens), dim=1) + self.identities
        available = torch.cat((torch.ones((batch, 1), dtype=torch.bool, device=ranks.device), observed, trace_observed), dim=1)
        diagnostics = [] if explain else None
        for block in self.blocks:
            anchors = tokens[:, -self.traces:]
            tokens = block(tokens, available, base, self.config.mode, diagnostics)
            if self.config.field_only:
                tokens = torch.cat((tokens[:, :-self.traces], anchors), dim=1)
        result = {"logit": self.classifier(tokens[:, 0]).squeeze(-1), "image_logit": image_logit}
        if diagnostics:
            result.update({name: torch.stack([item[name] for item in diagnostics], dim=1).mean(1) for name in diagnostics[0]})
        return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
