# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import dataclass
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
    if values.ndim != 4 or values.shape[-2] != values.shape[-1] or observed.shape != (values.shape[0], values.shape[-1]) or observed.dtype != torch.bool:
        raise ValueError("Attention contrasts require square head matrices and a Boolean node mask")
    keys = observed[:, None, None, :]
    count = observed.sum(-1)[:, None, None, None].clamp_min(1)
    # Remove the row offset that cancels in softmax.
    center = values.masked_fill(~keys, 0).sum(-1, keepdim=True) / count
    return (values - center).masked_fill(~(keys & observed[:, None, :, None]), 0)

def potential(logits: torch.Tensor, observed: torch.Tensor, relation: torch.Tensor, mode: str) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[-1] != logits.shape[-2] or not logits.is_floating_point() or relation.ndim != 3 or relation.dtype != logits.dtype or relation.device != logits.device or observed.dtype != torch.bool or observed.device != logits.device:
        raise ValueError("Attention potentials require square floating matrices and a Boolean node mask")
    batch, heads, length, _ = logits.shape
    traces = relation.shape[-1]
    prefix = length - traces
    if prefix < 1 or traces < 1 or relation.shape != (batch, traces, traces) or observed.shape != (batch, length):
        raise ValueError("Relation coordinates must match the attention sequence")
    mask = observed[:, prefix:]
    relation = relation * mask[:, :, None] * mask[:, None, :]
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
    affinity = logits[:, :, :prefix, prefix:]
    if mode == "uniform":
        weights = mask.to(logits.dtype) / mask.sum(-1, keepdim=True).clamp_min(1)
        weights = weights[:, None, None, :].expand(batch, heads, prefix, -1)
    else:
        weights = torch.softmax(affinity.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min), dim=-1)
    weights = weights * mask[:, None, None, :] * observed[:, None, :prefix, None]
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

    def validate(self) -> None:
        integers = (self.dimension, self.heads, self.layers, self.epochs, self.batch_size, self.patience, self.seed, self.folds)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers) or min(integers[:-2]) < 1 or not 0 <= self.seed < 2 ** 32 or self.folds < 2:
            raise ValueError("Neural sizes, epochs, folds and seed must be valid integers")
        scalars = (self.dropout, self.bias_limit, self.learning_rate, self.weight_decay, self.auxiliary_weight)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in scalars):
            raise ValueError("Neural numeric settings must be finite real numbers")
        if self.dimension % self.heads or not 0 <= self.dropout < 1 or not 0 < self.bias_limit <= 8:
            raise ValueError("Invalid attention dimensions, dropout or potential bound")
        if not all(math.isfinite(value) for value in (self.learning_rate, self.weight_decay, self.auxiliary_weight)) or self.learning_rate <= 0 or min(self.weight_decay, self.auxiliary_weight) < 0:
            raise ValueError("Invalid optimizer or auxiliary-loss settings")
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

    def forward(self, tokens: torch.Tensor, observed: torch.Tensor, relation: torch.Tensor, mode: str, diagnostics: list | None = None) -> torch.Tensor:
        batch, length, dimension = tokens.shape
        qkv = self.qkv(self.norm(tokens)).reshape(batch, length, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = query @ key.transpose(-1, -2) / math.sqrt(self.width)
        original, bias = logits, None
        if mode in BIASED:
            strength = self.limit * torch.sigmoid(self.strength)
            # Condition every interaction on content-selected reference relations.
            bias = strength * potential(logits, observed, relation, mode)
            logits = logits + bias
        weights = torch.softmax(logits.masked_fill(~observed[:, None, None, :], -torch.inf), dim=-1)
        if diagnostics is not None:
            prefix = length - relation.shape[-1]
            queries = observed[:, :prefix]
            centered = contrast(bias, observed) if bias is not None else torch.zeros_like(logits)
            pairs = (queries.sum(-1) * observed.sum(-1) * self.heads).clamp_min(1)
            rms = (centered[:, :, :prefix].square().sum((1, 2, 3)) / pairs).sqrt()
            plain = torch.softmax(original.masked_fill(~observed[:, None, None, :], -torch.inf), dim=-1)
            shifts = (weights[:, :, :prefix] - plain[:, :, :prefix]).abs().sum(-1) * queries[:, None]
            variation = shifts.sum((1, 2)) / (2 * self.heads * queries.sum(-1).clamp_min(1))
            measured = {"attention_contrast_rms": rms, "attention_shift_tv": variation}
            if mode == "coupled":
                summary = summarize(relation, observed[:, prefix:])
                # Remove only pair information while keeping the current layer state fixed.
                reduced = strength * potential(original, observed, summary, "coupled")
                difference = contrast(bias - reduced, observed)[:, :, :prefix]
                alternative = torch.softmax((original + reduced).masked_fill(~observed[:, None, None, :], -torch.inf), dim=-1)
                change = (weights[:, :, :prefix] - alternative[:, :, :prefix]).abs().sum(-1) * queries[:, None]
                energy = relation.square().sum((1, 2))
                fraction = (relation - summary).square().sum((1, 2)) / energy.clamp_min(torch.finfo(relation.dtype).tiny)
                measured.update(pair_relation_fraction=fraction.clamp(0, 1), pair_contrast_rms=(difference.square().sum((1, 2, 3)) / pairs).sqrt(), pair_attention_shift_tv=change.sum((1, 2)) / (2 * self.heads * queries.sum(-1).clamp_min(1)))
            diagnostics.append(measured)
        mixed = (weights @ value).transpose(1, 2).reshape(batch, length, dimension)
        tokens = tokens + self.dropout(self.output(mixed))
        return (tokens + self.dropout(self.feed(tokens))) * observed[:, :, None]

class ResidualAttention(nn.Module):
    def __init__(self, dimensions: dict[str, int], traces: int, config: NeuralConfig):
        super().__init__()
        config.validate()
        if isinstance(traces, bool) or not isinstance(traces, int) or traces < 2 or not isinstance(dimensions, dict) or not dimensions or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in dimensions.values()):
            raise ValueError("Attention requires valid encoder and trace dimensions")
        self.config, self.names, self.traces = config, list(dimensions), traces
        self.projectors = nn.ModuleDict({name: nn.Sequential(nn.LayerNorm(size), nn.Linear(size, config.dimension)) for name, size in dimensions.items()})
        self.trace_projection = nn.Linear(3, config.dimension)
        self.identities = nn.Parameter(torch.randn(1, 1 + len(dimensions) + traces, config.dimension) * 0.02)
        self.blocks = nn.ModuleList([ResidualBlock(config) for _ in range(config.layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(config.dimension), nn.Linear(config.dimension, 1))
        self.concatenation = nn.Sequential(nn.Linear((len(dimensions) + traces) * config.dimension, config.dimension), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.dimension, 1))
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

    def forward(self, features: dict[str, torch.Tensor], observed: torch.Tensor, ranks: torch.Tensor, graph: torch.Tensor, trace_observed: torch.Tensor, explain: bool = False) -> dict:
        deep = torch.stack([self.projectors[name](features[name]) for name in self.names], dim=1)
        graph = graph * trace_observed[:, :, None] * trace_observed[:, None, :]
        if self.config.mode == "independent":
            graph = torch.zeros_like(graph)
        base = graph / (1 + torch.linalg.matrix_norm(graph, keepdim=True))
        contributions = base.sum(-1) / math.sqrt(self.traces)
        trace_values = torch.stack((ranks, base.diagonal(dim1=-2, dim2=-1), contributions), dim=-1)
        trace_tokens = self.trace_projection(trace_values)
        if self.config.mode == "deep":
            trace_observed = torch.zeros_like(trace_observed)
        deep = deep * observed[:, :, None]
        trace_tokens = trace_tokens * trace_observed[:, :, None]
        image_columns = [index for index, name in enumerate(self.names) if name in ("mae", "clip_image", "forensic")]
        image_mask = observed[:, image_columns]
        image = (deep[:, image_columns] * image_mask[:, :, None]).sum(1) / image_mask.sum(1, keepdim=True).clamp_min(1)
        image_logit = self.image_head(image).squeeze(-1)
        if self.config.mode == "concat":
            news = self.concatenation(torch.cat((deep.flatten(1), trace_tokens.flatten(1)), dim=-1)).squeeze(-1)
            return {"logit": news, "image_logit": image_logit}
        batch = ranks.shape[0]
        tokens = torch.cat((torch.zeros_like(deep[:, :1]), deep, trace_tokens), dim=1) + self.identities
        available = torch.cat((torch.ones((batch, 1), dtype=torch.bool, device=ranks.device), observed, trace_observed), dim=1)
        diagnostics = [] if explain else None
        for block in self.blocks:
            tokens = block(tokens, available, base, self.config.mode, diagnostics)
        result = {"logit": self.classifier(tokens[:, 0]).squeeze(-1), "image_logit": image_logit}
        if diagnostics:
            result.update({name: torch.stack([item[name] for item in diagnostics], dim=1).mean(1) for name in diagnostics[0]})
        return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
