# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import ast
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import json
import math
from engine.data import SPLITS, SPLIT_ALIASES, identifier, missing, normalize, stable_identifier, portable_image

FIELDS = {"id": ("id", "news_id", "post_id", "tweet_id", "uid"), "image": ("image", "image_path", "images", "image_url", "picture"), "text": ("text", "content", "body", "news_text"), "label": ("label", "target", "class", "y"), "domain": ("domain", "topic", "category"), "platform": ("platform",), "group": ("group", "event_id", "cluster_id"), "split": ("split", "partition", "subset"), "semantic_logit": ("semantic_logit", "logit")}

@dataclass
class Conversion:
    dataset: str = "canonical"
    columns: dict[str, str] = field(default_factory=dict)
    labels: str | None = None
    label_map: dict[str, int] = field(default_factory=dict)
    domain_map: dict[str, str] = field(default_factory=dict)
    split_map: dict[str, str] = field(default_factory=dict)
    image_root: str = "."
    image_separator: str | None = None
    image_policy: str = "error"
    strip_image_prefix: str | None = None
    id_mode: str = "source"
    id_prefix: str = ""
    split: str | None = None
    exclude_domains: list[str] = field(default_factory=list)
    keep_columns: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.dataset, str):
            raise ValueError("The dataset name must be a string")
        # Normalize names before selecting dataset rules.
        self.dataset = self.dataset.strip().lower()
        if not isinstance(self.columns, dict) or any(not isinstance(value, list) for value in (self.exclude_domains, self.keep_columns)):
            raise ValueError("Columns must be an object and exclusions and retained fields must be lists")
        if self.dataset not in ("canonical", "weibo", "finefake", "weibo21", "custom") or self.labels not in (None, "fake-one", "real-one"):
            raise ValueError("Unknown dataset or label convention")
        if self.image_policy not in ("error", "first", "all") or self.id_mode not in ("source", "content"):
            raise ValueError("Unknown image policy or ID mode")
        if any(key not in FIELDS or not isinstance(value, str) or not value for key, value in self.columns.items()):
            raise ValueError("Column mappings require canonical field names and nonempty source names")
        for mapping in (self.label_map, self.domain_map, self.split_map):
            if not isinstance(mapping, dict) or any(not isinstance(key, str) for key in mapping):
                raise ValueError("Value mappings must be objects with string keys")
        if any(isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1) for value in self.label_map.values()):
            raise ValueError("Mapped labels must be canonical zero-real-one-fake integers")
        if any(not isinstance(value, str) or not value for value in (*self.domain_map.values(), *self.split_map.values())):
            raise ValueError("Mapped domain and split values must be nonempty strings")
        if self.split is not None and canonical_split(self.split) not in SPLITS:
            raise ValueError("The split override must name a supported partition")
        if not isinstance(self.image_root, str) or not self.image_root.strip():
            raise ValueError("The image root must be a nonempty relative path string")
        if any(value is not None and (not isinstance(value, str) or not value) for value in (self.image_separator, self.strip_image_prefix)):
            raise ValueError("Image separators and removable prefixes must be nonempty")
        if not isinstance(self.id_prefix, str) or any(not isinstance(item, str) for item in (*self.exclude_domains, *self.keep_columns)):
            raise ValueError("ID prefixes, excluded domains, and retained columns must be strings")

    def state(self) -> dict:
        return asdict(self)

def canonical_split(value: object) -> str:
    text = "unspecified" if missing(value) else str(value).strip().casefold()
    return SPLIT_ALIASES.get(text, text)

def resolve_columns(records: list[dict], config: Conversion) -> dict[str, str]:
    names = {key for row in records for key in row}
    folded = {}
    for name in names:
        key = name.strip().casefold()
        if not key or key in folded:
            raise ValueError("Column names must be unique after trimming and case folding")
        folded[key] = name
    selected = {}
    for field, aliases in FIELDS.items():
        explicit = config.columns.get(field)
        if explicit is not None:
            if explicit not in names:
                raise ValueError(f"Mapped column does not exist: {explicit}")
            selected[field] = explicit
        elif field in folded:
            selected[field] = folded[field]
        else:
            candidates = [folded[alias] for alias in aliases if alias in folded]
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous {field} columns; provide an explicit mapping")
            if candidates:
                selected[field] = candidates[0]
    required = ("text", "label", "domain") if config.dataset in ("weibo", "finefake", "weibo21") else ()
    if any(field not in selected for field in required):
        raise ValueError("Required columns are missing: " + ", ".join(field for field in required if field not in selected))
    if "text" not in selected and "image" not in selected:
        raise ValueError("The table requires a text or image column")
    if any(name not in names for name in config.keep_columns):
        raise ValueError("A requested retained column does not exist")
    return selected

def images(value: object, config: Conversion) -> list[str]:
    if missing(value):
        return [""]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = value.strip()
        if value.startswith(("[", "(")):
            try:
                value = json.loads(value) if value.startswith("[") else ast.literal_eval(value)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError) as error:
                    raise ValueError("Invalid serialized image list") from error
        elif config.image_separator:
            value = value.split(config.image_separator)
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    output = []
    for item in values:
        if missing(item):
            continue
        if not isinstance(item, str):
            raise ValueError("Image lists must contain path strings")
        path = item.strip().replace("\\", "/")
        if config.strip_image_prefix:
            prefix = config.strip_image_prefix.replace("\\", "/").rstrip("/")
            if not path.startswith(prefix + "/"):
                raise ValueError("An image path does not match the removable prefix")
            path = path[len(prefix) + 1:]
            if not path:
                raise ValueError("An image path is empty after prefix removal")
        if path in output:
            raise ValueError("An image list contains a duplicate path")
        output.append(path)
    return output or [""]

def label_token(value: object) -> str:
    token = str(value).strip()
    try:
        number = Decimal(token)
        if number.is_finite() and number == number.to_integral_value():
            return str(int(number))
    except InvalidOperation:
        pass
    return token

def convert_records(records: list[dict], config: Conversion, source: str = "") -> tuple[list[dict], dict]:
    config.validate()
    if not records:
        raise ValueError("The input table is empty")
    columns = resolve_columns(records, config)
    convention = config.labels or ("real-one" if config.dataset == "finefake" else "fake-one")
    rows, generated, excluded, expanded, discarded, removed_labels = [], 0, 0, 0, 0, 0
    for number, original in enumerate(records, 1):
        def value(field: str, default: object = "") -> object:
            return original.get(columns.get(field, ""), default)
        text_value = value("text")
        text = "" if text_value is None or type(text_value).__name__ in ("NAType", "NaTType") or isinstance(text_value, float) and math.isnan(text_value) else str(text_value)
        domain = "unknown" if missing(value("domain")) else str(value("domain")).strip()
        domain = config.domain_map.get(domain, domain)
        if domain in config.exclude_domains:
            excluded += 1
            continue
        paths = [portable_image(path, config.image_root) for path in images(value("image"), config)]
        if len(set(paths)) != len(paths):
            raise ValueError(f"Row {number} repeats an equivalent image path")
        if len(paths) > 1 and config.image_policy == "error":
            raise ValueError(f"Row {number} contains multiple images; choose first or all explicitly")
        raw_id = identifier(value("id")) if config.id_mode == "source" else ""
        if not raw_id:
            raw_id = stable_identifier(text, paths[0] if len(paths) == 1 else json.dumps(paths, ensure_ascii=True), domain)
            generated += 1
        group = identifier(value("group")) or raw_id
        split = config.split or config.split_map.get(str(value("split")).strip(), value("split"))
        raw_label = value("label", None)
        mapped = bool(config.label_map)
        if not missing(raw_label):
            token = label_token(raw_label)
            if mapped:
                if token not in config.label_map:
                    raise ValueError(f"Unmapped label in row {number}: {token}")
                raw_label = config.label_map[token]
            elif config.labels is None and config.dataset in ("weibo21", "custom") and token.casefold() not in ("real", "true", "fake", "false"):
                raise ValueError("Numeric labels require an explicit labels convention or label-map")
        base = {"text": text, "domain": domain, "platform": value("platform", "unknown"), "group": group, "split": split, "label": raw_label, "source_file": source, "source_row": number, "image_count": sum(bool(path) for path in paths)}
        protected = set(base) | {"id", "image", "has_image"}
        if config.dataset == "canonical":
            for key, item in original.items():
                if key not in columns.values() and key not in protected:
                    base[key] = item
        for key in ("semantic_logit", "physical_manipulated", "narrative_false", "patch_signs_valid_after_attack", "content_sha256"):
            if key in original or key in columns:
                base[key] = value(key) if key in columns else original[key]
        for key in config.keep_columns:
            if key in protected or key in FIELDS:
                raise ValueError("Retained columns cannot override canonical fields")
            base[key] = original[key]
        selected = paths if config.image_policy == "all" else paths[:1]
        expanded += max(0, len(selected) - 1)
        discarded += len(paths) - len(selected)
        if canonical_split(split) == "profile" and not missing(raw_label):
            removed_labels += 1
        for index, image in enumerate(selected, 1):
            suffix = f":image{index}" if len(selected) > 1 else ""
            row = dict(base, id=config.id_prefix + raw_id + suffix, image=image)
            try:
                rows.extend(normalize([row], config.image_root, "fake-one" if mapped else convention))
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"Invalid row {number}: {error}") from error
    rows = normalize(rows)
    report = {"input_rows": len(records), "output_rows": len(rows), "columns": columns, "input_label_convention": "explicit-map" if config.label_map else convention, "generated_ids": generated, "excluded_rows": excluded, "expanded_images": expanded, "discarded_images": discarded, "profile_labels_removed": removed_labels, "sample_unit": "image-news-pair" if expanded else "news", "image_policy": config.image_policy}
    return rows, report

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
