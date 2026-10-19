# -*- coding: utf-8 -*-
# Portable Storage: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import csv
import copy
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from os.path import abspath
from pathlib import Path, PureWindowsPath
from typing import Any
import numpy as np
def relative_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(str(value)).drive or str(value).startswith(("\\", "/")):
        raise ValueError("Use a relative filesystem path")
    return path
def resolved(value: str | Path) -> Path:
    return Path(abspath(relative_path(value))).resolve()
def same_path(first: str | Path, second: str | Path) -> bool:
    left, right = resolved(first), resolved(second)
    return left == right or left.exists() and right.exists() and os.path.samefile(left, right)
@contextmanager
def atomic_stream(path: str | Path, encoding: str = "utf-8", newline: str = "\n"):
    target = relative_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline=newline, dir=target.parent, prefix=".flawprint-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
def digest(path: str | Path) -> str:
    checksum = hashlib.sha256()
    with relative_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            checksum.update(block)
    return checksum.hexdigest()
def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value
def unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object keys must be unique: " + key)
        result[key] = value
    return result
def invalid_constant(value: str) -> None:
    raise ValueError("JSON values must be finite; use null for missing cells")
def read_json(path: str | Path) -> Any:
    with relative_path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=unique_object, parse_constant=invalid_constant)
def write_json(path: str | Path, value: Any) -> None:
    content = json.dumps(native(value), ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    with atomic_stream(path) as stream:
        stream.write(content)
def read_rows(path: str | Path, encoding: str = "utf-8-sig", delimiter: str | None = None, records_key: str | None = None, sheet: str | int = 0, trusted_pickle: bool = False) -> list[dict]:
    target = relative_path(path)
    suffix = target.suffix.lower()
    if suffix in (".pkl", ".pickle") and not trusted_pickle:
        raise ValueError("Pickle input requires the explicit trusted-pickle option")
    if suffix in (".xlsx", ".xls", ".parquet", ".pkl", ".pickle"):
        try:
            import pandas as pd
            if suffix in (".xlsx", ".xls"):
                table = pd.read_excel(target, sheet_name=sheet, dtype=object, header=None, keep_default_na=False)
                if table.empty:
                    raise ValueError("The selected Excel sheet is empty")
                headers = table.iloc[0].tolist()
                if any(not isinstance(name, str) or not name.strip() for name in headers) or len({name.strip().casefold() for name in headers}) != len(headers):
                    raise ValueError("Excel headers must be unique and nonempty strings")
                frame = table.iloc[1:].copy()
                frame.columns = [name.strip() for name in headers]
            elif suffix == ".parquet":
                frame = pd.read_parquet(target)
            else:
                frame = pd.read_pickle(target)
            if not isinstance(frame, pd.DataFrame):
                raise ValueError("Table input must contain a DataFrame")
            if any(not isinstance(name, str) or not name.strip() for name in frame.columns) or len({name.strip().casefold() for name in frame.columns}) != len(frame.columns):
                raise ValueError("Table columns must be unique and nonempty strings")
            result = frame.to_dict(orient="records")
        except ImportError as error:
            raise ValueError("Table support requires python -m pip install -r ./tables.txt") from error
    elif suffix in (".csv", ".tsv"):
        with target.open("r", encoding=encoding, newline="") as stream:
            chosen = delimiter or ("\t" if suffix == ".tsv" else ",")
            if chosen == "auto":
                sample = stream.read(65536)
                stream.seek(0)
                try:
                    chosen = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                except csv.Error as error:
                    raise ValueError("Cannot infer the delimiter; supply it explicitly") from error
            chosen = "\t" if chosen == "\\t" else chosen
            if len(chosen) != 1:
                raise ValueError("A table delimiter must be one character")
            reader = csv.DictReader(stream, delimiter=chosen, strict=True)
            headers = reader.fieldnames or []
            if not headers or any(not name.strip() for name in headers) or len({name.strip().casefold() for name in headers}) != len(headers):
                raise ValueError("CSV headers must be unique and nonempty")
            result = []
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"CSV row {line} has a different column count")
                result.append({key.strip(): value for key, value in row.items()})
    elif suffix == ".json":
        value = json.loads(target.read_text(encoding=encoding), object_pairs_hook=unique_object, parse_constant=invalid_constant)
        if records_key is not None:
            if not isinstance(value, dict) or records_key not in value:
                raise ValueError("The requested JSON record key does not exist")
            value = value[records_key]
        if not isinstance(value, list):
            raise ValueError("JSON input must be an array or use an explicit records-key")
        result = value
    elif suffix in (".jsonl", ".ndjson"):
        result = []
        with target.open("r", encoding=encoding) as stream:
            for line, content in enumerate(stream, 1):
                if content.strip():
                    try:
                        result.append(json.loads(content, object_pairs_hook=unique_object, parse_constant=invalid_constant))
                    except ValueError as error:
                        raise ValueError(f"Invalid JSON on line {line}: {error}") from error
    else:
        raise ValueError("Supported inputs are JSONL, NDJSON, JSON, CSV, TSV, Excel, Parquet, and trusted DataFrame pickle")
    if any(not isinstance(row, dict) or any(not isinstance(key, str) for key in row) for row in result):
        raise ValueError("Every input row must be an object with string column names")
    return result
def write_rows(path: str | Path, rows: list[dict]) -> None:
    target = relative_path(path)
    if target.suffix.lower() == ".json":
        write_json(target, rows)
        return
    if target.suffix.lower() in (".csv", ".tsv"):
        write_csv(target, rows, "\t" if target.suffix.lower() == ".tsv" else ",")
        return
    if target.suffix.lower() not in (".jsonl", ".ndjson"):
        raise ValueError("Row exports must use JSONL, NDJSON, JSON, CSV, or TSV")
    with atomic_stream(target) as stream:
        for row in rows:
            stream.write(json.dumps(native(row), ensure_ascii=True, allow_nan=False) + "\n")
def write_csv(path: str | Path, rows: list[dict], delimiter: str = ",") -> None:
    target = relative_path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        raise ValueError("CSV export requires at least one named column; use JSON for empty results")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("A table delimiter must be one character")
    with atomic_stream(target, "utf-8-sig", "") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in native(rows):
            writer.writerow({key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})
def save_bundle(path: str | Path, values: np.ndarray, observed: np.ndarray, patches: np.ndarray, rows: list[dict], names: list[str]) -> None:
    validate_bundle({"values": values, "observed": observed, "patches": patches, "rows": rows, "names": np.asarray(names)})
    target = relative_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        np.savez_compressed(stream, values=values, observed=observed, patches=patches, metadata=np.asarray(json.dumps(native(rows), ensure_ascii=True)), names=np.asarray(names))
def load_bundle(path: str | Path) -> dict:
    with np.load(relative_path(path), allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in ("values", "observed", "patches", "names")}
        result["rows"] = json.loads(str(archive["metadata"].item()))
    validate_bundle(result)
    return result
def validate_bundle(bundle: dict) -> None:
    if not isinstance(bundle, dict) or not {"values", "observed", "patches", "names", "rows"}.issubset(bundle):
        raise ValueError("The feature bundle is missing required fields")
    values, observed, patches = (np.asarray(bundle[key]) for key in ("values", "observed", "patches"))
    names, rows = np.asarray(bundle["names"]), bundle["rows"]
    if values.ndim != 2 or values.shape[1] < 1 or observed.shape != values.shape or observed.dtype.kind != "b":
        raise ValueError("Features require a two-dimensional matrix and a Boolean observation mask")
    if patches.ndim != 3 or patches.shape[0] != len(values) or patches.shape[1] < 1 or patches.shape[2] != values.shape[1]:
        raise ValueError("Patch dimensions must match sample and feature dimensions")
    if names.ndim != 1 or len(names) != values.shape[1] or names.dtype.kind not in "US" or any(not str(name).strip() for name in names) or len(set(names.tolist())) != len(names):
        raise ValueError("Feature names must be unique nonempty strings")
    if not isinstance(rows, list) or len(rows) != len(values) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Sample metadata must match the feature rows")
    identifiers = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Feature sample identifiers must be unique nonempty strings")
    if any(not isinstance(row.get("domain"), str) or not row["domain"].strip() for row in rows):
        raise ValueError("Every feature sample requires a nonempty domain")
    for array in (values, patches):
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError("Feature bundles must contain finite real values")
def subset(bundle: dict, indices: np.ndarray | list[int]) -> dict:
    selected = np.asarray(indices)
    if selected.size == 0:
        selected = np.asarray([], dtype=int)
    if selected.ndim != 1 or selected.dtype.kind not in "iu" or np.any(selected < 0) or np.any(selected >= len(bundle["rows"])) or len(np.unique(selected)) != len(selected):
        raise ValueError("Sample indices must be unique integers within the bundle")
    return {"values": bundle["values"][selected], "observed": bundle["observed"][selected], "patches": bundle["patches"][selected], "names": bundle["names"].copy(), "rows": [copy.deepcopy(bundle["rows"][int(index)]) for index in selected]}
