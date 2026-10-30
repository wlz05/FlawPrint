# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import csv
import copy
import hashlib
import json
import os
import shutil
import tempfile
from zipfile import BadZipFile
from zlib import error as CompressionError
from contextlib import contextmanager
from os.path import abspath
from pathlib import Path, PureWindowsPath
from typing import Any
import numpy as np

def relative_path(value: str | Path) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("Filesystem paths must be nonempty strings or Path objects")
    text = os.fspath(value)
    if not isinstance(text, str) or not text.strip() or "\0" in text:
        raise ValueError("Filesystem paths must be nonempty strings or Path objects")
    text = text.replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or PureWindowsPath(text).drive or text.startswith("/"):
        raise ValueError("Use a relative filesystem path")
    if text == "~" or text.startswith("~/") or text.casefold().startswith(("http://", "https://", "file://")):
        raise ValueError("Use an explicit relative filesystem path")
    return path

def resolved(value: str | Path) -> Path:
    return Path(abspath(relative_path(value))).resolve()

def same_path(first: str | Path, second: str | Path) -> bool:
    left, right = resolved(first), resolved(second)
    return left == right or left.exists() and right.exists() and os.path.samefile(left, right)

def protect_outputs(outputs: list[str | Path], inputs: list[str | Path]) -> None:
    targets = [resolved(path) for path in outputs]
    for target in targets:
        if target.exists() and not target.is_file():
            raise ValueError("Output paths must name regular files")
        parent = target.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir():
            raise ValueError("Output parents must be directories")
    if any(same_path(first, second) for index, first in enumerate(outputs) for second in outputs[index + 1:]):
        raise ValueError("Output files must be distinct")
    if any(first in second.parents or second in first.parents for index, first in enumerate(targets) for second in targets[index + 1:]):
        raise ValueError("Output files cannot contain other output paths")
    existing = [path for path in targets if path.exists()]
    seen = set()
    for value in inputs:
        source = resolved(value)
        if source in seen:
            continue
        seen.add(source)
        if source in targets or source.exists() and any(os.path.samefile(source, target) for target in existing):
            raise ValueError("Output paths cannot overwrite source files")

@contextmanager
def staged_outputs(outputs: list[str | Path]):
    targets = [relative_path(path) for path in outputs]
    protect_outputs(targets, [])
    temporary, retained, backups, committed = [], set(), {}, []
    def allocate(target: Path, suffix: str) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".flawprint-", suffix=suffix, delete=False) as stream:
            path = relative_path(os.path.relpath(stream.name))
        temporary.append(path)
        path.unlink()
        return path
    def current(target: Path) -> str | None:
        return digest(target) if target.exists() else None
    try:
        original = [current(target) for target in targets]
        stages = [allocate(target, ".tmp" + target.suffix) for target in targets]
        yield stages
        protect_outputs(targets, [])
        if any(not stage.is_file() for stage in stages):
            raise ValueError("Every staged output must be written before publication")
        hashes = [digest(stage) for stage in stages]
        for index, target in enumerate(targets):
            if current(target) != original[index]:
                raise ValueError("An output changed during preparation; use distinct output paths")
            if original[index] is not None:
                backup = allocate(target, ".recovery")
                backups[index] = backup
                shutil.copyfile(target, backup)
                if digest(backup) != original[index]:
                    raise ValueError("An output changed while its recovery copy was prepared")
        try:
            for index, target in enumerate(targets):
                if current(target) != original[index]:
                    raise ValueError("An output changed before publication")
                os.replace(stages[index], target)
                committed.append(index)
        except BaseException as error:
            failed = []
            # Restore only files that still belong to this publication.
            for index in reversed(committed):
                try:
                    if current(targets[index]) != hashes[index]:
                        raise ValueError("An output changed before rollback")
                    if index in backups:
                        os.replace(backups[index], targets[index])
                    else:
                        targets[index].unlink()
                except (OSError, ValueError):
                    if index in backups:
                        retained.add(backups[index])
                    failed.append(targets[index].as_posix())
            if failed:
                recovery = ", ".join(path.as_posix() for path in retained)
                raise OSError("Output rollback incomplete for " + ", ".join(failed) + "; recovery copies: " + recovery) from error
            raise
    finally:
        for path in temporary:
            if path not in retained and path.exists():
                path.unlink()

@contextmanager
def atomic_stream(path: str | Path, encoding: str = "utf-8", newline: str = "\n", binary: bool = False):
    target = relative_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        options = {"mode": "wb"} if binary else {"mode": "w", "encoding": encoding, "newline": newline}
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".flawprint-", suffix=".tmp", delete=False, **options) as stream:
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

def real_array(value: Any) -> np.ndarray:
    original = np.asarray(value)
    if np.iscomplexobj(original) or original.dtype.kind == "O" and any(isinstance(item, (complex, np.complexfloating)) for item in original.flat):
        raise ValueError("Numeric arrays must contain real values")
    return np.asarray(value, dtype=float)

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

def finite_float(value: str) -> float:
    # Reject exponent overflow before constructing numeric arrays.
    number = float(value)
    if not np.isfinite(number):
        invalid_constant(value)
    return number

def parse_json(content: str) -> Any:
    try:
        return json.loads(content, object_pairs_hook=unique_object, parse_constant=invalid_constant, parse_float=finite_float)
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds the supported depth") from error

def read_json(path: str | Path, checksum: str | None = None) -> Any:
    if checksum is not None:
        content = relative_path(path).read_bytes()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError("JSON content changed or has an invalid SHA256 fingerprint")
        # Decode the same bytes whose fingerprint was verified.
        return parse_json(content.decode("utf-8-sig"))
    with relative_path(path).open("r", encoding="utf-8-sig") as stream:
        return parse_json(stream.read())

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
                # Keep the table engine independent of machine-local pandas options.
                frame = pd.read_parquet(target, engine="pyarrow")
            else:
                frame = pd.read_pickle(target)
            if not isinstance(frame, pd.DataFrame):
                raise ValueError("Table input must contain a DataFrame")
            if any(not isinstance(name, str) or not name.strip() for name in frame.columns) or len({name.strip().casefold() for name in frame.columns}) != len(frame.columns):
                raise ValueError("Table columns must be unique and nonempty strings")
            result = frame.to_dict(orient="records")
        except ImportError as error:
            raise ValueError('Table support requires python -m pip install "pandas>=2.2,<4" "openpyxl>=3.1,<4" "pyarrow>=15,<30" "xlrd>=2,<3"') from error
    elif suffix in (".csv", ".tsv"):
        with target.open("r", encoding=encoding, newline="") as stream:
            chosen = ("\t" if suffix == ".tsv" else ",") if delimiter is None else delimiter
            if not isinstance(chosen, str):
                raise ValueError("A table delimiter must be one character")
            if chosen == "auto":
                sample = stream.read(65536)
                stream.seek(0)
                try:
                    chosen = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                except csv.Error as error:
                    raise ValueError("Cannot infer the delimiter; supply it explicitly") from error
            chosen = "\t" if chosen == "\\t" else chosen
            if len(chosen) != 1 or chosen in "\r\n\0":
                raise ValueError("A table delimiter must be one character")
            reader = csv.DictReader(stream, delimiter=chosen, strict=True)
            try:
                headers = reader.fieldnames or []
                if not headers or any(not name.strip() for name in headers) or len({name.strip().casefold() for name in headers}) != len(headers):
                    raise ValueError("CSV headers must be unique and nonempty")
                result = []
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"CSV row ending on line {reader.line_num} has a different column count")
                    result.append({key.strip(): value for key, value in row.items()})
            except csv.Error as error:
                raise ValueError(f"Invalid CSV near line {reader.reader.line_num}: {error}") from error
    elif suffix == ".json":
        value = parse_json(target.read_text(encoding=encoding))
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
                        result.append(parse_json(content))
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
    if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in "\r\n\0":
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
    with atomic_stream(target, binary=True) as stream:
        np.savez_compressed(stream, values=values, observed=observed, patches=patches, metadata=np.asarray(json.dumps(native(rows), ensure_ascii=True)), names=np.asarray(names))

def load_bundle(path: str | Path) -> dict:
    try:
        with relative_path(path).open("rb") as source, np.load(source, allow_pickle=False) as archive:
            result = {key: archive[key].copy() for key in ("values", "observed", "patches", "names")}
            if result["names"].dtype.kind == "S":
                result["names"] = np.char.decode(result["names"], "utf-8")
            serialized = archive["metadata"].item()
            if isinstance(serialized, bytes):
                serialized = serialized.decode("utf-8")
            if not isinstance(serialized, str):
                raise ValueError("Physical feature metadata must contain a JSON string")
            result["rows"] = parse_json(serialized)
        validate_bundle(result)
    except (BadZipFile, CompressionError, EOFError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid physical feature archive: " + str(error)) from error
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

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
