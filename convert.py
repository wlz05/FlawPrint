# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import argparse
from pathlib import Path
from engine.storage import digest, read_json, read_rows, relative_path, resolved, same_path, write_json, write_rows
from engine.data import audit_splits, normalize
from engine.formats import Conversion, FIELDS, convert_records

def convert_files(inputs: list[str], config: Conversion, encoding: str = "utf-8-sig", delimiter: str | None = None, records_key: str | None = None, sheet: str | int = 0, trusted_pickle: bool = False, check_files: bool = False) -> tuple[list[dict], dict]:
    paths = [relative_path(item) for item in inputs]
    if not paths or any(same_path(first, second) for index, first in enumerate(paths) for second in paths[index + 1:]):
        raise ValueError("Input files must be nonempty and distinct")
    rows, reports = [], []
    for path in paths:
        before = digest(path)
        records = read_rows(path, encoding, delimiter, records_key, sheet, trusted_pickle)
        converted, report = convert_records(records, config, path.as_posix())
        if digest(path) != before:
            raise ValueError("An input table changed during conversion; rerun with a stable copy")
        rows.extend(converted)
        reports.append(dict(report, file=path.as_posix(), sha256=before))
    rows = normalize(rows)
    if not rows:
        raise ValueError("No samples remain after conversion")
    audit = audit_splits(rows, check_files)
    return rows, {"version": "FlawPrint conversion 1", "configuration": config.state(), "reader": {"encoding": encoding, "delimiter": delimiter, "records_key": records_key, "sheet": sheet, "trusted_pickle": trusted_pickle}, "inputs": reports, "samples": len(rows), "audit": audit, "label_contract": "zero-real-one-fake", "path_base": "process working directory"}

def arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Convert dataset tables to an audited portable manifest.")
    parser.add_argument("input", nargs="+")
    parser.add_argument("--dataset", type=str.lower, choices=("canonical", "weibo", "finefake", "weibo21", "custom"))
    parser.add_argument("--mapping", help="Load a Conversion configuration from relative JSON")
    parser.add_argument("--trusted-pickle", action="store_true")
    parser.add_argument("--image-root")
    for field in FIELDS:
        parser.add_argument("--" + field.replace("_", "-") + "-column", dest=field + "_column")
    parser.add_argument("--labels", choices=("fake-one", "real-one"))
    for field in ("label", "domain", "split"):
        parser.add_argument("--" + field + "-map", help="Load a value mapping from relative JSON")
    parser.add_argument("--split", choices=("train", "validation", "calibration", "profile", "test"))
    parser.add_argument("--images", choices=("error", "first", "all"))
    parser.add_argument("--image-separator")
    parser.add_argument("--strip-image-prefix")
    parser.add_argument("--id-mode", choices=("source", "content"))
    parser.add_argument("--id-prefix")
    parser.add_argument("--exclude-domain", action="append")
    parser.add_argument("--keep-column", action="append")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--delimiter")
    parser.add_argument("--records-key")
    parser.add_argument("--sheet", default="0")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--output", default="./data/converted.jsonl")
    parser.add_argument("--report")
    return parser, parser.parse_args()

def main() -> None:
    parser, args = arguments()
    try:
        state = read_json(args.mapping) if args.mapping else {}
        if not isinstance(state, dict):
            raise ValueError("The conversion mapping must be a JSON object")
        state.setdefault("dataset", "canonical")
        for field in ("dataset", "image_root", "labels", "split", "image_separator", "strip_image_prefix", "id_mode", "id_prefix"):
            value = getattr(args, field)
            if value is not None:
                state[field] = value
        if args.images:
            state["image_policy"] = args.images
        for field in ("exclude_domains", "keep_columns"):
            value = args.exclude_domain if field == "exclude_domains" else args.keep_column
            if value is not None:
                state[field] = value
        columns = dict(state.get("columns", {}))
        for field in FIELDS:
            value = getattr(args, field + "_column")
            if value is not None:
                columns[field] = value
        state["columns"] = columns
        for field in ("label", "domain", "split"):
            value = getattr(args, field + "_map")
            if value:
                state[field + "_map"] = read_json(value)
        config = Conversion(**state)
        output = relative_path(args.output)
        report = relative_path(args.report) if args.report else output.with_name(output.stem + ".conversion.json")
        protected = args.input + [item for item in (args.mapping, args.label_map, args.domain_map, args.split_map) if item]
        if same_path(output, report) or any(same_path(item, target) for item in protected for target in (output, report)):
            raise ValueError("Input, output, and conversion report files must be distinct")
        sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
        rows, result = convert_files(args.input, config, args.encoding, args.delimiter, args.records_key, sheet, args.trusted_pickle, args.check_files)
        write_rows(output, rows)
        result["output"] = {"file": output.as_posix(), "sha256": digest(output)}
        write_json(report, result)
        print(f"Converted {len(rows)} samples; report: {report.as_posix()}")
    except (ValueError, TypeError, OSError, UnicodeError, KeyError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
