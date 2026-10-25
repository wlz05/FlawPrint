# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from convert import convert_files
from doctor import environment, compatible
from engine.data import identifier, normalize, portable_image, audit_splits
from engine.formats import Conversion, convert_records
from engine.provenance import conventions, manuscript_status
from engine.storage import read_rows, write_rows, write_json, resolved
from engine.traces import TraceConfig, extract_one
from engine.experiment import bundle_digest
from engine.synthetic import fixture
from engine.validation import preflight

class FormatTests(unittest.TestCase):

    def setUp(self) -> None:
        Path("./tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir="./tmp")
        self.root = Path(self.temporary.name).relative_to(Path.cwd())
        self.records = [{"id": "001", "text": "Report, with\nnew lines and \u4E2D\u6587.", "image": "", "label": 0, "domain": "science", "split": "val", "group": "event1", "patch_signs_valid_after_attack": False}, {"id": "002", "text": "Another report.", "image": "", "label": 1, "domain": "science", "split": "test", "group": "event2", "patch_signs_valid_after_attack": True}]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, content: str, encoding: str = "utf-8") -> Path:
        path = self.root / name
        path.write_text(content, encoding=encoding)
        return path

    def image(self, name: str = "image.png", size: tuple[int, int] = (64, 64)) -> Path:
        path = self.root / name
        Image.fromarray(np.random.default_rng(17).integers(0, 256, (*size, 3), dtype=np.uint8)).save(path)
        return path

    def test_canonical_roundtrip_preserves_text_ids_and_boolean_flags(self) -> None:
        expected = normalize(self.records)
        for suffix in ("json", "jsonl", "ndjson", "csv", "tsv"):
            with self.subTest(suffix=suffix):
                path = self.root / ("manifest." + suffix)
                write_rows(path, expected)
                self.assertEqual(normalize(read_rows(path)), expected)
                if suffix in ("csv", "tsv"):
                    self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_json_array_export_is_a_single_valid_document(self) -> None:
        path = self.root / "data.json"
        write_rows(path, self.records)
        self.assertEqual(json.loads(path.read_text()), self.records)

    def test_json_wrapper_requires_an_explicit_key(self) -> None:
        path = self.write("data.json", json.dumps({"records": self.records}))
        with self.assertRaises(ValueError):
            read_rows(path)
        self.assertEqual(read_rows(path, records_key="records"), self.records)

    def test_jsonl_errors_report_the_line(self) -> None:
        path = self.write("bad.jsonl", '{}\n{"id":\n')
        with self.assertRaisesRegex(ValueError, "line 2"):
            read_rows(path)

    def test_json_rejects_duplicate_keys_and_nonstandard_nan(self) -> None:
        for content in ('[{"id":"a","id":"b"}]', '[{"label":NaN}]'):
            with self.assertRaises(ValueError):
                read_rows(self.write("bad.json", content))

    def test_csv_rejects_duplicate_empty_or_colliding_headers(self) -> None:
        for text in ("id,id\na,b\n", "id, ID \na,b\n", "id,\na,b\n"):
            path = self.write("bad.csv", text)
            with self.assertRaises(ValueError):
                read_rows(path)

    def test_csv_rejects_variable_column_counts(self) -> None:
        for text in ("id,text\na,b,c\n", "id,text\na\n"):
            with self.assertRaises(ValueError):
                read_rows(self.write("bad.csv", text))

    def test_csv_delimiter_can_be_explicit_or_inferred(self) -> None:
        path = self.write("data.csv", "id;text;label\n001;report;0\n002;other;1\n")
        self.assertEqual(read_rows(path, delimiter=";"), read_rows(path, delimiter="auto"))
        tab = self.write("tab.csv", "id\ttext\n001\treport\n")
        self.assertEqual(read_rows(tab, delimiter="\\t")[0]["id"], "001")

    def test_non_utf8_encoding_requires_an_explicit_choice(self) -> None:
        path = self.write("data.csv", "id,text\n001,\u4E2D\u6587\n", "gb18030")
        with self.assertRaises(UnicodeError):
            read_rows(path)
        self.assertEqual(read_rows(path, encoding="gb18030")[0]["text"], "\u4E2D\u6587")

    def test_row_readers_reject_objects_without_string_columns(self) -> None:
        with self.assertRaises(ValueError):
            read_rows(self.write("data.json", '[1, {"id": "a"}]'))

    def test_unknown_extensions_are_not_silently_jsonl(self) -> None:
        with self.assertRaises(ValueError):
            read_rows(self.write("data.txt", '{}\n'))
        with self.assertRaises(ValueError):
            write_rows(self.root / "output.txt", self.records)

    @unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("openpyxl"), "Excel dependencies are not installed")
    def test_excel_preserves_string_identifiers_and_named_sheets(self) -> None:
        import pandas as pd
        path = self.root / "data.xlsx"
        pd.DataFrame(self.records).to_excel(path, sheet_name="samples", index=False)
        self.assertEqual(normalize(read_rows(path, sheet="samples")), normalize(self.records))

    @unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("openpyxl"), "Excel dependencies are not installed")
    def test_excel_duplicate_headers_are_not_mangled_and_accepted(self) -> None:
        import pandas as pd
        path = self.root / "bad.xlsx"
        pd.DataFrame([["id", "id"], ["a", "b"]]).to_excel(path, index=False, header=False)
        with self.assertRaises(ValueError):
            read_rows(path)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is not installed")
    def test_dataframe_pickle_requires_trust_and_a_dataframe(self) -> None:
        import pandas as pd
        path = self.root / "data.pkl"
        pd.DataFrame(self.records).to_pickle(path)
        with self.assertRaises(ValueError):
            read_rows(path)
        self.assertEqual(normalize(read_rows(path, trusted_pickle=True)), normalize(self.records))
        pd.to_pickle([{"id": "a"}], path)
        with self.assertRaises(ValueError):
            read_rows(path, trusted_pickle=True)

    @unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"), "Parquet dependencies are not installed")
    def test_parquet_preserves_the_canonical_contract(self) -> None:
        import pandas as pd
        path = self.root / "data.parquet"
        pd.DataFrame(self.records).to_parquet(path, index=False)
        self.assertEqual(normalize(read_rows(path)), normalize(self.records))

    def test_optional_table_dependency_failure_is_actionable(self) -> None:
        with patch.dict(sys.modules, {"pandas": None}):
            with self.assertRaisesRegex(ValueError, "tables.txt"):
                read_rows(self.root / "data.xlsx")

    def test_finefake_aliases_and_binary_labels_are_converted(self) -> None:
        records = [{"text": "report", "image_path": "photo.jpg", "topic": "science", "label": 1}]
        rows, report = convert_records(records, Conversion(dataset="finefake", image_root="./data/images"))
        self.assertEqual((rows[0]["label"], rows[0]["image"], rows[0]["domain"]), (0, "data/images/photo.jpg", "science"))
        self.assertEqual(report["generated_ids"], 1)

    def test_weibo21_aliases_and_numeric_label_declaration(self) -> None:
        records = [{"content": "report", "category": "science", "label": 1}]
        with self.assertRaisesRegex(ValueError, "Numeric labels"):
            convert_records(records, Conversion(dataset="weibo21"))
        rows, report = convert_records(records, Conversion(dataset="weibo21", labels="fake-one"))
        self.assertFalse(rows[0]["has_image"])
        self.assertEqual(report["columns"]["text"], "content")

    def test_named_labels_do_not_depend_on_numeric_orientation(self) -> None:
        rows, _ = convert_records([{"text": "report", "label": "fake"}], Conversion(dataset="custom"))
        self.assertEqual(rows[0]["label"], 1)

    def test_finegrained_labels_require_an_explicit_mapping(self) -> None:
        records = [{"text": "report", "domain": "science", "label": "3.0"}]
        with self.assertRaises(ValueError):
            convert_records(records, Conversion(dataset="finefake"))
        rows, _ = convert_records(records, Conversion(dataset="finefake", label_map={"3": 1}))
        self.assertEqual(rows[0]["label"], 1)

    def test_unmapped_labels_are_not_silently_dropped(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unmapped label"):
            convert_records(self.records, Conversion(label_map={"0": 0}))

    def test_missing_explicit_columns_are_not_defaulted(self) -> None:
        with self.assertRaisesRegex(ValueError, "Mapped column"):
            convert_records(self.records, Conversion(columns={"text": "missing"}))

    def test_ambiguous_aliases_require_a_mapping(self) -> None:
        records = [{"content": "one", "body": "two"}]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            convert_records(records, Conversion())
        rows, _ = convert_records(records, Conversion(columns={"text": "body"}))
        self.assertEqual(rows[0]["text"], "two")

    def test_missing_ids_are_stable_under_row_permutation(self) -> None:
        records = [{"text": "one"}, {"text": "two"}]
        first, _ = convert_records(records, Conversion())
        second, _ = convert_records(records[::-1], Conversion())
        self.assertEqual({row["text"]: row["id"] for row in first}, {row["text"]: row["id"] for row in second})

    def test_conversion_and_normalization_agree_on_generated_single_image_ids(self) -> None:
        records = [{"text": "report", "image": "photo.jpg", "domain": "science"}]
        rows, _ = convert_records(records, Conversion(image_root="./data/images"))
        self.assertEqual(rows[0]["id"], normalize(records, "./data/images")[0]["id"])

    def test_canonical_conversion_preserves_optional_metadata(self) -> None:
        records = [dict(self.records[0], scenario="real_photo_false_text", details={"version": 1})]
        rows, _ = convert_records(records, Conversion())
        self.assertEqual(rows[0]["scenario"], "real_photo_false_text")
        self.assertEqual(rows[0]["details"], {"version": 1})

    def test_image_only_tables_keep_an_empty_text_field(self) -> None:
        rows, _ = convert_records([{"id": "photo", "image": "photo.jpg"}], Conversion())
        self.assertEqual(rows[0]["text"], "")
        self.assertTrue(rows[0]["has_image"])

    def test_duplicate_content_without_source_ids_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            convert_records([{"text": "one"}, {"text": "one"}], Conversion())

    def test_unsafe_numeric_ids_are_rejected(self) -> None:
        for value in (float(2 ** 53), 1.5, True, ["a"]):
            with self.assertRaises(ValueError):
                identifier(value)
        self.assertEqual(identifier("001"), "001")

    def test_missing_cells_do_not_become_shared_nan_groups(self) -> None:
        rows = normalize([{"id": "a", "text": np.nan, "group": np.nan, "label": np.nan}, {"id": "b", "text": None, "group": None}])
        self.assertEqual([row["group"] for row in rows], ["a", "b"])
        self.assertEqual([row["text"] for row in rows], ["", ""])
        self.assertNotIn("label", rows[0])

    def test_split_aliases_profile_labels_and_domain_maps(self) -> None:
        rows, report = convert_records(self.records, Conversion(split="profile", domain_map={"science": "research"}))
        self.assertEqual(rows[0]["domain"], "research")
        self.assertNotIn("label", rows[0])
        self.assertEqual(report["profile_labels_removed"], 2)

    def test_exclusions_are_reported(self) -> None:
        records = [dict(self.records[0], domain="unknown"), self.records[1]]
        rows, report = convert_records(records, Conversion(exclude_domains=["unknown"]))
        self.assertEqual((len(rows), report["excluded_rows"]), (1, 1))

    def test_multiple_images_are_never_silently_selected(self) -> None:
        records = [{"text": "report", "images": '["one.jpg", "two.jpg"]'}]
        with self.assertRaisesRegex(ValueError, "multiple images"):
            convert_records(records, Conversion())
        rows, report = convert_records(records, Conversion(image_policy="first"))
        self.assertEqual(rows[0]["image"], "one.jpg")
        self.assertEqual(report["discarded_images"], 1)

    def test_image_expansion_preserves_news_group(self) -> None:
        rows, report = convert_records([{"id": "news", "text": "report", "images": ["one.jpg", "two.jpg"]}], Conversion(image_policy="all"))
        self.assertEqual([row["id"] for row in rows], ["news:image1", "news:image2"])
        self.assertEqual({row["group"] for row in rows}, {"news"})
        self.assertEqual(report["sample_unit"], "image-news-pair")

    def test_serialized_lists_and_explicit_separators(self) -> None:
        for value, separator in (("['one.jpg', 'two.jpg']", None), ("one.jpg|two.jpg", "|")):
            rows, _ = convert_records([{"text": "report", "images": value}], Conversion(image_policy="all", image_separator=separator))
            self.assertEqual(len(rows), 2)

    def test_duplicate_or_nonstring_images_are_rejected(self) -> None:
        for paths in (["one.jpg", "one.jpg"], [1, "two.jpg"]):
            with self.assertRaises(ValueError):
                convert_records([{"text": "report", "images": paths}], Conversion(image_policy="all"))

    def test_relative_image_root_is_idempotent_and_portable(self) -> None:
        path = portable_image("data\\images\\photo.jpg", "./data/images")
        self.assertEqual(path, "data/images/photo.jpg")
        self.assertEqual(portable_image(path, "./data/images"), path)

    def test_absolute_source_prefix_removal_is_explicit(self) -> None:
        prefix = "Z:" + "/old/images"
        records = [{"text": "report", "image": prefix + "/photo.jpg"}]
        with self.assertRaises(ValueError):
            convert_records(records, Conversion())
        rows, _ = convert_records(records, Conversion(strip_image_prefix=prefix, image_root="./data/images"))
        self.assertEqual(rows[0]["image"], "data/images/photo.jpg")

    def test_image_urls_are_not_used_as_local_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "Download"):
            convert_records([{"text": "report", "image": "https://example.test/photo.jpg"}], Conversion())

    def test_merge_detects_duplicate_ids_and_cross_file_groups(self) -> None:
        first, second = self.root / "first.jsonl", self.root / "second.csv"
        write_rows(first, [self.records[0]])
        write_rows(second, [dict(self.records[1], id="001")])
        with self.assertRaises(ValueError):
            convert_files([str(first), str(second)], Conversion())
        write_rows(second, [dict(self.records[1], group="event1")])
        with self.assertRaisesRegex(ValueError, "leakage"):
            convert_files([str(first), str(second)], Conversion())

    def test_conversion_report_records_hashes_and_contract(self) -> None:
        path = self.root / "data.csv"
        write_rows(path, self.records)
        rows, report = convert_files([str(path)], Conversion())
        self.assertEqual(report["label_contract"], "zero-real-one-fake")
        self.assertEqual(len(report["inputs"][0]["sha256"]), 64)
        self.assertEqual(rows[0]["source_row"], 1)

    def test_csv_flags_restore_false_without_bypassing_attack_checks(self) -> None:
        path = self.root / "data.csv"
        write_rows(path, self.records)
        rows, _ = convert_records(read_rows(path), Conversion())
        self.assertIs(rows[0]["patch_signs_valid_after_attack"], False)

    def test_preflight_collects_bad_images_and_missing_paths(self) -> None:
        small = self.image("small.png", (16, 16))
        bad = self.write("bad.jpg", "not an image")
        rows = [{"id": str(index), "text": "report", "image": str(path)} for index, path in enumerate((small, bad, self.root / "missing.jpg"))]
        result = preflight(rows, check_images=True)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["errors"]), 3)

    def test_preflight_detects_content_leakage_under_different_paths(self) -> None:
        first = self.image()
        second = self.root / "copied.png"
        second.write_bytes(first.read_bytes())
        rows = [{"id": "a", "image": str(first), "split": "train"}, {"id": "b", "image": str(second), "split": "test"}]
        result = preflight(rows, check_images=True)
        self.assertFalse(result["passed"])
        self.assertTrue(any("content hashes" in error for error in result["errors"]))

    def test_preflight_reports_text_only_and_incomplete_semantic_coverage(self) -> None:
        result = preflight(self.records)
        self.assertTrue(result["passed"])
        self.assertEqual(result["missing_images"], 2)
        self.assertGreaterEqual(len(result["warnings"]), 2)

    def test_preflight_benchmark_requires_classes_labels_and_profiles(self) -> None:
        rows = [{"id": str(index), "domain": "source", "split": split, "label": 0} for index, split in enumerate(("train", "validation", "calibration", "test"))]
        result = preflight(rows, benchmark=True, min_samples=12)
        self.assertFalse(result["passed"])
        self.assertTrue(any("both classes" in error for error in result["errors"]))
        self.assertTrue(any("fewer than 12" in error for error in result["errors"]))

    def test_integer_trace_dimensions_are_required(self) -> None:
        with self.assertRaises(ValueError):
            TraceConfig(tile_grid=2.5).validate()

    def test_multiframe_inputs_do_not_silently_use_the_first_frame(self) -> None:
        path = self.root / "animated.gif"
        Image.new("RGB", (64, 64), "red").save(path, save_all=True, append_images=[Image.new("RGB", (64, 64), "blue")])
        with self.assertRaisesRegex(ValueError, "still-image"):
            extract_one({"id": "a", "image": str(path), "text": "report"}, TraceConfig())
        self.assertFalse(preflight([{"id": "a", "image": str(path)}], check_images=True)["passed"])

    def test_feature_fingerprint_covers_values_patches_masks_and_metadata(self) -> None:
        bundle = fixture(per_split=12)
        original = bundle_digest(bundle)
        for key in ("values", "patches", "observed", "rows"):
            changed = copy.deepcopy(bundle)
            if key == "rows":
                changed[key][0]["text"] = "changed"
            elif key == "observed":
                changed[key][0, 0] = not changed[key][0, 0]
            elif key == "patches":
                changed[key][0, 0, 0] += 1
            else:
                changed[key][0, 0] += 1
            self.assertNotEqual(bundle_digest(changed), original)

    def test_environment_inspection_does_not_import_numeric_packages(self) -> None:
        with patch.dict(sys.modules, {"numpy": None, "scipy": None}):
            result = environment()
        self.assertIn("numpy", result["missing"])
        self.assertFalse(result["passed"])

    def test_environment_rejects_unsupported_dependency_versions(self) -> None:
        self.assertFalse(compatible("numpy", "1.25.9"))
        self.assertFalse(compatible("transformers", "6.0.0"))
        self.assertTrue(compatible("transformers", "5.15.0"))
        self.assertTrue(compatible("torch", "2.5.1+cpu"))

    def test_conversion_cli_rejects_input_overwrite(self) -> None:
        path = self.root / "data.json"
        write_rows(path, self.records)
        original = path.read_bytes()
        result = subprocess.run([sys.executable, "convert.py", str(path), "--dataset", "canonical", "--output", str(path)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(path.read_bytes(), original)
        alias = self.root / ".." / self.root.name / "data.json"
        self.assertEqual(resolved(alias), resolved(path))
        self.assertTrue(resolved(alias).is_file())
        result = subprocess.run([sys.executable, "convert.py", str(path), "--dataset", "canonical", "--output", str(alias)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be distinct", result.stderr)
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(ValueError):
            convert_files([str(path), str(alias)], Conversion())

    def test_manuscript_acceptance_rejects_a_concurrent_change(self) -> None:
        path = self.write("manuscript.md", "First version")
        snapshot = self.root / "accepted.md"
        original = Path.read_bytes
        reads = 0
        def changing(file: Path) -> bytes:
            nonlocal reads
            if file == path:
                reads += 1
                if reads == 2:
                    return b"Second version"
            return original(file)
        with patch.object(Path, "read_bytes", changing):
            with self.assertRaises(RuntimeError):
                manuscript_status(str(path), str(snapshot), accept=True)
        self.assertFalse(snapshot.exists())
        path.write_bytes(b"\xef\xbb\xbfFirst version\r\nSecond line\r\n")
        manuscript_status(str(path), str(snapshot), accept=True)
        self.assertEqual(snapshot.read_bytes(), path.read_bytes())
        self.assertFalse(manuscript_status(str(path), str(snapshot))["changed"])

    def test_dataset_names_are_case_insensitive_in_api_and_cli(self) -> None:
        for name in ("WeIbO", "WEIBO21", "FineFake"):
            with self.subTest(dataset=name):
                records = [{"id": "sample", "text": "A news report.", "label": 1, "domain": "science", "split": "train"}]
                config = Conversion(dataset=name, labels="fake-one")
                rows, _ = convert_records(records, config)
                self.assertEqual(config.dataset, name.lower())
                self.assertEqual(rows[0]["label"], 1)
                source, output = self.root / "source.jsonl", self.root / "converted.jsonl"
                write_rows(source, records)
                result = subprocess.run([sys.executable, "-B", "convert.py", source.as_posix(), "--dataset", name, "--labels", "fake-one", "--output", output.as_posix()], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(read_rows(output)[0]["id"], "sample")

    def test_dataset_mapping_normalizes_names_without_changing_label_direction(self) -> None:
        records = [{"id": "sample", "text": "A genuine report.", "label": 1, "domain": "science", "split": "train"}]
        source, mapping, output = self.root / "source.jsonl", self.root / "mapping.json", self.root / "converted.jsonl"
        write_rows(source, records)
        write_json(mapping, {"dataset": "FiNeFaKe"})
        result = subprocess.run([sys.executable, "-B", "convert.py", source.as_posix(), "--mapping", mapping.as_posix(), "--output", output.as_posix()], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_rows(output)[0]["label"], 0)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
