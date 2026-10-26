# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import replace
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
from PIL import Image
from engine.conformal import validate_labels, Platt, SplitConformal, fuse
from engine.copula import matrix, decompose
from engine.data import normalize, audit_splits
from engine.metrics import binary, valid_probabilities
from engine.profile import ProfileConfig
from engine.results import prediction_rows, numeric
from engine.storage import save_bundle, load_bundle, write_rows, read_rows, digest
from engine.synthetic import fixture
from engine.traces import extract_one, TraceConfig
from engine.validation import preflight

class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        Path("./tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir="./tmp")
        self.root = Path(self.temporary.name).relative_to(Path.cwd())
        self.bundle = fixture(seed=42, per_split=12)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_physical_cache_failure_preserves_previous_contents(self) -> None:
        path = self.root / "physical.npz"
        bundle = self.bundle
        save_bundle(path, bundle["values"], bundle["observed"], bundle["patches"], bundle["rows"], bundle["names"].tolist())
        previous = path.read_bytes()
        with patch("engine.storage.np.savez_compressed", side_effect=OSError("Interrupted physical write")):
            with self.assertRaises(OSError):
                save_bundle(path, bundle["values"], bundle["observed"], bundle["patches"], bundle["rows"], bundle["names"].tolist())
        self.assertEqual(path.read_bytes(), previous)
        np.testing.assert_array_equal(load_bundle(path)["values"], bundle["values"])
        self.assertFalse(list(self.root.glob(".flawprint-*.tmp")))

    def test_truncated_physical_archives_close_their_file_handles(self) -> None:
        path = self.root / "broken.npz"
        path.write_bytes(b"PK\x03\x04" + b"truncated archive")
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ResourceWarning)
            with self.assertRaisesRegex(ValueError, "Invalid physical feature archive"):
                load_bundle(path)
            gc.collect()
        self.assertFalse([item for item in recorded if issubclass(item.category, ResourceWarning)])
        path.unlink()

    def test_mixed_boolean_labels_are_not_coerced_to_numeric_classes(self) -> None:
        for labels in ([0, True], [False, 1], np.array([0, True], dtype=object)):
            with self.subTest(labels=str(labels)):
                with self.assertRaises(ValueError):
                    validate_labels(labels)
                with self.assertRaises(ValueError):
                    binary(labels, [-1., 1.])
        np.testing.assert_array_equal(validate_labels([0, 1]), [0, 1])

    def test_complex_inputs_are_rejected_without_discarding_imaginary_parts(self) -> None:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            operations = [
                lambda: validate_labels(np.array([0 + 2j, 1 + 3j])),
                lambda: normalize([{"id": "a", "label": np.complex128(1 + 2j)}]),
                lambda: normalize([{"id": "a", "semantic_logit": np.complex128(1 + 2j)}]),
                lambda: numeric(np.complex128(1 + 2j), "fusion_risk"),
                lambda: binary([0, 1], np.array([-1 + 2j, 1 + 3j])),
                lambda: valid_probabilities(np.array([0.2 + 1j, 0.8 + 1j]), np.array([0, 1])),
                lambda: matrix(np.array([[1 + 2j], [2 + 3j]])),
                lambda: decompose(np.array([1 + 2j, 2 + 3j]), np.eye(2), ["a", "b"]),
                lambda: fuse(np.array([1 + 2j])),
            ]
            for operation in operations:
                with self.assertRaises(ValueError):
                    operation()
        self.assertFalse(recorded)

    def test_neural_diagnostics_keep_boolean_and_numeric_types_in_csv(self) -> None:
        rows = [{"id": "sample", "domain": "a", "physical_reference_available": False, "embedding_protocol_verified": False, "potential_disabled_risk": -0.5, "potential_risk_delta": 0.25}]
        for suffix in ("csv", "tsv", "jsonl"):
            path = self.root / ("diagnostics." + suffix)
            write_rows(path, rows)
            restored = prediction_rows(read_rows(path))[0]
            self.assertIs(restored["physical_reference_available"], False)
            self.assertIs(restored["embedding_protocol_verified"], False)
            self.assertEqual(restored["potential_disabled_risk"], -0.5)
            self.assertEqual(restored["potential_risk_delta"], 0.25)

    def test_prediction_diagnostics_reject_nonfinite_and_boolean_scores(self) -> None:
        for key in ("potential_disabled_risk", "potential_risk_delta"):
            for value in (float("inf"), True):
                with self.assertRaises(ValueError):
                    prediction_rows([{"id": "a", key: value}])

    def test_profile_and_calibration_numeric_settings_reject_booleans(self) -> None:
        for name in ("shrinkage", "local_shrinkage", "min_observed", "sign_tolerance", "sign_weight", "kappa"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    replace(ProfileConfig(), **{name: True}).validate()
        for factory in (lambda: Platt(True, 0), lambda: Platt(1, False), lambda: SplitConformal(.1, True, 1, ["a"])):
            with self.assertRaises(ValueError):
                factory()

    def test_equivalent_image_paths_cannot_cross_partitions(self) -> None:
        path = self.root / "image.png"
        Image.new("RGB", (64, 64), (80, 100, 120)).save(path)
        alias = Path("..") / Path.cwd().name / path
        rows = normalize([{"id": "first", "split": "train", "image": path.as_posix()}, {"id": "second", "split": "test", "image": alias.as_posix()}])
        with self.assertRaisesRegex(ValueError, "Cross-split leakage"):
            audit_splits(rows)
        self.assertFalse(preflight(rows)["passed"])

    def test_stale_manifest_checksum_fails_preflight_and_physical_extraction(self) -> None:
        path = self.root / "image.png"
        Image.new("RGB", (64, 64), (80, 100, 120)).save(path)
        row = normalize([{"id": "sample", "domain": "a", "split": "test", "image": path.as_posix(), "content_sha256": "0" * 64}])[0]
        report = preflight([row], check_images=True)
        self.assertFalse(report["passed"])
        self.assertTrue(any("checksum" in error for error in report["errors"]))
        with self.assertRaisesRegex(ValueError, "checksum"):
            extract_one(row, TraceConfig())
        row["content_sha256"] = digest(path)
        self.assertTrue(preflight([row], check_images=True)["passed"])
        self.assertEqual(extract_one(row, TraceConfig())[3]["content_sha256"], row["content_sha256"])

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
