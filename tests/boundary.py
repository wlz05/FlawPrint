# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
from argparse import Namespace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from engine.conformal import calibrate_rows, apply_calibration, validate_partitions, validate_labels
from engine.data import normalize
from engine.decision import GatePolicy, route_rows, calibrate_routes, apply_routes, deploy
from engine.evidence import paired, stratified, transitions
from engine.experiment import indices
from engine.metrics import evaluate, group_bootstrap
from engine.profile import DomainProfile, ProfileConfig
from engine.provenance import conventions, manuscript_title
from engine.results import prediction_rows
from engine.storage import read_json, write_json, read_rows, write_rows, same_path, subset
from engine.stream import adapt, pool
from engine.synthetic import fixture

class BoundaryTests(unittest.TestCase):

    def setUp(self) -> None:
        Path("./tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir="./tmp")
        self.root = Path(self.temporary.name).relative_to(Path.cwd())
        self.validation = self.rows("v", "validation")
        self.calibration = self.rows("c", "calibration")
        self.test = self.rows("t", "test")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def rows(self, prefix: str, split: str) -> list[dict]:
        return [{"id": f"{prefix}{index}", "domain": "a", "group": f"{prefix}event{index}", "split": split, "label": index % 2, "fusion_risk": 2 * (index % 2) - 1} for index in range(4)]

    def profile(self, domain: str = "target_gamma") -> tuple[DomainProfile, dict]:
        bundle = fixture(per_split=12)
        data = subset(bundle, indices(bundle, "profile" if domain.startswith("target") else "train", domain))
        return DomainProfile.fit(data, domain, ProfileConfig(min_samples=12, bootstrap=2), [0, 1, 2]), data

    def test_boolean_numeric_fields_are_rejected(self) -> None:
        for field in ("label", "prediction", "probability", "fusion_risk"):
            for value in (True, np.bool_(False)):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    prediction_rows([{field: value}])

    def test_boolean_manifest_labels_are_rejected(self) -> None:
        for value in (True, np.bool_(False)):
            with self.assertRaises(ValueError):
                normalize([{"id": "a", "label": value}])
        with self.assertRaises(ValueError):
            validate_labels(np.array([True, False]))

    def test_nonscalar_scores_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            prediction_rows([{"fusion_risk": [1]}])

    def test_split_aliases_cannot_bypass_calibration_guard(self) -> None:
        validation = [dict(row, split=" Testing ") for row in self.validation]
        with self.assertRaises(ValueError):
            calibrate_rows(validation, self.calibration)

    def test_false_csv_overlap_flags_remain_false(self) -> None:
        validation = [dict(row, profile_overlap="False") for row in self.validation]
        self.assertIn("evidence_history", calibrate_rows(validation, self.calibration))

    def test_missing_bootstrap_groups_are_rejected(self) -> None:
        for value in (None, "", " NaN ", np.nan):
            rows = [dict(row, domain=value) for row in self.test]
            with self.subTest(value=value), self.assertRaises(ValueError):
                paired(rows, rows, repeats=20)
            with self.assertRaises(ValueError):
                group_bootstrap(rows, repeats=2)

    def test_zero_event_identifiers_are_not_discarded(self) -> None:
        validation, calibration = copy.deepcopy(self.validation), copy.deepcopy(self.calibration)
        validation[0]["group"], calibration[0]["group"] = 0, "0"
        with self.assertRaises(ValueError):
            calibrate_rows(validation, calibration)

    def test_parent_path_aliases_are_disjointness_violations(self) -> None:
        validation, calibration = copy.deepcopy(self.validation), copy.deepcopy(self.calibration)
        validation[0]["image"], calibration[0]["image"] = "./assets/../shared.jpg", "./shared.jpg"
        with self.assertRaises(ValueError):
            calibrate_rows(validation, calibration)

    def test_saved_calibration_blocks_new_ids_with_old_evidence(self) -> None:
        for field, value in (("group", "shared"), ("image", "./assets/../shared.jpg"), ("content_sha256", "content")):
            validation = copy.deepcopy(self.validation)
            validation[0][field] = value
            state = calibrate_rows(validation, self.calibration)
            path = self.root / "state.json"
            write_json(path, state)
            row = dict(self.test[0], **{field: "./shared.jpg" if field == "image" else value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                apply_calibration([row], read_json(path))

    def test_legacy_calibration_reports_identifier_only_scope(self) -> None:
        state = calibrate_rows(self.validation, self.calibration)
        state.pop("evidence_history")
        self.assertEqual(apply_calibration(self.test, state)[0]["calibration_overlap_audit"], "identifiers_only")

    def test_malformed_history_is_rejected(self) -> None:
        state = calibrate_rows(self.validation, self.calibration)
        for history in ({}, {"group": [None], "image": [], "content_sha256": []}, {"group": [], "image": ["./a.jpg", "a.jpg"], "content_sha256": []}):
            with self.subTest(history=history), self.assertRaises(ValueError):
                apply_calibration(self.test, dict(state, evidence_history=history))

    def test_nonoverlapping_state_preserves_scores(self) -> None:
        state = calibrate_rows(self.validation, self.calibration)
        result = apply_calibration(self.test, state)
        self.assertEqual([row["fusion_risk"] for row in result], [row["fusion_risk"] for row in self.test])
        self.assertEqual(result[0]["calibration_overlap_audit"], "identifiers_and_recorded_evidence")

    def test_custom_csv_scores_without_predictions_work(self) -> None:
        rows = [{key: value for key, value in dict(row, custom=str(row["fusion_risk"])).items() if key != "fusion_risk"} for row in self.test]
        path = self.root / "scores.csv"
        write_rows(path, rows)
        loaded = read_rows(path)
        annotations = [{"id": row["id"]} for row in rows]
        report = stratified(loaded, loaded, annotations, "custom", repeats=20)
        self.assertEqual(report["transitions"]["unchanged_correct"], 4)

    def test_custom_nonfinite_and_boolean_scores_are_rejected(self) -> None:
        for value in (True, "inf", [1]):
            rows = [dict(row, custom=value) for row in self.test]
            with self.subTest(value=value), self.assertRaises(ValueError):
                paired(rows, rows, "custom", repeats=20)

    def test_transition_alignment_cannot_silently_truncate(self) -> None:
        for candidate in (self.test[:-1], [dict(row, id="other" + row["id"]) for row in self.test], [dict(row, label=1 - row["label"]) for row in self.test]):
            with self.assertRaises(ValueError):
                transitions(self.test, candidate, "fusion_risk")

    def test_all_abstentions_have_undefined_transition_rates(self) -> None:
        rows = [dict(row, abstain="True") for row in self.test]
        report = transitions(rows, rows, "fusion_risk")
        self.assertEqual(report["joint_decided"], 0)
        self.assertIsNone(report["matched_accuracy_delta"])

    def test_partial_probability_coverage_is_explicit(self) -> None:
        rows = copy.deepcopy(self.test)
        rows[0]["probability"] = 0.2
        report = evaluate(rows)
        self.assertEqual(report["probability_coverage"]["observed"], 1)
        self.assertEqual(report["probability_coverage"]["fraction"], 0.25)
        self.assertFalse(report["probability_coverage"]["metrics_available"])
        self.assertNotIn("brier", report)

    def test_one_group_does_not_produce_spurious_cluster_intervals(self) -> None:
        report = group_bootstrap(self.test, repeats=20)
        self.assertEqual(report["independent_groups"], 1)
        self.assertEqual(report["intervals"], {})

    def test_route_change_cannot_hide_global_history_overlap(self) -> None:
        status = {"physical_available": False, "physical_gate_reasons": []}
        def routed(rows):
            return route_rows([dict(row, semantic_logit=row["fusion_risk"]) for row in rows], [status] * len(rows), GatePolicy())
        state = calibrate_routes(routed(self.validation), routed(self.calibration))
        row = routed([dict(self.test[0], group=self.validation[0]["group"])])[0]
        row["decision_route"] = "physical"
        with self.assertRaises(ValueError):
            apply_routes([row], state)

    def test_profile_roundtrip_flags_new_id_evidence_reuse(self) -> None:
        profile, data = self.profile()
        path = self.root / "profile.json"
        profile.save(path.as_posix())
        restored = DomainProfile.load(path.as_posix())
        for field in ("group", "image"):
            query = subset(data, [0])
            query["rows"][0].update(id="new", group="new-event", image="./new.jpg", split="test")
            query["rows"][0][field] = data["rows"][0][field]
            self.assertTrue(restored.predict(query)[0]["profile_overlap"])
            with self.assertRaises(ValueError):
                deploy(query, restored, GatePolicy())

    def test_profile_content_history_survives_loading(self) -> None:
        profile, data = self.profile()
        data["rows"][0]["content_sha256"] = "same-content"
        profile = DomainProfile.fit(data, profile.domain, profile.config, profile.columns)
        path = self.root / "profile.json"
        profile.save(path.as_posix())
        query = subset(data, [0])
        query["rows"][0].update(id="new", group="new-event", image="./new.jpg", split="test")
        self.assertTrue(DomainProfile.load(path.as_posix()).predict(query)[0]["profile_overlap"])

    def test_legacy_profile_reports_limited_history_scope(self) -> None:
        profile, data = self.profile()
        state = profile.document()
        state.pop("evidence_history")
        path = self.root / "legacy.json"
        write_json(path, state)
        self.assertEqual(DomainProfile.load(path.as_posix()).predict(data)[0]["profile_overlap_audit"], "identifiers_only")

    def test_test_alias_cannot_be_used_to_fit_profile(self) -> None:
        profile, data = self.profile()
        data["rows"][0]["split"] = " Testing "
        with self.assertRaises(ValueError):
            DomainProfile.fit(data, profile.domain, profile.config, profile.columns)

    def test_streaming_rejects_new_id_event_reuse(self) -> None:
        profile, data = self.profile()
        query = subset(data, [0])
        query["rows"][0].update(id="new", image="./new.jpg", domain="new-domain", split="profile")
        with self.assertRaises(ValueError):
            adapt(profile, query, "new-domain")

    def test_short_streaming_update_records_new_history(self) -> None:
        profile, data = self.profile()
        query = subset(data, [0])
        query["rows"][0].update(id="new", group="new-event", image="./new.jpg", domain="new-domain", split="profile")
        updated = adapt(profile, query, "new-domain")
        replay = copy.deepcopy(query)
        replay["rows"][0]["id"] = "other"
        with self.assertRaises(ValueError):
            adapt(updated, replay, "new-domain")

    def test_pooling_rejects_shared_events_under_distinct_ids(self) -> None:
        first, _ = self.profile("source_alpha")
        second, data = self.profile("source_beta")
        data["rows"][0]["group"] = first.history["group"][0]
        second = DomainProfile.fit(data, second.domain, second.config, second.columns)
        with self.assertRaises(ValueError):
            pool([first, second])

    def test_genuine_threshold_history_blocks_calibration_reuse(self) -> None:
        profile, _ = self.profile()
        bundle = fixture(per_split=12)
        validation = subset(bundle, indices(bundle, "validation", "source_alpha"))
        profile.calibrate_genuine(validation)
        selected = next(index for index, row in enumerate(validation["rows"]) if row["label"] == 0)
        query = subset(validation, [selected])
        query["rows"][0].update(id="new", split="test")
        path = self.root / "genuine.json"
        profile.save(path.as_posix())
        self.assertTrue(DomainProfile.load(path.as_posix()).predict(query)[0]["profile_overlap"])

    def test_failed_jsonl_serialization_preserves_previous_output(self) -> None:
        path = self.root / "predictions.jsonl"
        write_rows(path, [{"id": "old"}])
        original = path.read_bytes()
        with self.assertRaises(TypeError):
            write_rows(path, [{"id": "new"}, {"invalid": object()}])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_failed_atomic_replacement_cleans_temporary_output(self) -> None:
        path = self.root / "state.json"
        write_json(path, {"value": "old"})
        original = path.read_bytes()
        with patch("engine.storage.os.replace", side_effect=OSError("busy")), self.assertRaises(OSError):
            write_json(path, {"value": "new"})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_empty_csv_export_fails_before_overwriting(self) -> None:
        path = self.root / "rows.csv"
        write_rows(path, [{"id": "old"}])
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            write_rows(path, [])
        self.assertEqual(path.read_bytes(), original)
        write_rows(self.root / "empty.json", [])
        self.assertEqual(read_rows(self.root / "empty.json"), [])

    def test_hardlinked_input_is_protected_by_research_cli(self) -> None:
        source, output = self.root / "source.jsonl", self.root / "alias.jsonl"
        write_rows(source, self.test)
        try:
            os.link(source, output)
        except OSError as error:
            self.skipTest("Hard links are unavailable: " + str(error))
        self.assertTrue(same_path(source, output))
        before = source.read_bytes()
        result = subprocess.run([sys.executable, "-B", "research.py", "compare", source.as_posix(), source.as_posix(), "--output", output.as_posix(), "--scope", "synthetic", "--dataset", "boundary", "--repeats", "20"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot overwrite an input", result.stderr)
        self.assertEqual(source.read_bytes(), before)

    def test_header_exception_requires_exactly_one_blank_line(self) -> None:
        path = self.root / "sample.py"
        header = "# -*- coding: utf-8 -*-\n# " + manuscript_title() + "\n"
        path.write_text(header + "\nvalue = 1\n\n# Author: Weiliang Zhu\n# Email: wlzchina05@gmail.com\n", encoding="utf-8")
        self.assertTrue(conventions(self.root.as_posix())["passed"])
        path.write_text(header + "\nfirst = 1\n\nsecond = 2\n\n# Author: Weiliang Zhu\n# Email: wlzchina05@gmail.com\n", encoding="utf-8")
        self.assertTrue(conventions(self.root.as_posix())["passed"])
        path.write_text(header + "\n\nvalue = 1\n\n# Author: Weiliang Zhu\n# Email: wlzchina05@gmail.com\n", encoding="utf-8")
        self.assertTrue(any(issue["rule"] == "blank-line" for issue in conventions(self.root.as_posix())["issues"]))

    def test_header_cannot_substitute_the_project_abbreviation(self) -> None:
        path = self.root / "sample.py"
        path.write_text("# -*- coding: utf-8 -*-\n# FlawPrint\n\nvalue = 1\n\n# Author: Weiliang Zhu\n# Email: wlzchina05@gmail.com\n", encoding="utf-8")
        self.assertTrue(any(issue["rule"] == "title-header" for issue in conventions(self.root.as_posix())["issues"]))

    def test_failed_code_audit_cannot_accept_a_manuscript_snapshot(self) -> None:
        from main import command_audit
        manuscript, snapshot = self.root / "paper.md", self.root / "snapshot.md"
        manuscript.write_text("# Official English Title\n", encoding="utf-8")
        (self.root / "sample.py").write_text("# -*- coding: utf-8 -*-\n# Wrong Title\n\nvalue = 1\n\n# Author: Weiliang Zhu\n# Email: wlzchina05@gmail.com\n", encoding="utf-8")
        args = Namespace(root=self.root.as_posix(), manuscript=manuscript.as_posix(), snapshot=snapshot.as_posix(), accept=True, manifest=None, output=(self.root / "audit.json").as_posix())
        with self.assertRaises(ValueError):
            command_audit(args)
        self.assertFalse(snapshot.exists())
        self.assertFalse(read_json(args.output)["conventions"]["passed"])

    def test_public_audit_uses_readme_and_local_snapshot(self) -> None:
        from engine.provenance import manuscript_status
        from main import command_audit
        title = manuscript_title()
        (self.root / "README.md").write_text("# " + title + "\n", encoding="utf-8")
        (self.root / "sample.py").write_text("# -*- coding: utf-8 -*-\n# " + title + "\n\nvalue = 1\n\n# Author: Weiliang Zhu\n# Email: wlzchina05@gmail.com\n", encoding="utf-8")
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            report = conventions()
            self.assertTrue(report["passed"], report["issues"])
            state = manuscript_status(accept=True)
            self.assertEqual(state["manuscript"], "README.md")
            self.assertEqual(Path("./outputs/audit_snapshot.md").read_bytes(), Path("README.md").read_bytes())
            args = Namespace(root=".", manuscript=None, snapshot=None, accept=False, manifest=None, output="./README.md")
            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                command_audit(args)
            self.assertTrue(Path("README.md").read_text().startswith("# " + title))
        finally:
            os.chdir(previous)

    def test_missing_explicit_document_does_not_fall_back(self) -> None:
        with self.assertRaises(FileNotFoundError):
            manuscript_title((self.root / "missing.md").as_posix())

    def test_encoding_declaration_is_required(self) -> None:
        (self.root / "sample.py").write_text("# " + manuscript_title() + "\n\nvalue = 1\n\n# Author: Weiliang Zhu\n# Email: wlzchina05@gmail.com\n", encoding="utf-8")
        self.assertTrue(any(issue["rule"] == "utf8-header" for issue in conventions(self.root.as_posix())["issues"]))

    def test_snapshot_metadata_cannot_overwrite_document(self) -> None:
        from engine.provenance import manuscript_status
        document = self.root / "snapshot.json"
        document.write_text("# Official English Title\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            manuscript_status(document.as_posix(), (self.root / "snapshot.md").as_posix(), True)
        self.assertEqual(document.read_text(), "# Official English Title\n")

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
