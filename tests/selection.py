# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from engine.selection import select, fit, filtering, retained_contamination, selection_bias
from engine.channels import compare
from engine.synthetic import fixture
from engine.storage import subset
from engine.experiment import indices
from engine.profile import DomainProfile, ProfileConfig
from engine.lineage import collect
from engine.stream import pool, adapt

class SelectionTests(unittest.TestCase):

    def setUp(self) -> None:
        bundle = fixture(per_split=48)
        self.pool = subset(bundle, indices(bundle, "profile", "target_gamma"))
        for index, row in enumerate(self.pool["rows"]):
            row["label"] = index % 2
        self.test = subset(bundle, indices(bundle, "test", "target_gamma"))
        self.config = ProfileConfig(bootstrap=3, min_samples=8)
        self.metadata = {"version": "flawprint-selector-v1", "frozen": True, "task": "news_veracity", "calibration_scope": "source_validation", "score_origin": "external_news_model", "model": "fixture", "checkpoint_sha256": "a" * 64, "fit_identifiers": ["independent-source"], "fit_history": collect([{"id": "independent-source", "group": "selector-fit"}])}
        self.scores = [{"id": row["id"], "fake_probability": 0.01 if index % 2 == 0 else 0.9} for index, row in enumerate(self.pool["rows"])]

    def test_pool_labels_cannot_change_selection_or_reference(self) -> None:
        altered = copy.deepcopy(self.pool)
        for row in altered["rows"]:
            row["label"] = 1 - row["label"]
        first, second = [fit(pool, self.scores, self.metadata, "target_gamma", self.config) for pool in (self.pool, altered)]
        np.testing.assert_array_equal(first.reference, second.reference)
        self.assertEqual(first.quality["reference_selection"], second.quality["reference_selection"])
        selected, _ = select(altered, self.scores, self.metadata)
        self.assertTrue(all("label" not in row for row in selected["rows"]))

    def test_rejected_pool_and_selector_training_cannot_be_replayed(self) -> None:
        profile = fit(self.pool, self.scores, self.metadata, "target_gamma", self.config)
        self.assertTrue(all(profile.overlap(self.pool["rows"])))
        self.assertEqual(profile.overlap([{"id": "independent-source"}]), [True])
        self.assertFalse(any(profile.overlap(self.test["rows"])))

    def test_roundtrip_retains_selection_permissions(self) -> None:
        profile = fit(self.pool, self.scores, self.metadata, "target_gamma", self.config)
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = (Path(directory).resolve().relative_to(Path.cwd()) / "profile.json").as_posix()
            profile.save(path)
            restored = DomainProfile.load(path)
        self.assertEqual(restored.quality["reference_selection"], profile.quality["reference_selection"])
        np.testing.assert_array_equal(restored.reference, profile.reference)

    def test_selected_reference_cannot_be_bypassed_by_streaming(self) -> None:
        profile = fit(self.pool, self.scores, self.metadata, "target_gamma", self.config)
        incoming = copy.deepcopy(self.test)
        for row in incoming["rows"]:
            row["split"] = "profile"
        with self.assertRaisesRegex(ValueError, "Reselect"):
            adapt(profile, incoming, "target_gamma")
        frozen = adapt(profile, None, "target_gamma")
        np.testing.assert_array_equal(frozen.reference, profile.reference)
        self.assertEqual(frozen.quality["reference_selection"], profile.quality["reference_selection"])

    def test_selected_reference_cannot_lose_selector_history_in_pooling(self) -> None:
        profile = fit(self.pool, self.scores, self.metadata, "target_gamma", self.config)
        with self.assertRaisesRegex(ValueError, "joint selection"):
            pool([profile])

    def test_insufficient_selection_does_not_relax_threshold(self) -> None:
        scores = [dict(row, fake_probability=0.99) for row in self.scores]
        with self.assertRaisesRegex(ValueError, "Too few"):
            fit(self.pool, scores, self.metadata, "target_gamma", self.config)
        selected, report = select(self.pool, scores, self.metadata)
        self.assertEqual(len(selected["rows"]), 0)
        self.assertEqual(report["selected_samples"], 0)

    def test_prediction_identity_mismatch_and_duplicates_are_rejected(self) -> None:
        for scores in (self.scores[:-1], self.scores + [self.scores[0]], [dict(row, id="wrong" if index == 0 else row["id"]) for index, row in enumerate(self.scores)]):
            with self.assertRaises(ValueError):
                select(self.pool, scores, self.metadata)

    def test_source_fit_evidence_overlap_is_rejected(self) -> None:
        metadata = dict(self.metadata, fit_history=collect(self.pool["rows"]))
        with self.assertRaises(ValueError):
            select(self.pool, self.scores, metadata)

    def test_forgery_probability_cannot_select_news_genuine_references(self) -> None:
        for metadata in (dict(self.metadata, task="image_forgery"), dict(self.metadata, score_origin="physical_score"), dict(self.metadata, frozen=False)):
            with self.assertRaises(ValueError):
                select(self.pool, self.scores, metadata)

    def test_test_rows_and_nonfinite_confidence_are_rejected(self) -> None:
        changed = copy.deepcopy(self.pool)
        changed["rows"][0]["split"] = "test"
        with self.assertRaises(ValueError):
            select(changed, self.scores, self.metadata)
        for value in (np.nan, np.inf, True, -0.1, 1.1, "0.01"):
            scores = copy.deepcopy(self.scores)
            scores[0]["fake_probability"] = value
            with self.assertRaises(ValueError):
                select(self.pool, scores, self.metadata)

    def test_high_prior_and_wrong_selector_remain_failure_cases(self) -> None:
        self.assertEqual(retained_contamination(1.0, 0.8, 0.01), 1.0)
        self.assertGreater(retained_contamination(0.9, 0.8, 0.05), 0.3)
        self.assertGreater(retained_contamination(0.7, 0.1, 0.9), 0.7)
        self.assertIsNone(retained_contamination(0.5, 0, 0))
        self.assertLess(retained_contamination(0.7, 0.8, 0.01), 0.1)

    def test_offline_labels_are_used_only_for_composition_and_evaluation(self) -> None:
        report = filtering(self.pool, self.test, self.scores, self.metadata, "target_gamma", self.config, [0.5], size=16, repeats=1)
        self.assertEqual(len(report["results"]), 1)
        item = report["results"][0]
        self.assertEqual(item["actual_fake_fraction"], 0.5)
        self.assertEqual(item["methods"]["unfiltered"]["status"], "scored")
        self.assertIn(item["methods"]["filtered"]["status"], ("scored", "unavailable"))

    def test_dimension_diagnostic_exposes_rank_ceiling(self) -> None:
        profile = DomainProfile.fit(self.pool, "target_gamma", self.config)
        dimension = len(profile.columns)
        self.assertEqual(profile.quality["correlation_parameters"], dimension * (dimension - 1) // 2)
        self.assertEqual(profile.quality["unregularized_rank_bound"], min(dimension, len(self.pool["rows"]) - 1))

    def test_better_purity_can_coexist_with_class_conditional_bias(self) -> None:
        values = np.array([[-2, -2], [-1, -1], [1, -1], [2, -2], [-2, 2], [-1, 1], [1, -1], [2, -2]], dtype=float)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        admitted = np.array([True, True, True, False, True, False, False, False])
        report = selection_bias(values, labels, admitted)
        self.assertLess(labels[admitted].mean(), labels.mean())
        self.assertGreater(report["classes"]["0"]["covariance_shift_frobenius"], 0)
        self.assertIsNone(report["classes"]["1"]["covariance_shift_frobenius"])

class ChannelTests(unittest.TestCase):

    def setUp(self) -> None:
        self.channels = [[], [], []]
        rng = np.random.default_rng(42)
        for split, domain in (("validation", "source"), ("calibration", "source"), ("test", "target")):
            for index in range(32):
                label = index % 2
                row = {"id": f"{split}-{index}", "group": f"{split}-{index // 4}", "domain": domain, "platform": "fixture", "split": split, "label": label}
                values = np.array([0.4, 0.6, 0.8]) * (2 * label - 1) + rng.normal(size=3)
                for channel, value in zip(self.channels, values):
                    channel.append(dict(row, fusion_risk=float(value)))
        self.provenance = {name: {"frozen": True, "risk_direction": "high_fake", "output_task": "image_forgery" if name == "deep" else "news_veracity", "artifact_sha256": "b" * 64, "fit_identifiers": ["external-train"], "fit_history": collect([{"id": "external-train", "group": "external-fit"}])} for name in ("semantic", "physical", "deep")}

    def run_comparison(self, channels: list | None = None) -> dict:
        return compare(*(channels or self.channels), self.provenance, repeats=20)

    def test_same_identity_four_way_comparison_and_explicit_head_scope(self) -> None:
        report = self.run_comparison()
        self.assertEqual(report["denominator"], 32)
        self.assertEqual(len(report["methods"]), 4)
        self.assertIn("semantic_physical_deep_vs_semantic_deep", report["contrasts"])
        for rows in report["predictions"].values():
            self.assertEqual(len(rows), 32)
            self.assertTrue(all(row["split"] == "test" for row in rows))

    def test_target_labels_cannot_change_source_head_or_calibration(self) -> None:
        modified = copy.deepcopy(self.channels)
        for rows in modified:
            for row in rows:
                if row["split"] == "test":
                    row["label"] = 1 - row["label"]
        first, second = self.run_comparison(), self.run_comparison(modified)
        self.assertEqual(first["heads"], second["heads"])
        for name in first["predictions"]:
            self.assertEqual([row["fusion_risk"] for row in first["predictions"][name]], [row["fusion_risk"] for row in second["predictions"][name]])

    def test_mismatched_rows_and_forgery_labels_are_rejected(self) -> None:
        modified = copy.deepcopy(self.channels)
        modified[2][0]["label"] = 1 - modified[2][0]["label"]
        with self.assertRaises(ValueError):
            self.run_comparison(modified)
        modified[2] = modified[2][1:]
        with self.assertRaises(ValueError):
            self.run_comparison(modified)

    def test_target_domains_cannot_fit_source_comparison_head(self) -> None:
        modified = copy.deepcopy(self.channels)
        for rows in modified:
            for row in rows:
                row["domain"] = "target"
        with self.assertRaises(ValueError):
            self.run_comparison(modified)

    def test_calibration_content_overlap_is_rejected(self) -> None:
        modified = copy.deepcopy(self.channels)
        for rows in modified:
            rows[0]["content_sha256"] = rows[32]["content_sha256"] = "same-content"
        with self.assertRaises(ValueError):
            self.run_comparison(modified)

    def test_channel_training_identity_overlap_is_rejected(self) -> None:
        self.provenance["deep"]["fit_identifiers"] = [self.channels[0][-1]["id"]]
        with self.assertRaises(ValueError):
            self.run_comparison()

    def test_missing_scores_cannot_be_dropped_from_denominator(self) -> None:
        modified = copy.deepcopy(self.channels)
        modified[2][-1]["fusion_risk"] = None
        with self.assertRaises(ValueError):
            self.run_comparison(modified)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
