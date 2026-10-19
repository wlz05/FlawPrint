# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from engine.copula import correlation
from engine.conformal import validate_labels, validate_partitions, fuse
from engine.storage import validate_bundle
from engine.traces import NAMES
METHODS = ("trace_logistic", "marginal_nb", "single_trace", "text_tfidf", "gaussian_novelty", "semantic_external", "trace_rbf", "image_rbf", "trace_rbf_fusion", "image_rbf_fusion")
RBF_METHODS = ("trace_rbf", "image_rbf", "trace_rbf_fusion", "image_rbf_fusion")
def needs_semantic(method: str) -> bool:
    return method == "semantic_external" or method.endswith("_fusion")
def labels(bundle: dict) -> np.ndarray:
    if any("label" not in row for row in bundle["rows"]):
        raise ValueError("Supervised controls require source labels")
    result = validate_labels(np.asarray([row["label"] for row in bundle["rows"]]))
    if len(np.unique(result)) != 2:
        raise ValueError("Supervised training requires both classes")
    return result
@dataclass
class Baseline:
    method: str
    estimator: object | None = None
    medians: np.ndarray | None = None
    column: int | None = None
    mean: np.ndarray | None = None
    precision: np.ndarray | None = None
    scales: np.ndarray | None = None
    threshold: float = 0.0
    schema: list[str] | None = None
    columns: list[int] | None = None
    fit_state: dict | None = None
    def clean(self, bundle: dict) -> np.ndarray:
        validate_bundle(bundle)
        if self.schema is not None and bundle["names"].tolist() != self.schema:
            raise ValueError("Baseline feature names and order must match training")
        values = np.where(bundle["observed"], bundle["values"], self.medians)
        return values[:, self.columns] if self.columns is not None else values
    @classmethod
    def fit(cls, method: str, train: dict, validation: dict | None = None, seed: int = 42) -> Baseline:
        if method not in METHODS:
            raise ValueError("Unknown baseline")
        validate_bundle(train)
        if validation is not None:
            validate_bundle(validation)
            validate_partitions(train["rows"], validation["rows"])
            if validation["names"].tolist() != train["names"].tolist():
                raise ValueError("Baseline validation schema must match training")
            if any(row.get("split") != "validation" for row in validation["rows"]):
                raise ValueError("Baseline selection requires validation rows")
        if any(row.get("split") != "train" for row in train["rows"]):
            raise ValueError("Supervised baselines require source training rows")
        if method == "semantic_external":
            return cls(method, schema=train["names"].tolist())
        target = labels(train)
        medians = np.asarray([np.median(train["values"][train["observed"][:, index], index]) if train["observed"][:, index].any() else 0 for index in range(train["values"].shape[1])])
        instance = cls(method, medians=medians, schema=train["names"].tolist())
        if method in RBF_METHODS:
            instance.columns = [index for index, name in enumerate(instance.schema) if not method.startswith("image_") or name in NAMES[:20]]
            if len(instance.columns) < 2:
                raise ValueError("Joint RBF controls require at least two selected traces")
        values = instance.clean(train)
        if method in RBF_METHODS:
            instance.estimator = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", cache_size=200, tol=1e-3)).fit(values, target)
            scaler, estimator = instance.estimator.steps[0][1], instance.estimator.steps[1][1]
            instance.fit_state = {"features": [instance.schema[index] for index in instance.columns], "training_identifiers": [row["id"] for row in train["rows"]], "imputation_medians": medians[instance.columns].tolist(), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "kernel": "rbf", "C": 1.0, "gamma_policy": "scale", "fitted_gamma": float(estimator._gamma), "class_weight": "balanced", "cache_megabytes": 200, "tolerance": 1e-3, "support_vectors": int(estimator.n_support_.sum()), "source_labels_used": True, "target_profile_used": False, "validation_used_for_head_selection": False, "semantic_checkpoint": "Supplied frozen logits" if needs_semantic(method) else None, "fusion": "max(raw_svc_margin, semantic_logit)" if needs_semantic(method) else None, "scope": "Same-input nonlinear control; supervised source readout with recorded kernel capacity"}
        elif method == "trace_logistic":
            instance.estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)).fit(values, target)
        elif method == "marginal_nb":
            instance.estimator = GaussianNB(var_smoothing=1e-6).fit(values, target)
        elif method == "text_tfidf":
            instance.estimator = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=30000, sublinear_tf=True), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)).fit([row["text"] for row in train["rows"]], target)
        elif method == "single_trace":
            if validation is None:
                raise ValueError("Single-trace selection requires source validation")
            validation_labels = labels(validation)
            candidates = []
            for column in range(values.shape[1]):
                estimator = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)).fit(values[:, column:column + 1], target)
                risk = estimator.decision_function(instance.clean(validation)[:, column:column + 1])
                candidates.append((roc_auc_score(validation_labels, risk), column, estimator))
            _, instance.column, instance.estimator = max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
        elif method == "gaussian_novelty":
            genuine = values[target == 0]
            if len(genuine) < 4:
                raise ValueError("Novelty control requires four genuine source samples")
            instance.mean = genuine.mean(axis=0)
            instance.scales = np.maximum(genuine.std(axis=0), 1e-6)
            instance.precision = np.linalg.inv(correlation((genuine - instance.mean) / instance.scales, 0.2))
            reference_scores = instance.raw(train)
            if validation is not None:
                validation_scores = instance.raw(validation)
                validation_labels = labels(validation)
                thresholds = np.unique(np.quantile(validation_scores, np.linspace(0, 1, 101)))
                instance.threshold = float(min(thresholds, key=lambda threshold: np.mean((validation_scores >= threshold) != validation_labels)))
            else:
                instance.threshold = float(np.quantile(reference_scores[target == 0], 0.95))
        return instance
    def raw(self, bundle: dict) -> np.ndarray:
        validate_bundle(bundle)
        if self.schema is not None and bundle["names"].tolist() != self.schema:
            raise ValueError("Baseline prediction schema differs from training")
        if self.method == "semantic_external":
            if any("semantic_logit" not in row for row in bundle["rows"]):
                raise ValueError("External semantic logits are missing")
            return np.asarray([row["semantic_logit"] for row in bundle["rows"]])
        if self.method == "text_tfidf":
            return self.estimator.decision_function([row["text"] for row in bundle["rows"]])
        values = self.clean(bundle)
        if self.method == "gaussian_novelty":
            normalized = (values - self.mean) / self.scales
            return np.einsum("ni,ij,nj->n", normalized, self.precision, normalized)
        if self.method == "single_trace":
            values = values[:, self.column:self.column + 1]
        if self.method == "marginal_nb":
            probability = self.estimator.predict_proba(values)
            return np.log(np.maximum(probability[:, 1], 1e-12)) - np.log(np.maximum(probability[:, 0], 1e-12))
        risk = self.estimator.decision_function(values)
        if self.method in RBF_METHODS and needs_semantic(self.method):
            if any("semantic_logit" not in row for row in bundle["rows"]):
                raise ValueError("Fusion controls require complete frozen semantic logits")
            risk = fuse(risk, np.asarray([row["semantic_logit"] for row in bundle["rows"]], dtype=float))
        return risk
    def predict(self, bundle: dict) -> np.ndarray:
        result = self.raw(bundle) - self.threshold
        if not np.isfinite(result).all():
            raise ValueError("Baseline predictions must be finite")
        return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
