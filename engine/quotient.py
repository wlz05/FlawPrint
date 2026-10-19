# -*- coding: utf-8 -*-
# Attenuation Quotient: FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
from dataclasses import dataclass
from numbers import Real
import numpy as np
from scipy.optimize import nnls
def numeric(values: object, dimensions: int) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError("Values must form a rectangular numeric array") from error
    if raw.ndim != dimensions or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in raw.flat):
        raise ValueError("Values must be numeric with the requested dimensions")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Values must be finite")
    return result
def scalar(value: object, minimum: float, maximum: float) -> float:
    result = numeric([value], 1)[0]
    if not minimum <= result <= maximum:
        raise ValueError("Scalar is outside the supported interval")
    return float(result)
def pearson(values: object) -> np.ndarray:
    samples = numeric(values, 2)
    if samples.shape[0] < 4 or samples.shape[1] < 2:
        raise ValueError("Pearson estimation needs at least four rows and two columns")
    centered = samples - samples.mean(axis=0)
    scales = np.max(np.abs(centered), axis=0)
    if np.any(scales == 0) or not np.isfinite(centered).all():
        raise ValueError("Pearson estimation requires nonconstant finite columns")
    centered = centered / scales
    lengths = np.sqrt(np.sum(centered ** 2, axis=0))
    normalized = centered / lengths
    result = normalized.T @ normalized
    np.fill_diagonal(result, 1)
    return correlations(result)
def correlations(values: object) -> np.ndarray:
    result = numeric(values, 2)
    if result.shape[0] < 2 or result.shape[0] != result.shape[1]:
        raise ValueError("Correlation matrix must be square with at least two columns")
    if not np.allclose(result, result.T, rtol=0, atol=1e-10) or not np.allclose(np.diag(result), 1, rtol=0, atol=1e-10) or np.any(np.abs(result) > 1 + 1e-10):
        raise ValueError("Correlation matrix must be symmetric with a unit diagonal")
    if np.linalg.eigvalsh(result).min() < -1e-8:
        raise ValueError("Correlation matrix must be positive semidefinite")
    return result.copy()
def edge_indices(values: object, dimension: int) -> np.ndarray:
    try:
        rows = list(values)
    except TypeError as error:
        raise ValueError("Edges must be a list of index pairs") from error
    edges = []
    for row in rows:
        if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 2 or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in row):
            raise ValueError("Each edge needs two integer indices")
        left, right = map(int, row)
        if not 0 <= left < right < dimension:
            raise ValueError("Edge indices must be ordered, distinct, and in range")
        edges.append((left, right))
    if len(edges) != len(set(edges)):
        raise ValueError("Frozen edges cannot repeat")
    if edges != sorted(edges):
        raise ValueError("Frozen edges must use lexicographic order for error bounds")
    return np.asarray(edges, dtype=int).reshape(-1, 2)
@dataclass(frozen=True)
class Panel:
    reference: np.ndarray
    edges: np.ndarray
    incidence: np.ndarray
    projector: np.ndarray
    rank: int
    minimum_edge: float
    @classmethod
    def fit(cls, reference: object, edges: object | None = None, minimum_edge: float = 0.03) -> Panel:
        matrix = correlations(reference)
        floor = scalar(minimum_edge, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        dimension = len(matrix)
        selected = edge_indices([(left, right) for left in range(dimension) for right in range(left + 1, dimension) if abs(matrix[left, right]) > floor] if edges is None else edges, dimension)
        if len(selected) and np.any(np.abs(matrix[selected[:, 0], selected[:, 1]]) <= floor):
            raise ValueError("Every frozen reference edge must exceed the declared floor")
        incidence = np.zeros((len(selected), dimension))
        for index, (left, right) in enumerate(selected):
            incidence[index, [left, right]] = 1
        if len(selected):
            left_vectors, singular, _ = np.linalg.svd(incidence, full_matrices=True)
            tolerance = np.finfo(float).eps * max(incidence.shape) * singular[0]
            rank = int(np.sum(singular > tolerance))
            null = left_vectors[:, rank:]
            projector = null @ null.T
        else:
            rank, projector = 0, np.empty((0, 0))
        for value in (matrix, selected, incidence, projector):
            value.setflags(write=False)
        return cls(matrix, selected, incidence, projector, rank, floor)
    def document(self) -> dict:
        return {"version": "flawprint-quotient-v1", "reference": self.reference.tolist(), "edges": self.edges.tolist(), "minimum_edge": self.minimum_edge, "rank": self.rank, "invariant_dimensions": len(self.edges) - self.rank}
    @classmethod
    def load(cls, document: dict) -> Panel:
        expected = {"version", "reference", "edges", "minimum_edge", "rank", "invariant_dimensions"}
        if not isinstance(document, dict) or set(document) != expected or document["version"] != "flawprint-quotient-v1":
            raise ValueError("Unsupported quotient panel state")
        result = cls.fit(document["reference"], document["edges"], document["minimum_edge"])
        for key, value in (("rank", result.rank), ("invariant_dimensions", len(result.edges) - result.rank)):
            if isinstance(document[key], bool) or not isinstance(document[key], int) or document[key] != value:
                raise ValueError("Stored panel dimensions disagree with the frozen graph")
        return result
    def compare(self, query: object, reference_error: object | None = None, query_error: object | None = None) -> dict:
        matrix = correlations(query)
        if matrix.shape != self.reference.shape:
            raise ValueError("Query and reference dimensions must agree")
        size = len(self.edges)
        bounds = uncertainty(reference_error, query_error, size)
        result = {"panel_edges": size, "unmonitored_pairs": len(matrix) * (len(matrix) - 1) // 2 - size, "invariant_dimensions": size - self.rank, "gains_identifiable": self.rank == len(matrix), "status": "unresolved", "quotient_distance": None, "attenuation_distance": None, "attenuation_rejected": None, "error_radius": None, "sign_conflicts": [], "edges": [], "scope": "Frozen Pearson panel attenuation compatibility; not full-matrix compatibility or news veracity"}
        if not size:
            result["reason"] = "No reference edges exceed the frozen floor"
            return result
        left, right = self.edges.T
        reference_edges, query_edges = self.reference[left, right], matrix[left, right]
        weak = np.flatnonzero(np.abs(query_edges) <= self.minimum_edge)
        if len(weak):
            result.update(reason="A frozen query edge is weak; the entire panel abstains", weak_edges=self.edges[weak].tolist())
            return result
        conflicts = np.flatnonzero(np.sign(reference_edges) != np.sign(query_edges))
        delta = np.log(np.abs(query_edges)) - np.log(np.abs(reference_edges))
        projected = self.projector @ delta
        gains = np.linalg.lstsq(self.incidence, delta, rcond=None)[0]
        # Constrain gains to attenuation rather than amplification.
        loss, cone_distance = nnls(self.incidence, -delta, maxiter=max(100, 10 * len(matrix)))
        cone_residual = delta + self.incidence @ loss
        quotient_distance = float(np.linalg.norm(projected))
        numeric_tolerance = 1e-9 * max(1.0, float(np.linalg.norm(delta)))
        result.update(status="geometry_only", quotient_distance=quotient_distance, attenuation_distance=float(cone_distance), sign_conflicts=self.edges[conflicts].tolist(), unconstrained_log_gains=gains.tolist(), fitted_log_attenuation=(-loss).tolist(), fitted_coefficients=np.exp(-loss).tolist(), exact_geometry_compatible=bool(not len(conflicts) and cone_distance <= numeric_tolerance), numerical_tolerance=numeric_tolerance)
        for index, (i, j) in enumerate(self.edges):
            result["edges"].append({"left": int(i), "right": int(j), "reference": float(reference_edges[index]), "query": float(query_edges[index]), "log_change": float(delta[index]), "quotient_residual": float(projected[index]), "attenuation_residual": float(cone_residual[index]), "sign_conflict": bool(index in conflicts)})
        if bounds is None:
            result["reason"] = "Sampling error is unquantified; geometry does not certify compatibility"
        else:
            radius = float(np.linalg.norm(-np.log1p(-bounds[0]) - np.log1p(-bounds[1])))
            lower = max(0.0, float(cone_distance) - radius)
            rejected = bool(len(conflicts) or lower > numeric_tolerance)
            result.update(status="attenuation_rejected" if rejected else "not_rejected", reason="Conditional on simultaneous declared relative correlation error bounds", error_radius=radius, quotient_interval=[max(0.0, quotient_distance - radius), quotient_distance + radius], attenuation_interval=[lower, float(cone_distance) + radius], attenuation_rejected=rejected)
        if size == self.rank:
            result["magnitude_information"] = "No quotient magnitude degrees of freedom; only signs and attenuation cone remain"
        return result
def uncertainty(reference: object | None, query: object | None, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    if reference is None and query is None:
        return None
    if reference is None or query is None:
        raise ValueError("Both reference and query error bounds are required")
    results = []
    for value in (reference, query):
        if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
            item = scalar(value, 0, np.nextafter(1.0, 0.0))
            vector = np.full(size, item)
        else:
            vector = numeric(value, 1)
            if vector.shape != (size,) or np.any((vector < 0) | (vector >= 1)):
                raise ValueError("Error bounds must match all frozen edges and lie in [0, 1)")
        results.append(vector)
    return results[0], results[1]
def separation(panel: Panel, alternative: object, query: object, reference_error: object | None = None, alternative_error: object | None = None, query_error: object | None = None) -> dict:
    alternate = panel.compare(alternative, reference_error, alternative_error)
    observed = panel.compare(query, reference_error, query_error)
    result = {"status": "unresolved", "prototype_choice": None, "prototype_gap": alternate["quotient_distance"], "reference_distance": observed["quotient_distance"], "alternative_distance": None, "margin_sufficient": None, "scope": "Continuous same-sign quotient prototype diagnostic; neither task labels nor classification performance"}
    if alternate["quotient_distance"] is None or observed["quotient_distance"] is None:
        result["reason"] = "A prototype or query cannot measure the complete frozen panel"
        return result
    if alternate["sign_conflicts"] or observed["sign_conflicts"]:
        result["reason"] = "Signs differ; the continuous same-sign margin theorem does not apply"
        return result
    alternate_delta = np.asarray([row["log_change"] for row in alternate["edges"]])
    observed_delta = np.asarray([row["log_change"] for row in observed["edges"]])
    distance = float(np.linalg.norm(panel.projector @ (observed_delta - alternate_delta)))
    tolerance = max(alternate["numerical_tolerance"], observed["numerical_tolerance"])
    choice = None if abs(distance - observed["quotient_distance"]) <= tolerance else "reference" if observed["quotient_distance"] < distance else "alternative"
    result.update(status="geometry_only", prototype_choice=choice, alternative_distance=distance, reason="No bound or task interpretation is inferred from a prototype choice")
    if reference_error is not None:
        reference_bounds, alternative_bounds = uncertainty(reference_error, alternative_error, len(panel.edges))
        _, query_bounds = uncertainty(reference_error, query_error, len(panel.edges))
        radii = [float(np.linalg.norm(-np.log1p(-bound))) for bound in (reference_bounds, alternative_bounds, query_bounds)]
        measured_requirement = 2 * (max(radii[0], radii[1]) + radii[2])
        result.update(status="conditionally_bounded", margin_sufficient=bool(alternate["quotient_distance"] > measured_requirement + tolerance), measured_margin_requirement=measured_requirement, population_gap_lower=max(0.0, alternate["quotient_distance"] - radii[0] - radii[1]), error_radii=radii, reason="Conditional nearest-prototype guarantee if each query is a node attenuation of one same-sign population prototype and all error bounds hold")
    return result
def measurement(document: dict, columns: list[str], unit: str, query: bool = False) -> tuple[np.ndarray, dict]:
    allowed = {"matrix", "samples", "relative_error", "uncertainty_source", "columns", "observation_unit"} | ({"id"} if query else set())
    if not isinstance(document, dict) or set(document) - allowed or ("matrix" in document) == ("samples" in document):
        raise ValueError("A measurement needs exactly one matrix or sample array and known fields")
    if document.get("columns", columns) != columns or document.get("observation_unit", unit) != unit:
        raise ValueError("Measurements must use the same column order and observation unit")
    source = document.get("uncertainty_source")
    if "relative_error" in document and (not isinstance(source, str) or not source.strip()):
        raise ValueError("Declared error bounds require an uncertainty source")
    if "relative_error" not in document and source is not None:
        raise ValueError("An uncertainty source requires explicit error bounds")
    result = correlations(document["matrix"]) if "matrix" in document else pearson(document["samples"])
    if len(result) != len(columns):
        raise ValueError("Matrix dimensions must match the declared columns")
    return result, {"input_kind": "matrix" if "matrix" in document else "samples", "samples": None if "matrix" in document else len(document["samples"]), "uncertainty_source": source}
def audit(document: dict) -> dict:
    required = {"representation", "observation_unit", "columns", "reference", "queries"}
    if not isinstance(document, dict) or not required <= set(document) or set(document) - required - {"edges", "minimum_edge", "alternative"}:
        raise ValueError("Quotient audit requires the explicit measurement contract")
    if document["representation"] != "pearson":
        raise ValueError("The exact attenuation model requires raw Pearson correlations")
    columns, unit = document["columns"], document["observation_unit"]
    if not isinstance(columns, list) or len(columns) < 2 or any(not isinstance(name, str) or not name.strip() for name in columns) or len(set(columns)) != len(columns):
        raise ValueError("Column names must be nonempty, unique, and ordered")
    if not isinstance(unit, str) or not unit.strip() or not isinstance(document["queries"], list) or not document["queries"]:
        raise ValueError("A declared observation unit and nonempty query list are required")
    reference, reference_metadata = measurement(document["reference"], columns, unit)
    panel = Panel.fit(reference, document.get("edges"), document.get("minimum_edge", 0.03))
    alternative, alternative_metadata = measurement(document["alternative"], columns, unit) if "alternative" in document else (None, None)
    results, identifiers = [], set()
    for query in document["queries"]:
        if not isinstance(query, dict) or not isinstance(query.get("id"), str) or not query["id"].strip() or query["id"] in identifiers:
            raise ValueError("Queries require exact unique nonempty string identifiers")
        identifiers.add(query["id"])
        matrix, metadata = measurement(query, columns, unit, True)
        result = panel.compare(matrix, document["reference"].get("relative_error"), query.get("relative_error"))
        if alternative is not None:
            result["separation"] = separation(panel, alternative, matrix, document["reference"].get("relative_error"), document["alternative"].get("relative_error"), query.get("relative_error"))
        result.update(id=query["id"], measurement=metadata)
        results.append(result)
    return {"kind": "attenuation_quotient", "representation": "pearson", "observation_unit": unit, "columns": columns, "reference_measurement": reference_metadata, "alternative_measurement": alternative_metadata, "panel": panel.document(), "queries": results, "coverage": {"input": len(results), "measurable": sum(row["status"] != "unresolved" for row in results), "conditionally_bounded": sum(row["attenuation_rejected"] is not None for row in results)}, "scope": "Separate mechanism extension; error bounds are declared, not estimated; no class labels, p-values, or deployment gating"}
