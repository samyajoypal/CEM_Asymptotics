"""Strict adapter from FMVMM 3 hard-EM fits to the paper parameterization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np

from .distributions import (
    DistributionAdapter,
    EllipticalParameters,
    NormalAdapter,
    SkewNormalAdapter,
    SkewNormalParameters,
    StudentTAdapter,
    StudentTParameters,
)
from .model import ClassificationLikelihoodModel


ADAPTER_TO_FMVMM = {
    NormalAdapter: "mvn",
    StudentTAdapter: "mvt",
    SkewNormalAdapter: "mvsn",
}

MODULE_TO_FMVMM = {
    "multivariate_norm": "mvn",
    "multivariate_t": "mvt",
    "multivariate_skewnorm": "mvsn",
}


@dataclass(frozen=True)
class AdaptedFMVMMFit:
    model: ClassificationLikelihoodModel
    coordinates: np.ndarray
    weights: np.ndarray
    component_parameters: tuple
    assignments: np.ndarray
    bic: float
    log_likelihood: float
    canonical_from_fitted: tuple[int, ...]
    raw_model: Any
    classification_objective: float = np.nan
    objective_gap: float = np.nan
    admissible_candidates: int = 1
    total_candidates: int = 1
    initialization: str = "unknown"
    selected_admissible: bool = True


def adapter_code(adapter: DistributionAdapter) -> str:
    for adapter_type, code in ADAPTER_TO_FMVMM.items():
        if isinstance(adapter, adapter_type):
            return code
    raise TypeError(f"Unsupported adapter type: {type(adapter).__name__}")


def module_code(module: Any) -> str:
    name = str(getattr(module, "__name__", "")).split(".")[-1]
    if name not in MODULE_TO_FMVMM:
        raise ValueError(f"Unsupported FMVMM distribution module: {name!r}")
    return MODULE_TO_FMVMM[name]


def _location_from_raw(code: str, parameters: Any) -> np.ndarray:
    if code in {"mvn", "mvt", "mvsn"}:
        return np.asarray(np.real_if_close(parameters[0]), dtype=float).reshape(-1)
    raise ValueError(f"No canonical location rule for {code}.")


def canonical_permutation(
    fitted_codes: list[str], desired_codes: list[str], raw_parameters: list
) -> tuple[int, ...]:
    """Match families, ordering repeated families by fitted location."""
    if sorted(fitted_codes) != sorted(desired_codes):
        raise ValueError(
            f"Fitted families {fitted_codes} do not match requested {desired_codes}."
        )
    unused = set(range(len(fitted_codes)))
    permutation: list[int] = []
    for code in desired_codes:
        candidates = [index for index in unused if fitted_codes[index] == code]
        candidates.sort(
            key=lambda index: tuple(_location_from_raw(code, raw_parameters[index]))
        )
        chosen = candidates[0]
        permutation.append(chosen)
        unused.remove(chosen)
    return tuple(permutation)


def raw_parameters_to_adapter(adapter: DistributionAdapter, raw: Any):
    values = [np.real_if_close(value) for value in raw]
    if isinstance(adapter, NormalAdapter):
        return EllipticalParameters(
            np.asarray(values[0], dtype=float), np.asarray(values[1], dtype=float)
        )
    if isinstance(adapter, StudentTAdapter):
        return StudentTParameters(
            np.asarray(values[0], dtype=float),
            np.asarray(values[1], dtype=float),
            float(np.asarray(values[2]).reshape(-1)[0]),
        )
    if isinstance(adapter, SkewNormalAdapter):
        return SkewNormalParameters(
            np.asarray(values[0], dtype=float),
            np.asarray(values[1], dtype=float),
            np.asarray(values[2], dtype=float),
        )
    raise TypeError(f"Unsupported adapter type: {type(adapter).__name__}")


def adapt_fmvmm_fit(
    raw_model: Any, adapters: list[DistributionAdapter], criterion: str = "bic"
) -> AdaptedFMVMMFit:
    """Validate and adapt an already-fitted FMVMM object."""
    if getattr(raw_model, "em_type", None) != "hard":
        raise ValueError("FMVMM model must be fitted with em_type='hard'.")
    fields = ["worked_dist", "list_pi", "list_alpha", "list_cluster", "list_bic"]
    lengths = {field: len(getattr(raw_model, field, [])) for field in fields}
    if not lengths or len(set(lengths.values())) != 1 or next(iter(lengths.values())) == 0:
        raise ValueError(f"Inconsistent or empty FMVMM fit storage: {lengths}")
    criterion_field = {"bic": "list_bic", "aic": "list_aic", "icl": "list_icl"}.get(
        criterion
    )
    if criterion_field is None:
        raise ValueError("criterion must be one of 'bic', 'aic', or 'icl'.")
    criterion_values = np.asarray(getattr(raw_model, criterion_field), dtype=float)
    if not np.any(np.isfinite(criterion_values)):
        raise ValueError("FMVMM returned no finite model-selection criterion.")
    selected = int(np.nanargmin(criterion_values))
    fitted_codes = [module_code(module) for module in raw_model.worked_dist[selected]]
    desired_codes = [adapter_code(adapter) for adapter in adapters]
    raw_alpha = list(raw_model.list_alpha[selected])
    permutation = canonical_permutation(fitted_codes, desired_codes, raw_alpha)

    weights_fitted = np.asarray(raw_model.list_pi[selected], dtype=float)
    if np.any(weights_fitted <= 0) or not np.isclose(weights_fitted.sum(), 1.0):
        raise ValueError(f"Invalid fitted mixture weights: {weights_fitted}")
    weights = weights_fitted[list(permutation)]
    component_parameters = tuple(
        raw_parameters_to_adapter(adapter, raw_alpha[fitted_index])
        for adapter, fitted_index in zip(adapters, permutation)
    )
    model = ClassificationLikelihoodModel(adapters)
    coordinates = model.pack(weights, component_parameters)

    inverse_permutation = np.empty(len(permutation), dtype=int)
    for canonical_index, fitted_index in enumerate(permutation):
        inverse_permutation[fitted_index] = canonical_index
    fitted_assignments = np.asarray(raw_model.list_cluster[selected], dtype=int)
    assignments = inverse_permutation[fitted_assignments]
    log_likelihoods = getattr(raw_model, "list_log_likelihood", [np.nan] * len(criterion_values))
    return AdaptedFMVMMFit(
        model=model,
        coordinates=coordinates,
        weights=weights,
        component_parameters=component_parameters,
        assignments=assignments,
        bic=float(raw_model.list_bic[selected]),
        log_likelihood=float(log_likelihoods[selected]),
        canonical_from_fitted=permutation,
        raw_model=raw_model,
    )


def _coordinate_is_admissible(
    fit: AdaptedFMVMMFit,
    *,
    minimum_weight: float,
    minimum_df: float,
    maximum_df: float,
    maximum_shape: float,
    minimum_scale_eigenvalue: float,
    maximum_scale_eigenvalue: float,
) -> bool:
    """Check the compact parameter restrictions used by the theory."""
    if np.min(fit.weights) < minimum_weight:
        return False
    for parameters in fit.component_parameters:
        if isinstance(parameters, StudentTParameters):
            if not minimum_df <= parameters.degrees_of_freedom <= maximum_df:
                return False
            covariance = parameters.scale
        else:
            covariance = parameters.covariance
        eigenvalues = np.linalg.eigvalsh(np.asarray(covariance, dtype=float))
        if eigenvalues.min() < minimum_scale_eigenvalue or eigenvalues.max() > maximum_scale_eigenvalue:
            return False
        if isinstance(parameters, SkewNormalParameters):
            if np.max(np.abs(parameters.shape)) > maximum_shape:
                return False
    return bool(np.all(np.isfinite(fit.coordinates)))


def _single_candidate(raw_model: Any, adapters: list[DistributionAdapter], selected: int):
    """Adapt one stored FMVMM family-assignment candidate."""
    proxy = type("CandidateProxy", (), {})()
    proxy.em_type = raw_model.em_type
    for field in ("worked_dist", "list_pi", "list_alpha", "list_cluster", "list_bic",
                  "list_aic", "list_icl", "list_log_likelihood"):
        values = getattr(raw_model, field, None)
        if values is not None:
            setattr(proxy, field, [values[selected]])
    return adapt_fmvmm_fit(proxy, adapters, criterion="bic")


def fit_fmvmm_hard(
    observations: np.ndarray,
    adapters: list[DistributionAdapter],
    *,
    criterion: str = "classification",
    max_iter: int = 200,
    tol: float = 1e-6,
    initialization: str = "multistart",
    n_jobs: int = 1,
    verbose: bool = False,
    assignment_permutations: bool = True,
    minimum_weight: float = 0.05,
    minimum_df: float = 2.25,
    maximum_df: float = 50.0,
    maximum_shape: float = 10.0,
    minimum_scale_eigenvalue: float = 0.05,
    maximum_scale_eigenvalue: float = 20.0,
) -> AdaptedFMVMMFit:
    """Fit all family assignments and starts inside the compact parameter set."""
    from fmvmm.mixtures.FMVMM import fmvmm

    codes = [adapter_code(adapter) for adapter in adapters]
    observations = np.asarray(observations, dtype=float)
    starts = ("kmeans", "gmm") if initialization == "multistart" else (initialization,)
    candidates = []
    ordered_codes = list(dict.fromkeys(permutations(codes)))
    total_candidates = 0
    for start in starts:
        raw_model = fmvmm(
            n_clusters=len(adapters), list_of_dist=codes, specific_comb=True,
            candidate_combinations=ordered_codes if assignment_permutations else [codes],
            assignment_permutations=False, initialization=start,
            max_iter=max_iter, tol=tol, verbose=verbose, debug=False,
            em_type="hard", n_jobs=n_jobs,
        )
        raw_model.fit(observations)
        total_candidates += len(raw_model.list_bic)
        for selected in range(len(raw_model.list_bic)):
            try:
                fit = _single_candidate(raw_model, adapters, selected)
                objective = fit.model.objective(observations, fit.coordinates)
                admissible = _coordinate_is_admissible(
                    fit, minimum_weight=minimum_weight, minimum_df=minimum_df,
                    maximum_df=maximum_df, maximum_shape=maximum_shape,
                    minimum_scale_eigenvalue=minimum_scale_eigenvalue,
                    maximum_scale_eigenvalue=maximum_scale_eigenvalue,
                )
                candidates.append((objective, start, fit, admissible))
            except (ValueError, TypeError, np.linalg.LinAlgError):
                continue
    if not candidates:
        raise RuntimeError("No hard-EM candidate lies in the prespecified compact parameter set.")
    if criterion not in {"classification", "bic"}:
        raise ValueError("criterion must be 'classification' or 'bic'.")
    if criterion == "classification":
        candidates.sort(key=lambda item: item[0], reverse=True)
    else:
        candidates.sort(key=lambda item: item[2].bic)
    objective, start, fit, selected_admissible = candidates[0]
    objectives = sorted((item[0] for item in candidates), reverse=True)
    gap = objectives[0] - objectives[1] if len(objectives) > 1 else np.nan
    return AdaptedFMVMMFit(
        **{field: getattr(fit, field) for field in (
            "model", "coordinates", "weights", "component_parameters", "assignments",
            "bic", "log_likelihood", "canonical_from_fitted", "raw_model")},
        classification_objective=float(objective), objective_gap=float(gap),
        admissible_candidates=sum(item[3] for item in candidates),
        total_candidates=total_candidates, initialization=start,
        selected_admissible=bool(selected_admissible),
    )
