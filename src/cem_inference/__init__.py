"""Boundary-aware inference for classification likelihood estimators."""

from .boundary import (
    epanechnikov,
    estimate_boundary_curvature,
    sandwich_covariance,
    stabilize_boundary_curvature,
)
from .bandwidth import (
    PairBandwidthDiagnostic,
    bandwidth_warnings,
    select_pairwise_bandwidths,
)
from .distributions import (
    EllipticalParameters,
    NormalAdapter,
    SkewNormalAdapter,
    SkewNormalParameters,
    StudentTAdapter,
    StudentTParameters,
)
from .model import ClassificationLikelihoodModel
from .fmvmm_adapter import AdaptedFMVMMFit, adapt_fmvmm_fit, fit_fmvmm_hard
from .inference import InferenceResult, bootstrap_coordinates, compute_inference
from .sampling import MixtureSpecification
from .targets import CMLTargetStudy, approximate_cml_target
from .optimization import LocalOptimizationResult, optimize_local_cml

__version__ = "0.4.0"

__all__ = [
    "ClassificationLikelihoodModel",
    "AdaptedFMVMMFit",
    "CMLTargetStudy",
    "EllipticalParameters",
    "NormalAdapter",
    "InferenceResult",
    "MixtureSpecification",
    "LocalOptimizationResult",
    "SkewNormalAdapter",
    "SkewNormalParameters",
    "StudentTAdapter",
    "StudentTParameters",
    "PairBandwidthDiagnostic",
    "adapt_fmvmm_fit",
    "approximate_cml_target",
    "bandwidth_warnings",
    "bootstrap_coordinates",
    "compute_inference",
    "epanechnikov",
    "estimate_boundary_curvature",
    "fit_fmvmm_hard",
    "optimize_local_cml",
    "select_pairwise_bandwidths",
    "sandwich_covariance",
    "stabilize_boundary_curvature",
]
