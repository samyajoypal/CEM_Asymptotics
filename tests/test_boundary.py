import numpy as np
from scipy.stats import norm

from cem_inference import (
    ClassificationLikelihoodModel,
    EllipticalParameters,
    NormalAdapter,
    estimate_boundary_curvature,
)
from cem_inference.boundary import (
    epanechnikov,
    sandwich_covariance,
    stabilize_boundary_curvature,
)


def test_epanechnikov_integrates_to_one():
    grid = np.linspace(-1.5, 1.5, 100_001)
    assert np.isclose(np.trapezoid(epanechnikov(grid), grid), 1.0, atol=2e-8)
    assert np.all(epanechnikov(np.array([-2.0, 2.0])) == 0)


def test_tube_estimator_recovers_standard_normal_density_at_boundary():
    """For g(x; theta)=theta-x, B=p0(theta) because grad_theta g=1."""
    rng = np.random.default_rng(9173)
    observations = rng.normal(size=250_000)
    theta = 0.0
    bandwidth = 0.18
    estimate = np.mean(epanechnikov((theta - observations) / bandwidth)) / bandwidth
    assert np.isclose(estimate, norm.pdf(theta), atol=0.008)


def test_sandwich_uses_positive_curvature_i_minus_b():
    fixed = np.diag([3.0, 2.0])
    boundary = np.diag([0.5, 0.25])
    scores = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    covariance, positive_curvature, score_covariance = sandwich_covariance(
        fixed, boundary, scores
    )
    np.testing.assert_allclose(positive_curvature, fixed - boundary)
    inverse = np.linalg.inv(fixed - boundary)
    np.testing.assert_allclose(covariance, inverse @ score_covariance @ inverse.T)
    np.testing.assert_allclose(covariance, covariance.T)


def test_sandwich_rejects_nonpositive_curvature():
    scores = np.eye(2)
    try:
        sandwich_covariance(np.eye(2), 2 * np.eye(2), scores)
    except np.linalg.LinAlgError as error:
        assert "not positive definite" in str(error)
    else:
        raise AssertionError("Nonpositive curvature must not be silently regularized.")


def test_boundary_stabilization_enforces_relative_eigenvalue_constraint():
    fixed = np.diag([2.0, 1.0])
    boundary = np.diag([2.4, 0.2])
    stabilized, factor, raw_maximum = stabilize_boundary_curvature(
        fixed, boundary, sample_size=625
    )
    assert raw_maximum > 1
    assert 0 < factor < 1
    covariance, curvature, _ = sandwich_covariance(
        fixed, stabilized, np.eye(2)
    )
    assert np.min(np.linalg.eigvalsh(curvature)) > 0
    assert np.all(np.isfinite(covariance))


def test_model_boundary_estimator_is_symmetric_positive_semidefinite():
    rng = np.random.default_rng(88)
    observations = np.concatenate(
        [rng.normal(-1.0, 1.0, 400), rng.normal(1.0, 1.0, 400)]
    )[:, None]
    model = ClassificationLikelihoodModel([NormalAdapter(1), NormalAdapter(1)])
    coordinates = model.pack(
        np.array([0.5, 0.5]),
        [
            EllipticalParameters(np.array([-1.0]), np.array([[1.0]])),
            EllipticalParameters(np.array([1.0]), np.array([[1.0]])),
        ],
    )
    boundary = estimate_boundary_curvature(
        model, observations, coordinates, bandwidth=0.35
    )
    np.testing.assert_allclose(boundary, boundary.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(boundary)) >= -1e-10
    assert np.trace(boundary) > 0


def test_full_numerical_sandwich_pipeline_has_expected_shapes():
    observations = np.array([[-1.2], [-0.8], [-0.4], [0.4], [0.8], [1.2]])
    model = ClassificationLikelihoodModel([NormalAdapter(1), NormalAdapter(1)])
    coordinates = model.pack(
        np.array([0.5, 0.5]),
        [
            EllipticalParameters(np.array([-1.0]), np.array([[1.0]])),
            EllipticalParameters(np.array([1.0]), np.array([[1.0]])),
        ],
    )
    scores = model.classified_score_matrix(observations, coordinates)
    fixed = model.fixed_classification_information(observations, coordinates)
    boundary = estimate_boundary_curvature(
        model, observations, coordinates, bandwidth=0.8
    )
    covariance, positive_curvature, score_covariance = sandwich_covariance(
        fixed, boundary, scores
    )
    expected = (model.parameter_dimension, model.parameter_dimension)
    assert fixed.shape == boundary.shape == covariance.shape == expected
    assert positive_curvature.shape == score_covariance.shape == expected
    assert np.all(np.isfinite(covariance))
