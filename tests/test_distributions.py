import numpy as np

from cem_inference.distributions import (
    EllipticalParameters,
    NormalAdapter,
    SkewNormalAdapter,
    SkewNormalParameters,
    StudentTAdapter,
    StudentTParameters,
)


def test_distribution_round_trips():
    covariance = np.array([[1.3, 0.25], [0.25, 0.8]])
    cases = [
        (NormalAdapter(2), EllipticalParameters(np.array([1.0, -2.0]), covariance)),
        (
            StudentTAdapter(2),
            StudentTParameters(np.array([1.0, -2.0]), covariance, 7.5),
        ),
        (
            SkewNormalAdapter(2),
            SkewNormalParameters(
                np.array([1.0, -2.0]), covariance, np.array([2.5, -1.2])
            ),
        ),
    ]
    for adapter, parameters in cases:
        reconstructed = adapter.unpack(adapter.pack(parameters))
        np.testing.assert_allclose(reconstructed.location, parameters.location)
        matrix_name = "scale" if isinstance(parameters, StudentTParameters) else "covariance"
        np.testing.assert_allclose(
            getattr(reconstructed, matrix_name), getattr(parameters, matrix_name)
        )
        if isinstance(parameters, StudentTParameters):
            assert np.isclose(
                reconstructed.degrees_of_freedom, parameters.degrees_of_freedom
            )
        if isinstance(parameters, SkewNormalParameters):
            np.testing.assert_allclose(reconstructed.shape, parameters.shape)


def test_skew_normal_matches_fmvmm_exactly():
    from fmvmm.distributions import multivariate_skewnorm

    rng = np.random.default_rng(20260710)
    observations = rng.normal(size=(30, 2))
    location = np.array([0.3, -0.7])
    covariance = np.array([[1.4, -0.2], [-0.2, 0.9]])
    shape = np.array([3.0, -1.5])
    adapter = SkewNormalAdapter(2)
    ours = adapter.logpdf(
        observations, SkewNormalParameters(location, covariance, shape)
    )
    reference = multivariate_skewnorm.logpdf(
        observations, location, covariance, shape
    )
    np.testing.assert_allclose(ours, reference, rtol=2e-12, atol=2e-12)


def test_skew_normal_is_not_logistic_approximation():
    from scipy.special import expit
    from scipy.stats import multivariate_normal

    observations = np.array([[2.0, -1.0], [-2.0, 1.0]])
    location = np.zeros(2)
    covariance = np.eye(2)
    shape = np.array([4.0, -3.0])
    exact = SkewNormalAdapter(2).logpdf(
        observations, SkewNormalParameters(location, covariance, shape)
    )
    logistic = (
        np.log(2)
        + multivariate_normal.logpdf(observations, mean=location, cov=covariance)
        + np.log(expit((observations - location) @ shape))
    )
    assert np.max(np.abs(exact - logistic)) > 0.1
