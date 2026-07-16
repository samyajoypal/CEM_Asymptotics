import numpy as np

from cem_inference import (
    MixtureSpecification,
    NormalAdapter,
    SkewNormalAdapter,
    SkewNormalParameters,
)
from cem_inference.distributions import EllipticalParameters


def test_mixture_sampling_is_reproducible_and_has_expected_shapes():
    specification = MixtureSpecification(
        adapters=(NormalAdapter(2), NormalAdapter(2)),
        weights=np.array([0.3, 0.7]),
        component_parameters=(
            EllipticalParameters(np.zeros(2), np.eye(2)),
            EllipticalParameters(np.ones(2), np.eye(2)),
        ),
    )
    first = specification.sample(100, np.random.default_rng(4))
    second = specification.sample(100, np.random.default_rng(4))
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0].shape == (100, 2)


def test_skew_normal_sampler_matches_theoretical_mean():
    adapter = SkewNormalAdapter(2)
    parameters = SkewNormalParameters(
        np.array([0.2, -0.3]),
        np.array([[1.0, 0.2], [0.2, 1.4]]),
        np.array([2.0, -1.0]),
    )
    sample = adapter.sample(np.random.default_rng(22), 120_000, parameters)
    eigenvalues, eigenvectors = np.linalg.eigh(parameters.covariance)
    square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    delta = parameters.shape / np.sqrt(1 + parameters.shape @ parameters.shape)
    expected = parameters.location + np.sqrt(2 / np.pi) * square_root @ delta
    np.testing.assert_allclose(sample.mean(axis=0), expected, atol=0.012)
