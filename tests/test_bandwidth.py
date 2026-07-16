import numpy as np

from cem_inference import (
    ClassificationLikelihoodModel,
    EllipticalParameters,
    NormalAdapter,
    estimate_boundary_curvature,
    bandwidth_warnings,
    select_pairwise_bandwidths,
)


def test_pairwise_bandwidths_and_diagnostics_feed_estimator():
    rng = np.random.default_rng(991)
    observations = rng.normal(size=(600, 1))
    model = ClassificationLikelihoodModel([NormalAdapter(1), NormalAdapter(1)])
    coordinates = model.pack(
        np.array([0.5, 0.5]),
        [
            EllipticalParameters(np.array([-0.8]), np.array([[1.0]])),
            EllipticalParameters(np.array([0.8]), np.array([[1.0]])),
        ],
    )
    bandwidths, diagnostics = select_pairwise_bandwidths(
        model, observations, coordinates
    )
    assert set(bandwidths) == {(0, 1)}
    diagnostic = diagnostics[0]
    assert diagnostic.bandwidth > 0
    assert diagnostic.active_observations == len(observations)
    assert 0 <= diagnostic.effective_fraction <= 1
    matrix = estimate_boundary_curvature(
        model, observations, coordinates, bandwidth=bandwidths
    )
    assert np.all(np.isfinite(matrix))
    assert bandwidth_warnings(diagnostics, minimum_effective_count=1) == []
    warnings = bandwidth_warnings(diagnostics, minimum_effective_count=10_000)
    assert "kernel support" in warnings[0]
