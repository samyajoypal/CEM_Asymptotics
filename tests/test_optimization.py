import numpy as np

from cem_inference import (
    ClassificationLikelihoodModel,
    EllipticalParameters,
    NormalAdapter,
    optimize_local_cml,
)


def test_local_optimizer_improves_objective_without_leaving_trust_box():
    rng = np.random.default_rng(19)
    observations = np.concatenate(
        [rng.normal(-3.0, 0.8, 120), rng.normal(3.0, 0.8, 120)]
    )[:, None]
    model = ClassificationLikelihoodModel([NormalAdapter(1), NormalAdapter(1)])
    initial = model.pack(
        np.array([0.5, 0.5]),
        [
            EllipticalParameters(np.array([-2.8]), np.array([[1.0]])),
            EllipticalParameters(np.array([2.8]), np.array([[1.0]])),
        ],
    )
    result = optimize_local_cml(
        observations, model, initial, trust_radius=0.6, max_iter=150
    )
    assert result.success
    assert result.objective >= result.initial_objective
    assert np.max(np.abs(result.coordinates - initial)) < 0.6
