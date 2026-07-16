import numpy as np

from cem_inference import (
    ClassificationLikelihoodModel,
    EllipticalParameters,
    NormalAdapter,
    SkewNormalAdapter,
    SkewNormalParameters,
)


def make_model_and_coordinates():
    model = ClassificationLikelihoodModel([NormalAdapter(2), SkewNormalAdapter(2)])
    parameters = [
        EllipticalParameters(
            np.array([-1.0, -0.5]), np.array([[1.0, 0.2], [0.2, 0.8]])
        ),
        SkewNormalParameters(
            np.array([1.0, 0.5]),
            np.array([[0.9, -0.1], [-0.1, 1.2]]),
            np.array([2.0, -1.0]),
        ),
    ]
    coordinates = model.pack(np.array([0.4, 0.6]), parameters)
    return model, coordinates


def test_model_pack_unpack_and_score_shapes():
    model, coordinates = make_model_and_coordinates()
    unpacked = model.unpack(coordinates)
    np.testing.assert_allclose(unpacked.weights, [0.4, 0.6])
    observations = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    assert model.component_scores(observations, coordinates).shape == (3, 2)
    assert model.classified_score_matrix(observations, coordinates).shape == (
        3,
        model.parameter_dimension,
    )


def test_pairwise_gradient_is_antisymmetric():
    model, coordinates = make_model_and_coordinates()
    observations = np.array([[0.2, -0.1], [1.2, 0.3]])
    forward = model.pairwise_contrast_gradients(
        observations, coordinates, 0, 1
    )
    backward = model.pairwise_contrast_gradients(
        observations, coordinates, 1, 0
    )
    np.testing.assert_allclose(forward, -backward, atol=2e-9, rtol=2e-9)


def test_vectorized_derivatives_match_reference_implementation():
    model, coordinates = make_model_and_coordinates()
    observations = np.array([[0.2, -0.1], [1.2, 0.3], [-0.7, 0.4]])
    fast_contrast = model.pairwise_contrast_gradients(
        observations, coordinates, 0, 1
    )
    reference_contrast = model.pairwise_contrast_gradients_reference(
        observations, coordinates, 0, 1
    )
    np.testing.assert_allclose(fast_contrast, reference_contrast, atol=2e-9)
    np.testing.assert_allclose(
        model.classified_score_matrix(observations, coordinates),
        model.classified_score_matrix_reference(observations, coordinates),
        atol=2e-9,
    )


def test_objective_is_maximum_component_score_average():
    model, coordinates = make_model_and_coordinates()
    observations = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    scores = model.component_scores(observations, coordinates)
    assert np.isclose(model.objective(observations, coordinates), np.max(scores, axis=1).mean())


def test_block_information_matches_full_coordinate_reference():
    model, coordinates = make_model_and_coordinates()
    observations = np.array([
        [-1.2, -0.9], [-0.8, -0.5], [-0.5, -0.2],
        [0.6, 0.3], [1.0, 0.8], [1.4, 0.4],
    ])
    block = model.fixed_classification_information(
        observations, coordinates, step=3e-4
    )
    reference = model.fixed_classification_information_reference(
        observations, coordinates, step=3e-4
    )
    np.testing.assert_allclose(block, reference, atol=2e-6, rtol=2e-5)
