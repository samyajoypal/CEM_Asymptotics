import numpy as np

from cem_inference.numerical import gradient, hessian


def test_central_gradient_and_hessian_on_quadratic():
    matrix = np.array([[3.0, -0.4], [-0.4, 1.5]])
    linear = np.array([0.7, -1.2])
    point = np.array([0.3, -0.8])

    def function(value):
        return 0.5 * value @ matrix @ value + linear @ value

    np.testing.assert_allclose(
        gradient(function, point), matrix @ point + linear, atol=2e-10
    )
    np.testing.assert_allclose(hessian(function, point), matrix, atol=2e-8)
