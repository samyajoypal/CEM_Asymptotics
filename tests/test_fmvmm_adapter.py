from types import SimpleNamespace

import numpy as np

from cem_inference import NormalAdapter, StudentTAdapter
from cem_inference.fmvmm_adapter import adapt_fmvmm_fit, canonical_permutation
from fmvmm.distributions import multivariate_norm, multivariate_t


def test_canonical_permutation_uses_family_then_fitted_location():
    assert canonical_permutation(
        ["mvt", "mvn"], ["mvn", "mvt"],
        [[np.array([2.0])], [np.array([-1.0])]],
    ) == (1, 0)
    assert canonical_permutation(
        ["mvn", "mvn"], ["mvn", "mvn"],
        [[np.array([3.0])], [np.array([-2.0])]],
    ) == (1, 0)


def test_strict_adapter_reorders_without_oracle_parameters():
    raw = SimpleNamespace(
        em_type="hard",
        worked_dist=[(multivariate_t, multivariate_norm)],
        list_pi=[np.array([0.6, 0.4])],
        list_alpha=[[
            [np.array([1.0]), np.array([[1.2]]), 6.0],
            [np.array([-1.0]), np.array([[0.8]])],
        ]],
        list_cluster=[np.array([1, 1, 0, 0, 0])],
        list_bic=[123.0],
        list_aic=[118.0],
        list_icl=[125.0],
        list_log_likelihood=[-50.0],
    )
    result = adapt_fmvmm_fit(raw, [NormalAdapter(1), StudentTAdapter(1)])
    assert result.canonical_from_fitted == (1, 0)
    np.testing.assert_allclose(result.weights, [0.4, 0.6])
    np.testing.assert_array_equal(result.assignments, [0, 0, 1, 1, 1])
    assert np.isclose(result.component_parameters[0].location[0], -1.0)
    assert np.isclose(result.component_parameters[1].location[0], 1.0)


def test_adapter_rejects_soft_em():
    raw = SimpleNamespace(em_type="soft")
    try:
        adapt_fmvmm_fit(raw, [NormalAdapter(1), StudentTAdapter(1)])
    except ValueError as error:
        assert "em_type='hard'" in str(error)
    else:
        raise AssertionError("Soft EM must be rejected.")
