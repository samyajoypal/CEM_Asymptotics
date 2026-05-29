# simulation/scenarios.py

import numpy as np
from inference.utils import (

    covariance_to_cholesky_params,

    get_parameter_names,
)

# ============================================================
# Global dimension
# ============================================================

P = 2


# ============================================================
# Fixed covariance matrices
# ============================================================

SIGMA_NORMAL = np.array([
    [1.0, 0.30],
    [0.30, 1.0],
])

SIGMA_T = np.array([
    [1.2, 0.25],
    [0.25, 1.2],
])

SIGMA_SKEW = np.array([
    [1.0, -0.20],
    [-0.20, 1.0],
])


# ============================================================
# Scenario templates
# ============================================================

def make_normal_t_scenario(
    mu1,
    mu2,
    name,
):
    """
    Construct Normal + multivariate-t scenario.
    """

    return {

        "family": "normal_t",

        "description": (
            f"{name} Normal-multivariate-t mixture"
        ),

        "pis": [0.5, 0.5],

        # ----------------------------------------------------
        # Component 1 : multivariate normal
        # ----------------------------------------------------

        "alpha1": [

            np.array(mu1),

            SIGMA_NORMAL,
        ],

        # ----------------------------------------------------
        # Component 2 : multivariate t
        # ----------------------------------------------------

        "alpha2": [

            np.array(mu2),

            SIGMA_T,

            5.0,   # nu
        ],
    }


def make_normal_skewnormal_scenario(
    mu1,
    mu2,
    name,
):
    """
    Construct Normal + multivariate skew-normal scenario.
    """

    return {

        "family": "normal_skewnormal",

        "description": (
            f"{name} Normal-skew-normal mixture"
        ),

        "pis": [0.5, 0.5],

        # ----------------------------------------------------
        # Component 1 : multivariate normal
        # ----------------------------------------------------

        "alpha1": [

            np.array(mu1),

            SIGMA_NORMAL,
        ],

        # ----------------------------------------------------
        # Component 2 : multivariate skew-normal
        # ----------------------------------------------------

        "alpha2": [

            np.array(mu2),

            SIGMA_SKEW,

            np.array([5.0, -3.0]),
        ],
    }


# ============================================================
# Normal + multivariate-t scenarios
# ============================================================

NORMAL_T_SCENARIOS = {

    # ========================================================
    # Well-separated
    # ========================================================

    "well_separated":

        make_normal_t_scenario(
            mu1=[-4.0, -4.0],
            mu2=[4.0, 4.0],
            name="Well-separated",
        ),

    # ========================================================
    # Moderate overlap
    # ========================================================

    "moderate_overlap":

        make_normal_t_scenario(
            mu1=[-2.0, -2.0],
            mu2=[2.0, 2.0],
            name="Moderately overlapping",
        ),

    # ========================================================
    # Strong overlap
    # ========================================================

    "strong_overlap":

        make_normal_t_scenario(
            mu1=[-1.5, -1.5],
            mu2=[1.5, 1.5],
            name="Strongly overlapping",
        ),
}


# ============================================================
# Normal + skew-normal scenarios
# ============================================================

NORMAL_SKEWNORMAL_SCENARIOS = {

    # ========================================================
    # Well-separated
    # ========================================================

    "well_separated":

        make_normal_skewnormal_scenario(
            mu1=[-4.0, -4.0],
            mu2=[4.0, 4.0],
            name="Well-separated",
        ),

    # ========================================================
    # Moderate overlap
    # ========================================================

    "moderate_overlap":

        make_normal_skewnormal_scenario(
            mu1=[-2.0, -2.0],
            mu2=[2.0, 2.0],
            name="Moderately overlapping",
        ),

    # ========================================================
    # Strong overlap
    # ========================================================

    "strong_overlap":

        make_normal_skewnormal_scenario(
            mu1=[-1.5, -1.5],
            mu2=[1.5, 1.5],
            name="Strongly overlapping",
        ),
}


# ============================================================
# True unconstrained parameter vectors
# ============================================================

def get_true_unconstrained_params(
    scenario,
):
    """
    Construct true unconstrained parameter vector.

    Returns
    -------
    dict
    """

    family = scenario["family"]

    pi1 = scenario["pis"][0]

    rho = np.log(
        pi1 / (1.0 - pi1)
    )

    # ========================================================
    # Normal + t
    # ========================================================

    if family == "normal_t":

        mu1, Sigma1 = scenario["alpha1"]

        mu2, Sigma2, nu = scenario["alpha2"]

        chol1 = covariance_to_cholesky_params(
            Sigma1
        )

        chol2 = covariance_to_cholesky_params(
            Sigma2
        )

        return {

            "rho": rho,

            # ------------------------------------------------
            # Component 1
            # ------------------------------------------------

            "mu1_1": mu1[0],
            "mu1_2": mu1[1],

            "l11_1": chol1[0],
            "l21_1": chol1[1],
            "l22_1": chol1[2],

            # ------------------------------------------------
            # Component 2
            # ------------------------------------------------

            "mu2_1": mu2[0],
            "mu2_2": mu2[1],

            "l11_2": chol2[0],
            "l21_2": chol2[1],
            "l22_2": chol2[2],

            # ------------------------------------------------
            # Degrees of freedom
            # ------------------------------------------------

            "lambda": np.log(nu - 2.0),
        }

    # ========================================================
    # Normal + skew-normal
    # ========================================================

    elif family == "normal_skewnormal":

        mu1, Sigma1 = scenario["alpha1"]

        xi2, Omega2, alpha = scenario["alpha2"]

        chol1 = covariance_to_cholesky_params(
            Sigma1
        )

        chol2 = covariance_to_cholesky_params(
            Omega2
        )

        return {

            "rho": rho,

            # ------------------------------------------------
            # Component 1
            # ------------------------------------------------

            "mu1_1": mu1[0],
            "mu1_2": mu1[1],

            "l11_1": chol1[0],
            "l21_1": chol1[1],
            "l22_1": chol1[2],

            # ------------------------------------------------
            # Component 2
            # ------------------------------------------------

            "xi2_1": xi2[0],
            "xi2_2": xi2[1],

            "l11_2": chol2[0],
            "l21_2": chol2[1],
            "l22_2": chol2[2],

            # ------------------------------------------------
            # Skewness
            # ------------------------------------------------

            "alpha1": alpha[0],
            "alpha2": alpha[1],
        }

    else:

        raise ValueError(
            f"Unknown family: {family}"
        )
# ============================================================
# Parameter names
# ============================================================



# ============================================================
# Full simulation grid
# ============================================================

def get_scenario_grid(
    family_pair="all",
    sample_sizes=None,
):
    """
    Construct full simulation design grid.

    Parameters
    ----------
    family_pair : str

        "normal_t"
        "normal_skewnormal"
        "all"

    sample_sizes : list or None

    Returns
    -------
    list of dict
    """

    if sample_sizes is None:

        sample_sizes = [

            250,
            500,
            1000,
            2000,
        ]

    grid = []

    # --------------------------------------------------------
    # Select families
    # --------------------------------------------------------

    scenario_sets = []

    if family_pair == "normal_t":

        scenario_sets.append(
            ("normal_t", NORMAL_T_SCENARIOS)
        )

    elif family_pair == "normal_skewnormal":

        scenario_sets.append(
            (
                "normal_skewnormal",
                NORMAL_SKEWNORMAL_SCENARIOS,
            )
        )

    elif family_pair == "all":

        scenario_sets.extend([

            ("normal_t", NORMAL_T_SCENARIOS),

            (
                "normal_skewnormal",
                NORMAL_SKEWNORMAL_SCENARIOS,
            ),
        ])

    else:

        raise ValueError(
            f"Unknown family_pair: {family_pair}"
        )

    # --------------------------------------------------------
    # Build full grid
    # --------------------------------------------------------

    for family_name, scenarios in scenario_sets:

        for scenario_name, scenario in scenarios.items():

            for n in sample_sizes:

                grid.append({

                    "family": family_name,

                    "scenario_name": scenario_name,

                    "description":
                        scenario["description"],

                    "n": n,

                    "p": P,

                    "pis": scenario["pis"],

                    "alpha1": scenario["alpha1"],

                    "alpha2": scenario["alpha2"],

                    "true_params":
                        get_true_unconstrained_params(
                            scenario
                        ),

                    "parameter_names":
                        get_parameter_names(
                            family_name
                        ),
                })

    return grid


# ============================================================
# Pretty-print helper
# ============================================================

def print_scenario_summary(grid):
    """
    Print compact scenario summary.
    """

    print("\nSimulation Scenario Grid")
    print("=" * 70)

    for i, g in enumerate(grid):

        print(f"\nScenario {i+1}")

        print(f"Family      : {g['family']}")

        print(f"Setting     : {g['scenario_name']}")

        print(f"Sample size : {g['n']}")

        print(f"Dimension   : {g['p']}")

        print(f"pis         : {g['pis']}")

        print(f"alpha1      : {g['alpha1']}")

        print(f"alpha2      : {g['alpha2']}")

    print("\n" + "=" * 70)
