# simulation/monte_carlo.py

import traceback
import time

import numpy as np
import pandas as pd

import json
import os
from joblib import Parallel, delayed

from fmvmm.mixtures.FMVMM import fmvmm

from fmvmm.utils.utils_mixture import (
    sample_mixture_distribution,
)
from inference.utils import (

    flatten_full_parameters,

    get_parameter_names,

    enforce_family_order,
)

# ============================================================
# Distributions
# ============================================================

from fmvmm.distributions import (
    multivariate_norm,
)

from fmvmm.distributions import (
    multivariate_t,
)

from fmvmm.distributions import (
    multivariate_skewnorm,
)

# ============================================================
# Inference utilities
# ============================================================

from inference.sandwich import (

    compute_boundary_corrected_covariance,

    compute_naive_covariance,

    covariance_to_se,
)


# ============================================================
# Distribution map
# ============================================================

DIST_MAP = {

    "normal_t": [

        multivariate_norm.rvs,

        multivariate_t.rvs,
    ],

    "normal_skewnormal": [

        multivariate_norm.rvs,

        multivariate_skewnorm.rvs,
    ],
}


# ============================================================
# Data generation
# ============================================================

def generate_data(
    scenario,
    random_state=None,
):
    """
    Generate heterogeneous mixture data.
    """

    family = scenario["family"]

    n = scenario["n"]

    pis = scenario["pis"]

    alpha1 = scenario["alpha1"]

    alpha2 = scenario["alpha2"]

    # --------------------------------------------------------
    # IMPORTANT:
    # sample_mixture_distribution expects
    # a LIST of generators/modules
    # --------------------------------------------------------

    rand_func = DIST_MAP[family]

    X, labels = sample_mixture_distribution(

        N=n,

        rand_func=rand_func,

        pis=pis,

        alphas=[alpha1, alpha2],

        mixture_type="non-identical",

        random_state=random_state,
    )

    X = np.asarray(X)

    # --------------------------------------------------------
    # Ensure matrix structure
    # --------------------------------------------------------

    if X.ndim == 1:

        X = X.reshape(-1, 1)

    return X, labels

# ============================================================
# Fit heterogeneous CEM
# ============================================================

def fit_cem_model(
    X,
    family,
):
    """
    Fit heterogeneous CEM model using fmvmm.
    """

def fit_cem_model(X, family, scenario, selection_method="oracle"):
    """
    Fits mixture model using fmvmm.
    Evaluates both permutations to handle label switching.
    selection_method: 'oracle' (distance to true parameters) or 'bic' (model selection criterion).
    """

    if family == "normal_t":
        dist_list_1 = ["mvn", "mvt"]
        dist_list_2 = ["mvt", "mvn"]
    elif family == "normal_skewnormal":
        dist_list_1 = ["mvn", "mvsn"]
        dist_list_2 = ["mvsn", "mvn"]
    else:
        raise ValueError(f"Unknown family: {family}")

    # ========================================================
    # Fit models
    # ========================================================

    model1 = fmvmm(
        n_clusters=2,
        list_of_dist=dist_list_1,
        specific_comb=True,
        initialization="kmeans",
        max_iter=200,
        tol=1e-6,
        verbose=False,
        debug=False,
    )
    model1.fit(X)

    model2 = fmvmm(
        n_clusters=2,
        list_of_dist=dist_list_2,
        specific_comb=True,
        initialization="kmeans",
        max_iter=200,
        tol=1e-6,
        verbose=False,
        debug=False,
    )
    model2.fit(X)

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    if selection_method == "bic":
        bic1 = model1.list_bic[0] if len(model1.list_bic) > 0 else np.inf
        bic2 = model2.list_bic[0] if len(model2.list_bic) > 0 else np.inf

        if bic1 <= bic2 and bic1 != np.inf:
            model = model1
        elif bic2 < bic1:
            model = model2
        else:
            raise RuntimeError("fmvmm failed to return parameters.")
            
    elif selection_method == "oracle":
        
        def compute_component_distance(alpha_hat_comp, true_alpha_comp):
            d_mu = np.linalg.norm(alpha_hat_comp[0] - true_alpha_comp[0])
            d_Sigma = np.linalg.norm(alpha_hat_comp[1] - true_alpha_comp[1], ord='fro')
            return d_mu + d_Sigma
            
        true_alpha1 = scenario["alpha1"]
        true_alpha2 = scenario["alpha2"]
        
        dist1 = np.inf
        if len(model1.list_alpha) > 0:
            alpha_hat1 = model1.list_alpha[0]
            dist1 = compute_component_distance(alpha_hat1[0], true_alpha1) + \
                    compute_component_distance(alpha_hat1[1], true_alpha2)
                    
        dist2 = np.inf
        if len(model2.list_alpha) > 0:
            alpha_hat2 = model2.list_alpha[0]
            # model2 order is swapped
            dist2 = compute_component_distance(alpha_hat2[0], true_alpha2) + \
                    compute_component_distance(alpha_hat2[1], true_alpha1)
                    
        if dist1 <= dist2 and dist1 != np.inf:
            model = model1
        elif dist2 < dist1:
            model = model2
        else:
            raise RuntimeError("fmvmm failed to return parameters.")
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")

    best_idx = 0

    pi_hat = model.list_pi[best_idx]

    alpha_hat = model.list_alpha[best_idx]

    z_hat = model.list_cluster[best_idx]
    
    pi_hat, alpha_hat, z_hat = (

        enforce_family_order(

            family=family,

            pi_hat=pi_hat,

            alpha_hat=alpha_hat,

            z_hat=z_hat,
        )
    )

    return {

        "model":
            model,

        "pi_hat":
            pi_hat,

        "alpha_hat":
            alpha_hat,

        "z_hat":
            z_hat,

        "best_idx":
            best_idx,
    }


# ============================================================
# Extract unconstrained parameters
# ============================================================




# ============================================================
# Single Monte Carlo replication
# ============================================================

def run_single_replication(
    scenario,
    replication_id,
    random_seed,
    cml_targets=None,
):
    """
    Run one Monte Carlo replication.
    """

    # ========================================================
    # Default failure object
    # ========================================================

    base_failure = {

        "family":
            scenario["family"],

        "scenario_name":
            scenario["scenario_name"],

        "n":
            scenario["n"],

        "replication":
            replication_id,
    }

    try:

        # ----------------------------------------------------
        # Generate data
        # ----------------------------------------------------

        X, labels_true = generate_data(

            scenario,

            random_state=(
                random_seed
                + replication_id
            ),
        )

        # ----------------------------------------------------
        # Fit model
        # ----------------------------------------------------

        t0 = time.time()
        fit = fit_cem_model(

            X=X,

            family=scenario["family"],
            
            scenario=scenario,
            
            selection_method="oracle",
        )
        t1 = time.time()
        compute_time = t1 - t0

        # ----------------------------------------------------
        # Flatten parameter vector
        # ----------------------------------------------------

        theta_vector = flatten_full_parameters(

            family=scenario["family"],

            pi_hat=fit["pi_hat"],

            alpha_hat=fit["alpha_hat"],
        )

        parameter_names = get_parameter_names(

            scenario["family"]
        )

        theta_hat = dict(

            zip(
                parameter_names,
                theta_vector,
            )
        )

        # ====================================================
        # Naive covariance
        # ====================================================

        try:

            naive_cov = compute_naive_covariance(

                X=X,

                family=scenario["family"],

                fit=fit,
            )

            naive_se = covariance_to_se(
                naive_cov
            )

        except Exception as e:

            raise RuntimeError(
                f"Naive covariance failed: {e}"
            )

        # ====================================================
        # Boundary-corrected covariance
        # ====================================================

        try:

            bc_cov = (

                compute_boundary_corrected_covariance(

                    X=X,

                    family=scenario["family"],

                    fit=fit,

                    scenario=scenario,
                )
            )

            bc_se = covariance_to_se(
                bc_cov
            )

        except Exception as e:

            raise RuntimeError(
                f"Boundary-corrected covariance failed: {e}"
            )

        # ====================================================
        # Store parameter rows
        # ====================================================

        rows = []

        true_params = scenario["true_params"]

        for idx, param in enumerate(
            parameter_names
        ):

            theta_true = true_params[param]

            theta_est = theta_hat[param]
            
            cml_target = np.nan
            if cml_targets is not None:
                family = scenario["family"]
                scenario_name = scenario["scenario_name"]
                if family in cml_targets and scenario_name in cml_targets[family]:
                    cml_target = cml_targets[family][scenario_name].get(param, np.nan)

            row = {

                **base_failure,

                "status":
                    "success",

                "parameter":
                    param,

                f"theta_{param}":
                    theta_est,

                f"true_{param}":
                    theta_true,
                    
                f"cml_target_{param}":
                    cml_target,

                f"naive_se_{param}":
                    naive_se[idx],

                f"bc_se_{param}":
                    bc_se[idx],

                "compute_time":
                    compute_time,
            }

            rows.append(row)

        return rows

    # ========================================================
    # Failure handling
    # ========================================================

    except Exception as e:

        traceback.print_exc()

        return [{

            **base_failure,

            "status":
                "failed",

            "parameter":
                "FAILURE",

            "error":
                str(e),

            "compute_time":
                np.nan,
        }]


# ============================================================
# Monte Carlo driver
# ============================================================

def run_monte_carlo(
    scenario_grid,
    n_replications=100,
    random_seed=12345,
    n_jobs=-1,
):
    """
    Run complete Monte Carlo simulation study.
    """
    
    # Try loading CML targets
    cml_targets = None
    target_path = os.path.join(os.path.dirname(__file__), "cml_targets.json")
    if os.path.exists(target_path):
        with open(target_path, "r") as f:
            cml_targets = json.load(f)

    tasks = []

    # --------------------------------------------------------
    # Build tasks
    # --------------------------------------------------------

    for scenario in scenario_grid:

        for r in range(n_replications):

            tasks.append(

                delayed(
                    run_single_replication
                )(
                    scenario=scenario,

                    replication_id=r,

                    random_seed=random_seed,
                    
                    cml_targets=cml_targets,
                )
            )

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    results = Parallel(

        n_jobs=n_jobs,

        verbose=10,

        backend="loky",
    )(tasks)

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    flat_rows = []

    for block in results:

        flat_rows.extend(block)

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        flat_rows
    )

    return results_df
