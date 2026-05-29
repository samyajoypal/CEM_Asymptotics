# inference/utils.py

import numpy as np


# ============================================================
# Basic transforms
# ============================================================

def sigmoid(x):
    """
    Logistic transform.
    """

    return 1.0 / (1.0 + np.exp(-x))


def logit(
    p,
    eps=1e-12,
):
    """
    Stable logit transform.
    """

    p = np.clip(
        p,
        eps,
        1.0 - eps,
    )

    return np.log(
        p / (1.0 - p)
    )


# ============================================================
# PSD utilities
# ============================================================

def project_to_psd(
    A,
    eps=1e-8,
):
    """
    Project symmetric matrix onto PSD cone.
    """

    A = 0.5 * (A + A.T)

    eigvals, eigvecs = np.linalg.eigh(A)

    eigvals = np.maximum(
        eigvals,
        eps,
    )

    A_psd = (

        eigvecs
        @
        np.diag(eigvals)
        @
        eigvecs.T
    )

    return 0.5 * (
        A_psd + A_psd.T
    )


def safe_inverse(
    A,
    ridge=1e-8,
):
    """
    Numerically stable inverse.
    """

    A = project_to_psd(
        A,
        eps=ridge,
    )

    try:

        return np.linalg.inv(A)

    except np.linalg.LinAlgError:

        return np.linalg.pinv(A)

# ============================================================
# Cholesky parameterization
# ============================================================

def covariance_to_cholesky_params(
    Sigma,
):
    """
    Convert covariance matrix to unconstrained
    Cholesky parameterization.

    For:

        Sigma = L L^T

    where

        L = [[exp(l11), 0],
             [l21,      exp(l22)]]

    Returns
    -------
    ndarray shape (3,)
    """

    Sigma = np.asarray(Sigma)

    L = np.linalg.cholesky(Sigma)

    l11 = np.log(L[0, 0])

    l21 = L[1, 0]

    l22 = np.log(L[1, 1])

    return np.array([

        l11,

        l21,

        l22,
    ])


def cholesky_params_to_covariance(
    params,
):
    """
    Convert unconstrained Cholesky parameters
    back to covariance matrix.

    Parameters
    ----------
    params : array-like length 3

    Returns
    -------
    2x2 covariance matrix
    """

    l11, l21, l22 = params

    L = np.array([

        [np.exp(l11), 0.0],

        [l21, np.exp(l22)],
    ])

    Sigma = L @ L.T

    return Sigma

# ============================================================
# Full parameter flattening
# ============================================================

def flatten_full_parameters(
    family,
    pi_hat,
    alpha_hat,
):
    """
    Flatten full model parameters into a single vector.

    Returns
    -------
    ndarray
    """

    rho = logit(pi_hat[0])

    # ========================================================
    # Normal-t
    # ========================================================

    if family == "normal_t":

        mu1, Sigma1 = alpha_hat[0]

        mu2, Sigma2, nu = alpha_hat[1]

        chol1 = covariance_to_cholesky_params(
            Sigma1
        )

        chol2 = covariance_to_cholesky_params(
            Sigma2
        )

        nu = float(np.ravel(nu)[0])

        theta = np.concatenate([

            [rho],

            mu1,

            chol1,

            mu2,

            chol2,

            [np.log(nu - 2.0)],
        ])

    # ========================================================
    # Normal-skewnormal
    # ========================================================

    elif family == "normal_skewnormal":

        mu1, Sigma1 = alpha_hat[0]

        xi2, Omega2, alpha_vec = alpha_hat[1]

        chol1 = covariance_to_cholesky_params(
            Sigma1
        )

        chol2 = covariance_to_cholesky_params(
            Omega2
        )

        theta = np.concatenate([

            [rho],

            mu1,

            chol1,

            xi2,

            chol2,

            alpha_vec,
        ])

    else:

        raise ValueError(
            f"Unknown family: {family}"
        )

    return theta

# ============================================================
# Reconstruct parameters
# ============================================================

def reconstruct_full_parameters(
    theta,
    family,
):
    """
    Reconstruct model parameters from vectorized form.

    Returns
    -------
    pi_hat
    alpha_hat
    """

    theta = np.asarray(theta)

    # ========================================================
    # Normal-t
    # ========================================================

    if family == "normal_t":

        rho = theta[0]

        pi1 = sigmoid(rho)

        pi_hat = np.array([
            pi1,
            1.0 - pi1,
        ])

        mu1 = theta[1:3]

        chol1 = theta[3:6]

        mu2 = theta[6:8]

        chol2 = theta[8:11]

        lambda_ = theta[11]

        Sigma1 = cholesky_params_to_covariance(
            chol1
        )

        Sigma2 = cholesky_params_to_covariance(
            chol2
        )

        nu = np.exp(lambda_) + 2.0

        alpha_hat = [

            (
                mu1,
                Sigma1,
            ),

            (
                mu2,
                Sigma2,
                nu,
            ),
        ]

    # ========================================================
    # Normal-skewnormal
    # ========================================================

    elif family == "normal_skewnormal":

        rho = theta[0]

        pi1 = sigmoid(rho)

        pi_hat = np.array([
            pi1,
            1.0 - pi1,
        ])

        mu1 = theta[1:3]

        chol1 = theta[3:6]

        xi2 = theta[6:8]

        chol2 = theta[8:11]

        alpha_vec = theta[11:13]

        Sigma1 = cholesky_params_to_covariance(
            chol1
        )

        Omega2 = cholesky_params_to_covariance(
            chol2
        )

        alpha_hat = [

            (
                mu1,
                Sigma1,
            ),

            (
                xi2,
                Omega2,
                alpha_vec,
            ),
        ]

    else:

        raise ValueError(
            f"Unknown family: {family}"
        )

    return pi_hat, alpha_hat

# ============================================================
# Parameter names
# ============================================================

def get_parameter_names(
    family,
):
    """
    Parameter names for summaries/tables.
    """

    if family == "normal_t":

        return [

            "rho",

            "mu1_1",
            "mu1_2",

            "l11_1",
            "l21_1",
            "l22_1",

            "mu2_1",
            "mu2_2",

            "l11_2",
            "l21_2",
            "l22_2",

            "lambda",
        ]

    elif family == "normal_skewnormal":

        return [

            "rho",

            "mu1_1",
            "mu1_2",

            "l11_1",
            "l21_1",
            "l22_1",

            "xi2_1",
            "xi2_2",

            "l11_2",
            "l21_2",
            "l22_2",

            "alpha1",
            "alpha2",
        ]

    else:

        raise ValueError(
            f"Unknown family: {family}"
        )

# ============================================================
# Confidence intervals
# ============================================================

def compute_confidence_intervals(
    theta_hat,
    se_hat,
    z_value=1.96,
):
    """
    Wald confidence intervals.
    """

    theta_hat = np.asarray(theta_hat)

    se_hat = np.asarray(se_hat)

    lower = (

        theta_hat
        -
        z_value * se_hat
    )

    upper = (

        theta_hat
        +
        z_value * se_hat
    )

    return lower, upper


def coverage_indicator(
    theta_true,
    lower,
    upper,
):
    """
    Coverage indicator.
    """

    theta_true = np.asarray(
        theta_true
    )

    return (

        (theta_true >= lower)
        &
        (theta_true <= upper)

    ).astype(int)


# ============================================================
# Label switching utilities
# ============================================================

def align_labels_by_mean_distance(
    family,
    alpha_hat,
    true_alpha1,
    true_alpha2,
):
    """
    Resolve label switching by comparing
    estimated component means to truth.

    Parameters
    ----------
    family : str

    alpha_hat : list

    true_alpha1 : list

    true_alpha2 : list

    Returns
    -------
    reordered alpha_hat
    """

    # ========================================================
    # Extract estimated locations
    # ========================================================

    if family == "normal_t":

        est_mu1 = np.asarray(
            alpha_hat[0][0]
        )

        est_mu2 = np.asarray(
            alpha_hat[1][0]
        )

        true_mu1 = np.asarray(
            true_alpha1[0]
        )

        true_mu2 = np.asarray(
            true_alpha2[0]
        )

    # ========================================================
    # Normal + skew-normal
    # ========================================================

    elif family == "normal_skewnormal":

        est_mu1 = np.asarray(
            alpha_hat[0][0]
        )

        est_mu2 = np.asarray(
            alpha_hat[1][0]
        )

        true_mu1 = np.asarray(
            true_alpha1[0]
        )

        true_mu2 = np.asarray(
            true_alpha2[0]
        )

    else:

        raise ValueError(
            f"Unknown family: {family}"
        )

    # ========================================================
    # Distance without swapping
    # ========================================================

    d_identity = (

        np.linalg.norm(
            est_mu1 - true_mu1
        )

        +

        np.linalg.norm(
            est_mu2 - true_mu2
        )
    )

    # ========================================================
    # Distance with swapping
    # ========================================================

    d_swap = (

        np.linalg.norm(
            est_mu1 - true_mu2
        )

        +

        np.linalg.norm(
            est_mu2 - true_mu1
        )
    )

    # ========================================================
    # Swap if needed
    # ========================================================

    if d_swap < d_identity:

        return [

            alpha_hat[1],

            alpha_hat[0],
        ]

    return alpha_hat


# ============================================================
# Parameter conversion
# ============================================================

def params_to_vector(
    params,
):
    """
    Dictionary -> vector.
    """

    return np.array(

        list(params.values()),

        dtype=float,
    )


# ============================================================
# Pretty printing
# ============================================================

def print_parameter_summary(
    params,
):
    """
    Compact parameter summary.
    """

    print("\nParameter Summary")

    print("-" * 50)

    for k, v in params.items():

        print(f"{k:15s}: {v}")

    print("-" * 50)

def enforce_family_order(
    family,
    pi_hat,
    alpha_hat,
    z_hat,
):
    """
    Enforce deterministic component ordering.

    Returns
    -------
    pi_hat_new
    alpha_hat_new
    z_hat_new
    """

    if family in [

        "normal_t",

        "normal_skewnormal",
    ]:

        lengths = [

            len(a)

            for a in alpha_hat
        ]

        special_idx = lengths.index(3)

        normal_idx = 1 - special_idx

        order = [

            normal_idx,

            special_idx,
        ]

    else:

        return (
            pi_hat,
            alpha_hat,
            z_hat,
        )

    # ----------------------------------------------------
    # Reorder parameters
    # ----------------------------------------------------

    pi_new = np.array([

        pi_hat[order[0]],

        pi_hat[order[1]],
    ])

    alpha_new = [

        alpha_hat[order[0]],

        alpha_hat[order[1]],
    ]

    # ----------------------------------------------------
    # Relabel cluster assignments
    # ----------------------------------------------------

    mapping = {

        order[0]: 0,

        order[1]: 1,
    }

    z_new = np.array([

        mapping[z]

        for z in z_hat
    ])

    return (

        pi_new,

        alpha_new,

        z_new,
    )
