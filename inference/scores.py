# inference/scores.py

import numpy as np

from scipy.stats import multivariate_normal
from scipy.stats import multivariate_t
from scipy.optimize import approx_fprime
from inference.utils import (

    flatten_full_parameters,

    reconstruct_full_parameters,
)

# ============================================================
# Numerical constants
# ============================================================

EPS = 1e-6


# ============================================================
# Log-density wrappers
# ============================================================

def logpdf_normal(
    x,
    mu,
    Sigma,
):
    """
    Multivariate normal log-density.
    """

    return multivariate_normal.logpdf(

        x,

        mean=mu,

        cov=Sigma,
    )


def logpdf_t(
    x,
    mu,
    Sigma,
    nu,
):
    """
    Multivariate t log-density.
    """

    return multivariate_t.logpdf(

        x,

        loc=mu,

        shape=Sigma,

        df=float(np.ravel(nu)[0]),
    )


# ============================================================
# Approximate skew-normal log-density
# ============================================================

def logpdf_skewnormal(
    x,
    xi,
    Omega,
    alpha,
):
    """
    Approximate multivariate skew-normal log-density.

    NOTE:
    Used for numerical score approximation.
    """

    # --------------------------------------------------------
    # Gaussian core
    # --------------------------------------------------------

    log_phi = multivariate_normal.logpdf(

        x,

        mean=xi,

        cov=Omega,
    )

    # --------------------------------------------------------
    # Skewness term
    # --------------------------------------------------------

    z = np.linalg.solve(

        np.linalg.cholesky(Omega),

        x - xi,
    )

    skew_term = np.dot(alpha, z)

    return log_phi + np.log(
        2.0 / (1.0 + np.exp(-skew_term))
    )


# ============================================================
# Hard classification assignment
# ============================================================

def classify_observation(
    x,
    family,
    pi_hat,
    alpha_hat,
):
    """
    Determine hard component assignment.

    Returns
    -------
    int
    """

    pi1 = pi_hat[0]

    pi2 = pi_hat[1]

    # ========================================================
    # Normal + t
    # ========================================================

    if family == "normal_t":

        mu1, Sigma1 = alpha_hat[0]

        mu2, Sigma2, nu = alpha_hat[1]
        nu = float(np.ravel(nu)[0])

        log1 = np.log(pi1) + logpdf_normal(

            x,

            mu1,

            Sigma1,
        )

        log2 = np.log(pi2) + logpdf_t(

            x,

            mu2,

            Sigma2,

            nu,
        )

    # ========================================================
    # Normal + skew-normal
    # ========================================================

    elif family == "normal_skewnormal":

        mu1, Sigma1 = alpha_hat[0]

        xi2, Omega2, alpha = alpha_hat[1]

        log1 = np.log(pi1) + logpdf_normal(

            x,

            mu1,

            Sigma1,
        )

        log2 = np.log(pi2) + logpdf_skewnormal(

            x,

            xi2,

            Omega2,

            alpha,
        )

    else:

        raise ValueError(
            f"Unknown family: {family}"
        )

    return int(log2 > log1)


# ============================================================
# Parameter vectorization
# ============================================================



# ============================================================
# Observation log-likelihood
# ============================================================

def observation_loglikelihood(
    x,
    family,
    pi_hat,
    alpha_hat,
    fixed_cluster=None,
):
    """
    Hard-classification log-likelihood contribution.
    """

    if fixed_cluster is None:
        cluster = classify_observation(

            x,

            family=family,

            pi_hat=pi_hat,

            alpha_hat=alpha_hat,
        )
    else:
        cluster = fixed_cluster

    # ========================================================
    # Component 1
    # ========================================================

    if cluster == 0:

        mu1, Sigma1 = alpha_hat[0]

        return np.log(pi_hat[0]) + logpdf_normal(

            x,

            mu1,

            Sigma1,
        )

    # ========================================================
    # Component 2
    # ========================================================

    else:

        # ----------------------------------------------------
        # Normal + t
        # ----------------------------------------------------

        if family == "normal_t":

            mu2, Sigma2, nu = alpha_hat[1]
            nu = float(np.ravel(nu)[0])

            return np.log(pi_hat[1]) + logpdf_t(

                x,

                mu2,

                Sigma2,

                nu,
            )

        # ----------------------------------------------------
        # Normal + skew-normal
        # ----------------------------------------------------

        elif family == "normal_skewnormal":

            xi2, Omega2, alpha = alpha_hat[1]

            return np.log(pi_hat[1]) + logpdf_skewnormal(

                x,

                xi2,

                Omega2,

                alpha,
            )

        else:

            raise ValueError(
                f"Unknown family: {family}"
            )


# ============================================================
# Numerical score
# ============================================================

def numerical_score(
    x,
    family,
    pi_hat,
    alpha_hat,
):
    """
    Numerical gradient of hard-classification log-likelihood.

    Returns
    -------
    ndarray
    """

    theta0 = flatten_full_parameters(

    family=family,

    pi_hat=pi_hat,

    alpha_hat=alpha_hat,
)

    # --------------------------------------------------------
    # Pre-compute fixed cluster assignment
    # --------------------------------------------------------

    fixed_cluster = classify_observation(
        x,
        family=family,
        pi_hat=pi_hat,
        alpha_hat=alpha_hat,
    )

    # --------------------------------------------------------
    # Wrapper
    # --------------------------------------------------------

    def wrapped(theta):

        pi_local, alpha_local = (

    reconstruct_full_parameters(

        theta=theta,

        family=family,
    )
)
        return observation_loglikelihood(

            x,

            family=family,

            pi_hat=pi_local,

            alpha_hat=alpha_local,

            fixed_cluster=fixed_cluster,
        )

    # --------------------------------------------------------
    # Numerical gradient
    # --------------------------------------------------------

    grad = approx_fprime(

        theta0,

        wrapped,

        epsilon=EPS,
    )

    return grad


# ============================================================
# Score matrix
# ============================================================

def compute_score_matrix(
    X,
    family,
    pi_hat,
    alpha_hat,
):
    """
    Compute score matrix.

    Returns
    -------
    ndarray shape (n, p)
    """

    scores = []

    for x in X:

        s = numerical_score(

            x,

            family=family,

            pi_hat=pi_hat,

            alpha_hat=alpha_hat,
        )

        scores.append(s)

    return np.asarray(scores)


# ============================================================
# Outer-product covariance
# ============================================================

def compute_J_matrix(
    X,
    family,
    pi_hat,
    alpha_hat,
):
    """
    Compute OPG covariance matrix:

        J = n^{-1} Σ s_i s_i^T

    Returns
    -------
    ndarray
    """

    S = compute_score_matrix(

        X,

        family,

        pi_hat,

        alpha_hat,
    )

    J = S.T @ S

    J /= X.shape[0]

    return J


# ============================================================
# Empirical score mean
# ============================================================

def compute_score_mean(
    X,
    family,
    pi_hat,
    alpha_hat,
):
    """
    Empirical mean score vector.

    Useful diagnostic for asymptotics.
    """

    S = compute_score_matrix(

        X,

        family,

        pi_hat,

        alpha_hat,
    )

    return np.mean(S, axis=0)
