# inference/sandwich.py

import numpy as np

from inference.scores import (
    compute_J_matrix,
)

from inference.boundary import (
    compute_boundary_information,decision_function,
)
from inference.utils import (

    flatten_full_parameters,

    reconstruct_full_parameters,
)
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_t
# ============================================================
# Numerical stabilization
# ============================================================

RIDGE = 1e-8


# ============================================================
# Matrix utilities
# ============================================================

def symmetrize_matrix(A):
    """
    Force symmetry.
    """

    return 0.5 * (A + A.T)


def stabilize_matrix(
    A,
    ridge=RIDGE,
):
    """
    Add small ridge stabilization.
    """

    A = symmetrize_matrix(A)

    A += ridge * np.eye(A.shape[0])

    return A


def safe_inverse(
    A,
    ridge=RIDGE,
):
    """
    Stable matrix inverse.

    Uses pseudo-inverse fallback if needed.
    """

    A = stabilize_matrix(
        A,
        ridge=ridge,
    )

    try:

        return np.linalg.inv(A)

    except np.linalg.LinAlgError:

        return np.linalg.pinv(A)



# ============================================================
# Build corrected Hessian
# ============================================================

def compute_corrected_hessian(
    X,
    family,
    fit,
    boundary_information,
    eps=1e-5,
):
    """
    Construct reduced corrected Hessian:

        H = H_classified + B

    where H_classified is the numerical Hessian of the
    classified log-likelihood.
    """

    import numpy as np
    from scipy.optimize import approx_fprime
    from scipy.stats import multivariate_normal, multivariate_t

    from inference.boundary import decision_function, logpdf_skewnormal

    # --------------------------------------------------------
    # Extract fitted parameters
    # --------------------------------------------------------

    pi_hat = fit["pi_hat"]
    alpha_hat = fit["alpha_hat"]
    # --------------------------------------------------------
    # Full parameter vector
    # --------------------------------------------------------

    theta0 = flatten_full_parameters(

        family=family,

        pi_hat=pi_hat,

        alpha_hat=alpha_hat,
    )    


    # --------------------------------------------------------
    # Pre-compute fixed cluster assignments based on MLE
    # --------------------------------------------------------

    assignments = []
    for x in X:
        g = decision_function(
            x,
            family=family,
            pi_hat=pi_hat,
            alpha_hat=alpha_hat,
        )
        assignments.append(g >= 0)

    # --------------------------------------------------------
    # Classified log-likelihood
    # --------------------------------------------------------

    def classified_loglik(theta):
        """
        Classified log-likelihood.
        """

        pi_local, alpha_local = (

            reconstruct_full_parameters(

                theta=theta,

                family=family,
            )
        )

        total = 0.0

        for i, x in enumerate(X):

            g_is_positive = assignments[i]

            # ====================================================
            # Normal-t
            # ====================================================

            if family == "normal_t":

                mu1, Sigma1 = alpha_local[0]

                mu2, Sigma2, nu = alpha_local[1]

                nu = float(np.ravel(nu)[0])

                if g_is_positive:

                    logf = multivariate_normal.logpdf(

                        x,

                        mean=mu1,

                        cov=Sigma1,
                    )

                    total += (
                        np.log(pi_local[0]) + logf
                    )

                else:

                    logf = multivariate_t.logpdf(

                        x,

                        loc=mu2,

                        shape=Sigma2,

                        df=nu,
                    )

                    total += (
                        np.log(pi_local[1]) + logf
                    )

            # ====================================================
            # Normal-skewnormal
            # ====================================================

            elif family == "normal_skewnormal":

                mu1, Sigma1 = alpha_local[0]

                xi2, Omega2, alpha_vec = alpha_local[1]

                if g_is_positive:

                    logf = multivariate_normal.logpdf(

                        x,

                        mean=mu1,

                        cov=Sigma1,
                    )

                    total += (
                        np.log(pi_local[0]) + logf
                    )

                else:

                    logf = logpdf_skewnormal(

                        x,

                        xi=xi2,

                        Omega=Omega2,

                        alpha=alpha_vec,
                    )

                    total += (
                        np.log(pi_local[1]) + logf
                    )

        return total / len(X)

    # --------------------------------------------------------
    # Numerical Hessian
    # --------------------------------------------------------

    p = len(theta0)
    Hc = np.zeros((p, p))

    def grad_fn(theta):
        return approx_fprime(
            theta,
            classified_loglik,
            epsilon=eps,
        )

    for j in range(p):

        e_j = np.zeros(p)
        e_j[j] = eps

        grad_plus = grad_fn(theta0 + e_j)
        grad_minus = grad_fn(theta0 - e_j)

        Hc[:, j] = (grad_plus - grad_minus) / (2.0 * eps)

    Hc = 0.5 * (Hc + Hc.T)

    # --------------------------------------------------------
    # Corrected Hessian
    # --------------------------------------------------------

    H = Hc + boundary_information
    H = stabilize_matrix(H)

    return H

# ============================================================
# Compute sandwich covariance
# ============================================================

def compute_sandwich_covariance(
    H,
    J,
):
    """
    Sandwich covariance estimator:

        V = H^{-1} J H^{-1}

    Returns
    -------
    ndarray
    """

    H_inv = safe_inverse(H)

    V = H_inv @ J @ H_inv.T

    V = symmetrize_matrix(V)

    return V


# ============================================================
# Main boundary-corrected covariance
# ============================================================

def compute_boundary_corrected_covariance(
    X,
    family,
    fit,
    scenario,
):
    """
    Compute full boundary-corrected covariance matrix.

    Implements:

        H = H_c + B

        V = H^{-1} J H^{-1}

    Parameters
    ----------
    X : ndarray

    family : str

    fit : dict

    scenario : dict

    Returns
    -------
    ndarray
    """

    # --------------------------------------------------------
    # Extract fitted quantities
    # --------------------------------------------------------

    pi_hat = fit["pi_hat"]

    alpha_hat = fit["alpha_hat"]

    # --------------------------------------------------------
    # Complete-data information
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Boundary information
    # --------------------------------------------------------

    B = compute_boundary_information(

        X=X,

        family=family,

        pi_hat=pi_hat,

        alpha_hat=alpha_hat,
    )

    # --------------------------------------------------------
    # Corrected Hessian
    # --------------------------------------------------------

    H = compute_corrected_hessian(
    X=X,
    family=family,
    fit=fit,
    boundary_information=B,
)

    # --------------------------------------------------------
    # Score covariance
    # --------------------------------------------------------

    J = compute_J_matrix(

        X=X,

        family=family,

        pi_hat=pi_hat,

        alpha_hat=alpha_hat,
    )

    # --------------------------------------------------------
    # Sandwich covariance
    # --------------------------------------------------------

    V = compute_sandwich_covariance(

        H=H,

        J=J,
    )

    return V / len(X)


# ============================================================
# Naive covariance
# ============================================================

def compute_naive_covariance(
    X,
    family,
    fit,
):
    """
    Naive covariance estimator using only the
    classified likelihood Hessian without
    boundary correction (sandwich form for M-estimation).
    """

    p = len(

        flatten_full_parameters(

            family=family,

            pi_hat=fit["pi_hat"],

            alpha_hat=fit["alpha_hat"],
        )
    )

    H_naive = compute_corrected_hessian(

        X=X,

        family=family,

        fit=fit,

        boundary_information=np.zeros((p, p)),
    )

    J = compute_J_matrix(
        X=X,
        family=family,
        pi_hat=fit["pi_hat"],
        alpha_hat=fit["alpha_hat"],
    )

    cov = compute_sandwich_covariance(H_naive, J)

    return cov / len(X)

# ============================================================
# Standard errors
# ============================================================

def covariance_to_se(
    V,
):
    """
    Convert covariance matrix to standard errors.
    """

    se = np.sqrt(

        np.maximum(
            np.diag(V),
            1e-12,
        )
    )

    return se


# ============================================================
# Confidence intervals
# ============================================================

def confidence_intervals(
    theta_hat,
    se,
    alpha=0.05,
):
    """
    Wald confidence intervals.

    Returns
    -------
    lower, upper
    """

    z = 1.959963984540054

    lower = theta_hat - z * se

    upper = theta_hat + z * se

    return lower, upper


# ============================================================
# Coverage indicator
# ============================================================

def coverage_indicator(
    theta_true,
    lower,
    upper,
):
    """
    Indicator for CI coverage.
    """

    return float(

        (theta_true >= lower)
        and
        (theta_true <= upper)
    )


# ============================================================
# Diagnostic summary
# ============================================================

def summarize_information_matrices(
    H,
    J,
    V,
):
    """
    Numerical diagnostics for inference stability.

    Returns
    -------
    dict
    """

    eig_H = np.linalg.eigvalsh(H)

    eig_J = np.linalg.eigvalsh(J)

    eig_V = np.linalg.eigvalsh(V)

    return {

        "min_eig_H":
            np.min(eig_H),

        "max_eig_H":
            np.max(eig_H),

        "condition_H":
            np.linalg.cond(H),

        "min_eig_J":
            np.min(eig_J),

        "max_eig_J":
            np.max(eig_J),

        "condition_J":
            np.linalg.cond(J),

        "min_eig_V":
            np.min(eig_V),

        "max_eig_V":
            np.max(eig_V),

        "condition_V":
            np.linalg.cond(V),
    }
