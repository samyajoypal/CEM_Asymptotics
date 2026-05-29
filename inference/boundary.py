# inference/boundary.py

import numpy as np

from scipy.stats import multivariate_normal
from scipy.stats import multivariate_t
from scipy.optimize import approx_fprime
from inference.utils import (

    flatten_full_parameters,

    reconstruct_full_parameters,

    get_parameter_names,
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
# Skew-normal approximation
# ============================================================

def logpdf_skewnormal(
    x,
    xi,
    Omega,
    alpha,
):
    """
    Approximate multivariate skew-normal log-density.
    """

    log_phi = multivariate_normal.logpdf(

        x,

        mean=xi,

        cov=Omega,
    )

    z = np.linalg.solve(

        np.linalg.cholesky(Omega),

        x - xi,
    )

    skew_term = np.dot(
        alpha,
        z,
    )

    return log_phi + np.log(
        2.0 / (1.0 + np.exp(-skew_term))
    )


# ============================================================
# Pairwise decision function
# ============================================================

def decision_function(
    x,
    family,
    pi_hat,
    alpha_hat,
):
    """
    Pairwise decision function.
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

    return log1 - log2


# ============================================================
# Build decision grid
# ============================================================

def build_decision_grid(
    X,
    family,
    pi_hat,
    alpha_hat,
    grid_size=250,
    padding=2.0,
):
    """
    Evaluate decision function on dense grid.
    """

    x_min = X[:, 0].min() - padding

    x_max = X[:, 0].max() + padding

    y_min = X[:, 1].min() - padding

    y_max = X[:, 1].max() + padding

    xx = np.linspace(

        x_min,

        x_max,

        grid_size,
    )

    yy = np.linspace(

        y_min,

        y_max,

        grid_size,
    )

    GX, GY = np.meshgrid(
        xx,
        yy,
    )

    G = np.zeros_like(GX)

    for i in range(grid_size):

        for j in range(grid_size):

            point = np.array([

                GX[i, j],

                GY[i, j],
            ])

            G[i, j] = decision_function(

                point,

                family=family,

                pi_hat=pi_hat,

                alpha_hat=alpha_hat,
            )

    return {

        "GX": GX,

        "GY": GY,

        "G": G,
    }


# ============================================================
# Approximate boundary points
# ============================================================

def extract_boundary_points(
    grid_object,
    threshold=0.05,
):
    """
    Approximate decision boundary using:

        |g(x)| < threshold
    """

    GX = grid_object["GX"]

    GY = grid_object["GY"]

    G = grid_object["G"]

    mask = np.abs(G) < threshold

    x_coords = GX[mask]

    y_coords = GY[mask]

    if len(x_coords) == 0:

        return None

    boundary_points = np.column_stack([

        x_coords,

        y_coords,
    ])

    return boundary_points


# ============================================================
# Compatibility wrapper
# ============================================================

def extract_boundary_contour(
    grid_object,
    threshold=0.05,
):
    """
    Compatibility wrapper for plotting module.

    Returns same output as boundary points.
    """

    return extract_boundary_points(

        grid_object,

        threshold=threshold,
    )


# ============================================================
# Numerical parameter gradient
# ============================================================

def parameter_gradient(
    x,
    family,
    pi_hat,
    alpha_hat,
):
    """
    Numerical gradient wrt full parameter vector.
    """

    # --------------------------------------------------------
    # Flatten full parameter vector
    # --------------------------------------------------------

    theta0 = flatten_full_parameters(

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

        return decision_function(

            x,

            family=family,

            pi_hat=pi_local,

            alpha_hat=alpha_local,
        )

    grad = approx_fprime(

        theta0,

        wrapped,

        epsilon=EPS,
    )

    return grad


# ============================================================
# Spatial gradient
# ============================================================

def spatial_gradient(
    x,
    family,
    pi_hat,
    alpha_hat,
):
    """
    Numerical gradient wrt x.
    """

    def wrapped(z):

        return decision_function(

            z,

            family=family,

            pi_hat=pi_hat,

            alpha_hat=alpha_hat,
        )

    grad = approx_fprime(

        x,

        wrapped,

        epsilon=EPS,
    )

    return grad


# ============================================================
# Boundary information matrix
# ============================================================

def compute_boundary_information(
    X,
    family,
    pi_hat,
    alpha_hat,
):
    """
    Approximate boundary information matrix.
    """

    # ========================================================
    # Build grid
    # ========================================================

    grid_object = build_decision_grid(

        X=X,

        family=family,

        pi_hat=pi_hat,

        alpha_hat=alpha_hat,
    )
    
    GX = grid_object["GX"]
    GY = grid_object["GY"]
    dx = GX[0, 1] - GX[0, 0]
    dy = GY[1, 0] - GY[0, 0]
    threshold = 0.05

    # ========================================================
    # Extract boundary
    # ========================================================

    boundary_points = extract_boundary_points(

        grid_object,

        threshold=threshold,
    )

    if boundary_points is None:

        raise RuntimeError(
            "No boundary points detected."
        )

    # ========================================================
    # Parameter dimension
    # ========================================================

    p_dim = len(

    flatten_full_parameters(

        family=family,

        pi_hat=pi_hat,

        alpha_hat=alpha_hat,
    )
)

    B = np.zeros((p_dim, p_dim))

    # ========================================================
    # Approximate integral
    # ========================================================

    for point in boundary_points:

        grad_theta = parameter_gradient(

            point,

            family=family,

            pi_hat=pi_hat,

            alpha_hat=alpha_hat,
        )
        
        pi1 = pi_hat[0]
        pi2 = pi_hat[1]
        
        if family == "normal_t":
            mu1, Sigma1 = alpha_hat[0]
            mu2, Sigma2, nu = alpha_hat[1]
            f1 = np.exp(logpdf_normal(point, mu1, Sigma1))
            f2 = np.exp(logpdf_t(point, mu2, Sigma2, nu))
            f_x = pi1 * f1 + pi2 * f2
        elif family == "normal_skewnormal":
            mu1, Sigma1 = alpha_hat[0]
            xi2, Omega2, alpha = alpha_hat[1]
            f1 = np.exp(logpdf_normal(point, mu1, Sigma1))
            f2 = np.exp(logpdf_skewnormal(point, xi2, Omega2, alpha))
            f_x = pi1 * f1 + pi2 * f2

        contribution = f_x * np.outer(

            grad_theta,

            grad_theta,
        )

        B += contribution * dx * dy / (2.0 * threshold)

    return B
