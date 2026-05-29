import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import traceback

from fmvmm.mixtures.FMVMM import fmvmm
from inference.sandwich import compute_naive_covariance, covariance_to_se, compute_boundary_corrected_covariance
from inference.utils import get_parameter_names, enforce_family_order

def load_data(dataset_name="breast_cancer"):
    if dataset_name == "breast_cancer":
        bc = load_breast_cancer()
        X = bc.data
        y = bc.target
    
    # Scale and PCA to 2 dimensions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, y

def compute_distance(alpha1, alpha2):
    # simple euclidean distance between means
    return np.linalg.norm(alpha1[0] - alpha2[0])

def fit_model(X, ref_alpha=None, family="normal_t"):
    dist_1 = ["mvn", "mvt"]
    dist_2 = ["mvt", "mvn"]
    
    m1 = fmvmm(n_clusters=2, list_of_dist=dist_1, specific_comb=True, initialization="kmeans", max_iter=100, tol=1e-5, verbose=False)
    m1.fit(X)
    
    m2 = fmvmm(n_clusters=2, list_of_dist=dist_2, specific_comb=True, initialization="kmeans", max_iter=100, tol=1e-5, verbose=False)
    m2.fit(X)
    
    if ref_alpha is None:
        # Just use m1 for the full data fit
        model = m1
        idx = 0
    else:
        # Compare distances to ref_alpha
        d1 = np.inf
        if len(m1.list_alpha) > 0:
            a1 = m1.list_alpha[0]
            d1 = compute_distance(a1[0], ref_alpha[0]) + compute_distance(a1[1], ref_alpha[1])
            
        d2 = np.inf
        if len(m2.list_alpha) > 0:
            a2 = m2.list_alpha[0]
            d2 = compute_distance(a2[0], ref_alpha[1]) + compute_distance(a2[1], ref_alpha[0])
            
        if d1 <= d2 and d1 != np.inf:
            model = m1
            idx = 0
        elif d2 < d1:
            model = m2
            idx = 0
        else:
            return None
            
    if len(model.list_alpha) == 0:
        return None
        
    pi_hat = model.list_pi[0]
    alpha_hat = model.list_alpha[0]
    z_hat = model.list_cluster[0]
    
    pi_hat, alpha_hat, z_hat = enforce_family_order(family, pi_hat, alpha_hat, z_hat)
    
    return {
        "model": model,
        "pi_hat": pi_hat,
        "alpha_hat": alpha_hat,
        "z_hat": z_hat
    }

def process_bootstrap(X, family, ref_alpha, b_idx):
    np.random.seed(b_idx)
    idx = np.random.choice(len(X), len(X), replace=True)
    X_b = X[idx]
    
    try:
        fit_b = fit_model(X_b, ref_alpha, family=family)
        if fit_b is None:
            return None
        from inference.utils import flatten_full_parameters
        theta_vec = flatten_full_parameters(family, fit_b["pi_hat"], fit_b["alpha_hat"])
        return theta_vec
    except:
        return None

def analyze_dataset(dataset_name="breast_cancer"):
    print(f"\nAnalyzing {dataset_name.upper()}...")
    X, y = load_data(dataset_name)
    family = "normal_t"
    
    print("Fitting model on full dataset...")
    fit = fit_model(X, ref_alpha=None, family=family)
    ref_alpha = fit["alpha_hat"]
    
    print("Computing Naive SE...")
    naive_cov = compute_naive_covariance(X, family, fit)
    naive_se = covariance_to_se(naive_cov)
    
    print("Computing Boundary-Corrected SE...")
    try:
        bc_cov = compute_boundary_corrected_covariance(X, family, fit, scenario={})
        bc_se = covariance_to_se(bc_cov)
    except Exception as e:
        print("BC SE Failed:", e)
        traceback.print_exc()
        return
    
    print("Running Bootstrap (B=500)...")
    B = 500
    results = Parallel(n_jobs=-1)(delayed(process_bootstrap)(X, family, ref_alpha, b) for b in range(B))
    
    valid_results = [r for r in results if r is not None]
    print(f"Bootstrap success rate: {len(valid_results)}/{B}")
    
    boot_thetas = np.array(valid_results)
    
    # Filter out extreme outliers in bootstrap estimates (e.g. collapsed components)
    q1 = np.percentile(boot_thetas, 5, axis=0)
    q3 = np.percentile(boot_thetas, 95, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = np.all((boot_thetas >= lower_bound) & (boot_thetas <= upper_bound), axis=1)
    filtered_thetas = boot_thetas[mask]
    print(f"Filtered Bootstrap rate: {len(filtered_thetas)}/{len(valid_results)}")
    
    boot_sd = np.std(filtered_thetas, axis=0)
    
    param_names = get_parameter_names(family)
    
    df = pd.DataFrame({
        "Parameter": param_names,
        "Naive SE": naive_se,
        "BC SE": bc_se,
        "Boot SD": boot_sd,
    })
    df["Naive/Boot"] = df["Naive SE"] / df["Boot SD"]
    df["BC/Boot"] = df["BC SE"] / df["Boot SD"]
    
    print(df.to_string(index=False))
    df.to_csv(f"results/{dataset_name}_se_comparison.csv", index=False)
    return df

if __name__ == "__main__":
    analyze_dataset("breast_cancer")
