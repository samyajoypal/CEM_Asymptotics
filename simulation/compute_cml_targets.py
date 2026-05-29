import os
import json
import time
import numpy as np

from simulation.scenarios import get_scenario_grid
from simulation.monte_carlo import generate_data, fit_cem_model
from inference.utils import flatten_full_parameters, get_parameter_names

def compute_targets():
    # We only need one huge N per distinct scenario (not for every N in the grid)
    # We will use N=1,000,000 to approximate the population target.
    # Note: 1,000,000 might be too large for RAM depending on the system, let's try 1,000,000 or 500,000.
    N_MASSIVE = 100000 
    
    # Get all scenarios but just with one sample size.
    grid = get_scenario_grid(family_pair="all", sample_sizes=[N_MASSIVE])
    
    targets = {}
    
    for scenario in grid:
        family = scenario["family"]
        scenario_name = scenario["scenario_name"]
        
        print(f"Computing CML target for {family} - {scenario_name} (N={N_MASSIVE})...")
        
        # 1. Generate massive dataset
        # We use a fixed seed so it's reproducible.
        X, labels_true = generate_data(scenario, random_state=42)
        
        # 2. Fit CEM model
        t0 = time.time()
        fit = fit_cem_model(X=X, family=family, scenario=scenario, selection_method="oracle")
        t1 = time.time()
        print(f"  Fit took {t1-t0:.2f} seconds.")
        
        # 3. Extract and flatten parameters
        theta_vector = flatten_full_parameters(
            family=family,
            pi_hat=fit["pi_hat"],
            alpha_hat=fit["alpha_hat"],
        )
        
        parameter_names = get_parameter_names(family)
        theta_star = dict(zip(parameter_names, theta_vector))
        
        # 4. Save to dict
        if family not in targets:
            targets[family] = {}
        targets[family][scenario_name] = theta_star
        
        # For debugging, let's print the bias from the true generating params
        true_params = scenario["true_params"]
        bias_norm = 0.0
        for p_name in parameter_names:
            bias = theta_star[p_name] - true_params[p_name]
            bias_norm += bias**2
        bias_norm = np.sqrt(bias_norm)
        print(f"  Distance from true params (theta_0): {bias_norm:.6f}")
        
    # Write to JSON
    out_path = os.path.join(os.path.dirname(__file__), "cml_targets.json")
    with open(out_path, "w") as f:
        json.dump(targets, f, indent=4)
    print(f"Targets saved to {out_path}")

if __name__ == "__main__":
    compute_targets()
