"""Prespecified real-data screen for scientifically useful boundary examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from cem_inference import (
    NormalAdapter, SkewNormalAdapter, StudentTAdapter,
    estimate_boundary_curvature, fit_fmvmm_hard, sandwich_covariance,
    select_pairwise_bandwidths, stabilize_boundary_curvature,
)


def load_dataset(name: str):
    if name == "breast_cancer":
        bunch = load_breast_cancer(); return bunch.data, bunch.target
    if name == "wine":
        bunch = load_wine(); return bunch.data, bunch.target
    if name == "digits_0_1":
        bunch = load_digits(); mask = np.isin(bunch.target, [0, 1])
        return bunch.data[mask], bunch.target[mask]
    if name.startswith("dry_bean_"):
        pairs = {
            "dry_bean_cali_barbunya": ("CALI", "BARBUNYA"),
            "dry_bean_dermason_seker": ("DERMASON", "SEKER"),
            "dry_bean_horoz_sira": ("HOROZ", "SIRA"),
        }
        first, second = pairs[name]
        path = Path("data/external/dry_bean/DryBeanDataset/Dry_Bean_Dataset.arff")
        records, _ = arff.loadarff(path)
        frame = pd.DataFrame(records)
        labels = frame.pop("Class").str.decode("utf-8")
        mask = labels.isin([first, second])
        return frame.loc[mask].to_numpy(float), labels.loc[mask].to_numpy()
    raise KeyError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dataset in (
        "breast_cancer", "wine", "digits_0_1",
        "dry_bean_cali_barbunya", "dry_bean_dermason_seker", "dry_bean_horoz_sira",
    ):
        raw, labels = load_dataset(dataset)
        standardized = StandardScaler().fit_transform(raw)
        for dimension in (2, 5):
            if dimension >= min(standardized.shape):
                continue
            pca = PCA(n_components=dimension, svd_solver="full")
            observations = pca.fit_transform(standardized)
            for family, adapters in (
                ("normal_t", [NormalAdapter(dimension), StudentTAdapter(dimension)]),
                ("normal_skewnormal", [NormalAdapter(dimension), SkewNormalAdapter(dimension)]),
            ):
                base = {
                    "dataset": dataset, "family": family, "p": dimension,
                    "n": len(observations),
                    "pca_variance_explained": pca.explained_variance_ratio_.sum(),
                }
                try:
                    fit = fit_fmvmm_hard(observations, adapters, max_iter=160, tol=1e-6)
                    model, coordinates = fit.model, fit.coordinates
                    scores = model.classified_score_matrix(observations, coordinates)
                    fixed = model.fixed_classification_information(observations, coordinates)
                    bandwidths, diagnostics = select_pairwise_bandwidths(
                        model, observations, coordinates, multiplier=1.06
                    )
                    boundary = estimate_boundary_curvature(
                        model, observations, coordinates, bandwidth=bandwidths
                    )
                    zeros = np.zeros_like(fixed)
                    naive, _, _ = sandwich_covariance(fixed, zeros, scores)
                    stabilized_boundary, shrinkage, raw_relative = stabilize_boundary_curvature(
                        fixed, boundary, len(observations)
                    )
                    stabilized, stabilized_curvature, _ = sandwich_covariance(
                        fixed, stabilized_boundary, scores
                    )
                    naive_se = np.sqrt(np.maximum(np.diag(naive), 0) / len(observations))
                    stabilized_se = np.sqrt(
                        np.maximum(np.diag(stabilized), 0) / len(observations)
                    )
                    ratio = stabilized_se / naive_se
                    assignments = model.assignments(observations, coordinates)
                    rows.append({
                        **base, "status": "success",
                        "classification_objective": fit.classification_objective,
                        "objective_gap": fit.objective_gap,
                        "minimum_weight": fit.weights.min(),
                        "selected_admissible": fit.selected_admissible,
                        "initialization": fit.initialization,
                        "ari_external_labels": adjusted_rand_score(labels, assignments),
                        "minimum_kernel_support": min(
                            item.observations_in_kernel_support for item in diagnostics
                        ),
                        "median_se_inflation": np.median(ratio),
                        "maximum_se_inflation": np.max(ratio),
                        "boundary_shrinkage_factor": shrinkage,
                        "raw_generalized_boundary_eigenvalue": raw_relative,
                        "stabilized_minimum_curvature_eigenvalue": np.linalg.eigvalsh(
                            stabilized_curvature
                        ).min(),
                    })
                except Exception as error:
                    rows.append({
                        **base, "status": "failed",
                        "error_message": f"{type(error).__name__}: {error}",
                    })
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
