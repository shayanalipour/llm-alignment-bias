import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Tuple
from scipy.stats import pearsonr

import argparse


### load & prepare data ###
def load_data(input_file, valid_labels) -> pd.DataFrame:
    df = pd.read_csv(input_file, usecols=["post_id", "worker_id", "worker_label"])
    df["worker_label"] = (
        pd.to_numeric(df["worker_label"], errors="coerce").fillna(0).astype(int)
    )
    df = df[df["worker_label"].isin(valid_labels)]

    post_counts = df.groupby("post_id").size()
    posts_to_keep = post_counts[post_counts > 1].index
    df = df[df["post_id"].isin(posts_to_keep)]

    # each worker can only annotate a post once
    duplicates = df.groupby(["post_id", "worker_id"]).size().reset_index(name="count")
    duplicates = duplicates[duplicates["count"] > 1]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} posts with duplicate worker IDs")
        posts_to_remove = duplicates["post_id"].unique()
        df = df[~df["post_id"].isin(posts_to_remove)]
        print(f"Removed {len(posts_to_remove)} posts with duplicate worker IDs")

    print(f"Loaded {len(df)} annotations for {df['post_id'].nunique()} posts")
    return df


def validate_data(df: pd.DataFrame, valid_labels) -> pd.DataFrame:
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found missing values:\n{missing_values}")
        df = df.dropna()
        print(f"Removed rows with missing values. New shape: {df.shape}")

    # labels out of range
    invalid_labels = df[~df["worker_label"].isin(valid_labels)]
    if not invalid_labels.empty:
        print(f"Found {len(invalid_labels)} rows with invalid labels")
        df = df[df["worker_label"].isin(valid_labels)]
        print(f"Removed rows with invalid labels. New shape: {df.shape}")

    # posts with insufficient annotations
    post_counts = df.groupby("post_id").size()
    insufficient_posts = post_counts[post_counts < 2].index
    if len(insufficient_posts) > 0:
        print(f"Found {len(insufficient_posts)} posts with insufficient annotations")
        df = df[~df["post_id"].isin(insufficient_posts)]
        print(f"Removed posts with insufficient annotations. New shape: {df.shape}")

    return df


### compute human agreement ###
def has_constant_values(x: np.ndarray) -> bool:
    return len(set(x)) == 1


def calculate_ground_truth(df: pd.DataFrame, method: str) -> pd.Series:
    if method == "average":
        return df.groupby("post_id")["worker_label"].mean()
    elif method == "majority":
        return df.groupby("post_id")["worker_label"].agg(
            lambda x: x.value_counts().index[0]
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y):
        return np.nan
    if len(x) < 2:
        return np.nan
    if has_constant_values(x):
        return np.nan
    if has_constant_values(y):
        return np.nan
    try:
        correlation, p_value = pearsonr(x, y)
        return correlation
    except Exception as e:
        return np.nan


def calculate_annotator_correlations(
    df: pd.DataFrame, method: str = "average"
) -> List[float]:
    correlations = []
    for annotator in df["worker_id"].unique():
        annotator_posts = df[df["worker_id"] == annotator]["post_id"].unique()

        if len(annotator_posts) == 1:
            continue

        annotator_labels = []
        ground_truth_labels = []

        for post in annotator_posts:
            post_df = df[df["post_id"] == post]
            annotator_label = post_df[post_df["worker_id"] == annotator][
                "worker_label"
            ].values[0]
            others_df = post_df[post_df["worker_id"] != annotator]

            if method == "average":
                ground_truth = others_df["worker_label"].mean()
            elif method == "majority":
                ground_truth = others_df["worker_label"].mode().values[0]

            annotator_labels.append(annotator_label)
            ground_truth_labels.append(ground_truth)

        correlation = calculate_correlation(
            np.array(annotator_labels), np.array(ground_truth_labels)
        )
        correlations.append(correlation)

    correlations = [c for c in correlations if not np.isnan(c)]
    return correlations


def bootstrap_ci(
    data: List[float], num_bootstrap: int = 1000, ci: float = 0.95
) -> Tuple[float, float]:

    bootstrap_means = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)

    return lower, upper


### main function ###
def calculate_human_agreement(data_dir, max_label, method):
    valid_labels = list(range(1, max_label + 1))
    df = load_data(data_dir, valid_labels)
    df = validate_data(df, valid_labels)

    correlations = calculate_annotator_correlations(df, method)
    human_ci = bootstrap_ci(correlations)

    return np.mean(correlations), human_ci


def main():
    dataset_human_agreement = []

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=["awa", "mhsc", "nlpos", "popq", "sbic"],
        required=True,
    )
    parser.add_argument(
        "--max_labels",
        nargs="+",
        type=int,
        default=[5, 3, 3, 5, 3],
        required=True,
    )
    parser.add_argument(
        "--method",
        type=str,
        default="average",
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
    )

    args = parser.parse_args()
    data_path = args.data_path
    datasets = args.datasets
    max_labels = args.max_labels
    method = args.method
    output_file = args.output_file

    for i, dataset in enumerate(datasets):
        dataset_max_label = max_labels[i]
        data_dir = data_path / f"{dataset}.csv"
        if not data_dir.exists():
            print(f"Data file {data_dir} not found. Skipping {dataset}")
            continue

        print(
            f"Calculating human agreement for {dataset} - max label: {dataset_max_label} - method: {method}"
        )
        mean_correlation, ci = calculate_human_agreement(
            data_dir, dataset_max_label, method
        )
        print(f"Dataset: {dataset}")
        print(f"Mean correlation: {mean_correlation:.4f}")
        print(f"95% CI: {ci[0]:.4f} - {ci[1]:.4f}")
        dataset_human_agreement.append(
            {
                "dataset": dataset,
                "mean_correlation": mean_correlation,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }
        )

    dataset_human_agreement = pd.DataFrame(dataset_human_agreement)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_human_agreement.to_csv(output_file, index=False)
    print(f"Saved human agreement to {output_file}")


if __name__ == "__main__":
    main()
