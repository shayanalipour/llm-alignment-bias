import pandas as pd
import numpy as np
import argparse

from pathlib import Path
from itertools import combinations
from scipy.stats import pearsonr, bootstrap, norm
from statsmodels.stats.multitest import multipletests


def get_demographic_pairs(data, gt_type="average"):
    prefix = f"{gt_type}_"
    demographic_columns = [col for col in data.columns if col.startswith(prefix)]

    gender_columns = sorted(
        [col for col in demographic_columns if col in ["man", "woman"]]
    )

    race_columns = sorted(
        [
            col
            for col in demographic_columns
            if col.split("_")[-1] in ["asian", "black", "white", "hispanic"]
        ]
    )

    gender_combinations = list(combinations(gender_columns, 2))
    race_combinations = list(combinations(race_columns, 2))

    return {
        "gender": gender_combinations,
        "race": race_combinations,
    }


### correlation difference with bootstrapped confidence intervals ###
def pearson_difference(x, y, z):
    return pearsonr(x, z)[0] - pearsonr(y, z)[0]


def ci_to_pvalue(low, high):
    if low > 0 or high < 0:
        point_estimate = max(abs(low), abs(high))
        se = (high - low) / (2 * norm.ppf(0.975))
        z_score = point_estimate / se
        return 2 * (1 - norm.cdf(z_score))
    return 1.0


def get_bootstrap_corr_diff(data, dataset_name, model):
    bootstrap_results = []
    demographic_pairs = get_demographic_pairs(data)
    for _, pairs in demographic_pairs.items():
        for pair in pairs:
            pair_df = data[[pair[0], pair[1], f"{model}_cot"]].dropna()

            if pair_df.shape[0] < 2:
                continue

            pair_results = bootstrap(
                data=pair_df.values.T,
                statistic=pearson_difference,
                n_resamples=1000,
                paired=True,
                vectorized=False,
                random_state=np.random.default_rng(),
                method="percentile",
            )

            ci_lower = pair_results.confidence_interval.low
            ci_upper = pair_results.confidence_interval.high
            p_value_from_ci = ci_to_pvalue(ci_lower, ci_upper)
            corrected_p_value = multipletests([p_value_from_ci], method="holm")[1][0]

            bootstrap_results.append(
                {
                    "demo_1": pair[0],
                    "demo_2": pair[1],
                    "model": model,
                    "dataset": dataset_name,
                    "sample_size": pair_df.shape[0],
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value_from_ci": p_value_from_ci,
                    "corrected_p_value": corrected_p_value,
                }
            )

    bootstrap_results_df = pd.DataFrame(bootstrap_results)
    bootstrap_results_df["significant"] = (
        bootstrap_results_df["corrected_p_value"] < 0.05
    )
    return bootstrap_results_df


### correlation difference with steiger's test ###
def steigers_z_test(r12, r13, r23, n):
    def fisher_z(r):
        return 0.5 * np.log((1 + r) / (1 - r))

    r_avg = (r12 + r13) / 2
    f = (1 - r23) / (2 * (1 - r_avg**2))
    h = (1 - f * r_avg**2) / (1 - r_avg**2)

    z = (fisher_z(r12) - fisher_z(r13)) * np.sqrt((n - 3) / (2 * (1 - r23) * h))
    p = 2 * (1 - norm.cdf(abs(z)))

    return z, p


def get_steigers_corr_diff(data, dataset_name, model):
    results = []
    demographic_pairs = get_demographic_pairs(data)
    for _, pairs in demographic_pairs.items():
        for pair in pairs:
            pair_df = data[[pair[0], pair[1], f"{model}_cot"]].dropna()

            n = pair_df.shape[0]
            if n < 4:
                continue

            r12 = pearsonr(pair_df[pair[0]], pair_df[f"{model}_cot"])[0]
            r13 = pearsonr(pair_df[pair[1]], pair_df[f"{model}_cot"])[0]
            r23 = pearsonr(pair_df[pair[0]], pair_df[pair[1]])[0]

            z, p = steigers_z_test(r12, r13, r23, n)

            results.append(
                {
                    "demo_1": pair[0],
                    "demo_2": pair[1],
                    "model": model,
                    "dataset": dataset_name,
                    "sample_size": n,
                    "z": z,
                    "p_value": p,
                }
            )

    results_df = pd.DataFrame(results)
    results_df["corrected_p_value"] = multipletests(
        results_df["p_value"], method="holm"
    )[1]
    results_df["significant"] = results_df["corrected_p_value"] < 0.05
    return results_df


### main function ###
def run_corr_diff(datasets, models, data_dir, output_dir):
    bootstrap_results = []
    steigers_results = []
    for dataset_name in datasets:
        data = pd.read_csv(data_dir / f"{dataset_name}_human_all_models.csv")
        for model in models:
            bootstrap_results.append(get_bootstrap_corr_diff(data, dataset_name, model))
            steigers_results.append(get_steigers_corr_diff(data, dataset_name, model))

    bootstrap_results_df = pd.concat(bootstrap_results)
    steigers_results_df = pd.concat(steigers_results)

    # merge results
    bootstrap_results_df = bootstrap_results_df[
        ["demo_1", "demo_2", "model", "dataset", "ci_lower", "ci_upper"]
    ]
    results_df = bootstrap_results_df.merge(
        steigers_results_df, on=["demo_1", "demo_2", "model", "dataset"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "robust_human_vs_model.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dataset_names", nargs="+", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    run_corr_diff(args.dataset_names, args.models, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
