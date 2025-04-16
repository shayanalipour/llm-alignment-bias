import numpy as np
import pandas as pd
import argparse

from pathlib import Path
from scipy.stats import pearsonr, bootstrap, norm
from statsmodels.stats.multitest import multipletests
from typing import List, Tuple


def get_demographics(df, method):
    return [
        col.replace(f"{method}_", "")
        for col in df.columns
        if col.startswith(f"{method}_")
    ]


### correlations with bootstrapped confidence intervals ###
def correlation_coefficient(x, y):
    r, _ = pearsonr(x, y)
    return r


def bootstrap_correlation(data: pd.DataFrame) -> Tuple[float, float]:

    result = bootstrap(
        data=(data.iloc[:, 0], data.iloc[:, 1]),
        statistic=correlation_coefficient,
        paired=True,
        n_resamples=1000,
        vectorized=False,
        random_state=np.random.default_rng(),
        method="percentile",
    )
    return result.confidence_interval.low, result.confidence_interval.high


def analyze_correlations_bootstrap(
    dataset_name: str,
    df: pd.DataFrame,
    models: List[str],
    alpha: float = 0.05,
    method: str = "average",
) -> pd.DataFrame:
    demographics = get_demographics(df, method)
    results = []

    for model in models:
        for demo in demographics:
            demo_col = f"{method}_{demo}"
            model = f"{model}_cot"
            data = df[[demo_col, model]].dropna()

            if data.empty:
                print(
                    f"no valid data for {demo_col} and {model}, skipping this combination."
                )
                continue

            correlation, p_value = pearsonr(data[demo_col], data[model])

            try:
                ci_lower, ci_upper = bootstrap_correlation(data)
            except Exception as e:
                print(f"bootstrap failed for {demo_col} and {model}. error: {str(e)}")
                ci_lower, ci_upper = np.nan, np.nan

            results.append(
                {
                    "dataset": dataset_name,
                    "model": model,
                    "demographic": demo,
                    "correlation": correlation,
                    "p_value": p_value,
                    "sample_size": len(data),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

    if not results:
        print(f"no valid results for {method} method.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df["p_corrected"] = multipletests(results_df["p_value"], method="holm")[1]
    results_df["significant"] = results_df["p_corrected"] < alpha

    return results_df


### correlations with fisher's z transformation ###
def fisher_z_transformation(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float, float, float]:
    r, p = pearsonr(x, y)
    n = len(x)
    if n < 4:
        raise ValueError("n must be >= 4")

    # fisher z transformation
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_critical = norm.ppf(1 - alpha / 2)
    z_lower = z - z_critical * se
    z_upper = z + z_critical * se
    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)
    return r, p, r_lower, r_upper


def analyze_correlations_fisher(
    dataset_name: str, df: pd.DataFrame, models: List[str], method: str = "average"
) -> pd.DataFrame:
    demographics = get_demographics(df, method)
    results = []
    for model in models:
        for demo in demographics:
            demo_col = f"{method}_{demo}"
            model_col = f"{model}_cot"
            data = df[[demo_col, model_col]].dropna()
            if data.empty:
                print(
                    f"no valid data for {demo_col} and {model_col}, skipping this combination."
                )
                continue
            x = data[demo_col]
            y = data[model_col]
            try:
                correlation, p_value, ci_lower, ci_upper = fisher_z_transformation(x, y)
            except Exception as e:
                print(
                    f"fisher z transformation failed for {demo_col} and {model_col}. error: {str(e)}"
                )
                ci_lower, ci_upper = np.nan, np.nan

            results.append(
                {
                    "dataset": dataset_name,
                    "model": model,
                    "demographic": demo,
                    "correlation": correlation,
                    "p_value": p_value,
                    "sample_size": len(data),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )
    if not results:
        print(f"no valid results for {method} method.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    results_df["p_corrected"] = multipletests(results_df["p_value"], method="holm")[1]
    results_df["significant"] = results_df["p_corrected"] < 0.05
    print(f"all correlations were significant: {results_df['significant'].all()}")
    return results_df


### main function ###
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dataset_names", nargs="+", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--method", type=str, default="average")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    all_results = []
    for dataset_name in args.dataset_names:
        df = pd.read_csv(args.data_dir / f"{dataset_name}_human_all_models.csv")
        results = analyze_correlations_fisher(
            dataset_name, df, args.models, args.method
        )
        # results.to_csv(output_dir / f"{dataset_name}_human_vs_model.csv", index=False)
        all_results.append(results)
    all_results = pd.concat(all_results)
    all_results.to_csv(args.output_dir / "human_vs_model.csv", index=False)


if __name__ == "__main__":
    main()
