import numpy as np
import pandas as pd
import pickle
import argparse

from pathlib import Path
from scipy import stats


def format_demographic(text):
    sub_parts = text.split("_")[1:]
    formatted_sub_parts = [sub_part.capitalize() for sub_part in sub_parts]
    return " ".join(formatted_sub_parts)


def get_demographic_category(demographic, gt_type="average"):
    if demographic == f"{gt_type}_overall":
        return "overall"
    elif demographic.startswith(f"{gt_type}_"):
        parts = demographic.split("_")[1:]
        if len(parts) == 1:
            return "gender" if parts[0] in ["man", "woman"] else "race"
    else:
        return "other"


def get_sort_key(demographic):
    category_order = {
        "overall": 0,
        "gender": 1,
        "race": 2,
        "other": 4,
    }
    gender_order = {"Man": 0, "Woman": 1}
    race_order = {"Asian": 0, "Black": 1, "Hispanic": 2, "White": 3}

    category = get_demographic_category(
        "average_" + "_".join(demographic.lower().split())
    )
    parts = demographic.split()

    if category == "overall":
        return (category_order[category], 0)
    elif category == "gender":
        return (category_order[category], gender_order.get(demographic, 2))
    elif category == "race":
        return (category_order[category], race_order.get(demographic, 3))
    else:
        return (category_order[category], 4)


def sort_correlation_results(df):
    df["sort_key"] = df.index.map(get_sort_key)
    sorted_df = df.sort_values("sort_key")
    return sorted_df.drop("sort_key", axis=1)


def calculate_correlation(x, y, valid_labels):
    mask = (x.isin(valid_labels)) & (y.isin(valid_labels))
    x_valid, y_valid = x[mask], y[mask]

    if len(x_valid) < 2:
        return np.nan

    correlation, p_value = stats.pearsonr(x_valid, y_valid)
    return correlation


def compute_corrs(
    data, valid_labels, demographic_columns, model_col, gt_type="average"
):
    corrs = pd.DataFrame(index=demographic_columns, columns=[model_col])

    for demographic in demographic_columns:
        x = data[demographic]
        y = data[model_col]
        corrs.loc[demographic, model_col] = calculate_correlation(x, y, valid_labels)

    corrs.index = corrs.index.map(format_demographic)

    corrs["demographic_cat"] = corrs.index.map(
        lambda x: get_demographic_category(f"{gt_type}_" + "_".join(x.lower().split()))
    )

    corrs = sort_correlation_results(corrs)

    return corrs


def compute_p_values(corrs, shuffled_corrs, model_col, n_resamples=1000):
    demographic_columns = corrs.index
    entries = list()
    for demo in demographic_columns:
        corrs_by_model = pd.concat([d[model_col] for d in shuffled_corrs], axis=1)
        entries.append(
            (
                demo,
                model_col,
                (corrs_by_model.loc[demo] >= corrs.loc[demo, model_col]).sum(),
            )
        )

    results = pd.DataFrame(entries, columns=["sociodemographic", "model", "n_ge"])
    results["frac_larger"] = results.n_ge / n_resamples
    print(
        f"permutations with correlation greater than or equal to the observed correlation: {results.n_ge.sum()}"
    )
    return results


def permutate(ground_truths, demographic_columns):
    permutated = ground_truths.copy()
    for col in demographic_columns:
        permutated[col] = permutated[col].sample(frac=1).reset_index(drop=True)
    return permutated


def permutation_test(
    data, valid_labels, demographic_columns, model_col, n_resamples=1000
):
    corrs = compute_corrs(data, valid_labels, demographic_columns, model_col=model_col)
    shuffled_corrs = [
        compute_corrs(
            permutate(data, demographic_columns),
            valid_labels,
            demographic_columns,
            model_col=model_col,
        )
        for _ in range(n_resamples)
    ]
    p_values = compute_p_values(corrs, shuffled_corrs, model_col=model_col)
    return corrs, p_values, shuffled_corrs


def permutation_per_dataset(
    model, dataset_name, data_dir, valid_labels, gt_type="average"
):
    _df = pd.read_csv(data_dir / f"{dataset_name}_human_all_models.csv")
    demographic_columns = [col for col in _df.columns if col.startswith(f"{gt_type}_")]

    _corrs, _p_values, _shuffled_corrs = permutation_test(
        _df, valid_labels, demographic_columns, model_col=f"{model}_cot"
    )
    return _corrs, _p_values, _shuffled_corrs


def run_permutation(data_dir, output_dir, gt_type="average"):
    n_resamples = 1000
    models = ["gpt", "gemini", "solar"]
    datasets = ["awa", "popq", "nlpos", "sbic", "mhsc"]

    valid_labels = {
        "awa": [1, 2, 3, 4, 5],
        "popq": [1, 2, 3, 4, 5],
        "nlpos": [1, 2, 3],
        "sbic": [1, 2, 3],
        "mhsc": [1, 2, 3],
    }

    permutation_results = {}

    for model in models:
        if model not in permutation_results:
            permutation_results[model] = {}

        for dataset in datasets:
            print(f"Running {model} on {dataset}")
            corrs, _, shuffled_corrs = permutation_per_dataset(
                model,
                dataset,
                data_dir,
                valid_labels[dataset],
                gt_type,
            )
            permutation_results[model][dataset] = {
                "corrs": corrs,
                "shuffled_corrs": shuffled_corrs,
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "permutation_results.pkl", "wb") as f:
        pickle.dump(permutation_results, f)
    print("Results saved to", output_dir / "permutation_results.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    run_permutation(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
