import os
import pandas as pd
import awa, mhsc, nlpos, popq, sbic
import data_summary
from pathlib import Path
import argparse


def load_datasets(data_path, datasets):
    datasets_summary_dict = {}

    for dataset in datasets:
        try:
            print(f"Processing {dataset}")
            dataset_file = data_path / f"{dataset}.csv"

            if dataset_file.exists():
                print(f"Loading {dataset}.csv")
                df = pd.read_csv(dataset_file)
            else:
                print(f"Processing raw data for {dataset}")
                if dataset == "awa":
                    df = awa.get_awa()
                elif dataset == "mhsc":
                    df = mhsc.get_mhsc()
                elif dataset == "nlpos":
                    df = nlpos.get_nlpos()
                elif dataset == "popq":
                    df = popq.get_popq()
                elif dataset == "sbic":
                    df = sbic.get_sbic()
                else:
                    raise ValueError(f"Invalid dataset: {dataset}")

                df.to_csv(dataset_file, index=False)

            datasets_summary_dict[dataset] = df
            print(f"Successfully processed {dataset}")

        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            print(f"Skipping {dataset}")

    return datasets_summary_dict


def main():
    parser = argparse.ArgumentParser(description="Process and summarize datasets.")
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["awa", "mhsc", "nlpos", "popq", "sbic"],
    )

    args = parser.parse_args()
    data_path = args.data_path
    datasets = args.datasets

    if not data_path.exists():
        print(f"Creating data directory: {data_path}")
        data_path.mkdir(parents=True, exist_ok=True)

    datasets_summary_dict = load_datasets(data_path, datasets)
    if datasets_summary_dict:
        data_summary.get_summary(datasets_summary_dict, data_path)
        print("Summary generated successfully.")
    else:
        print("No datasets were successfully loaded. Summary generation skipped.")


if __name__ == "__main__":
    main()
