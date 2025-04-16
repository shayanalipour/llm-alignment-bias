import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import List


class JoinHumanModel:
    def __init__(
        self,
        human_ground_truth: Path,
        model_base_path: Path,
        model_names: List[str],
        dataset: str,
        output_file: Path,
    ):
        self.human_ground_truth = human_ground_truth
        self.model_base_path = model_base_path
        self.model_names = model_names
        self.dataset = dataset
        self.output_file = output_file

    def get_model_paths(self):
        return [
            self.model_base_path / f"{self.dataset}_{model}.csv"
            for model in self.model_names
        ]

    def load_data(self):
        human_ground_truth_df = pd.read_csv(self.human_ground_truth)
        model_dfs = []

        for model_name, model_path in zip(self.model_names, self.get_model_paths()):
            df = pd.read_csv(model_path)
            model_dfs.append(df[["post", f"{model_name}_cot"]])

        return human_ground_truth_df, model_dfs

    def merge_data(self, human_ground_truth_df, model_dfs):
        merged_df = human_ground_truth_df

        for model_df in model_dfs:
            merged_df = pd.merge(merged_df, model_df, on="post", how="outer")
        return merged_df

    def labels_to_int(self, merged_df):
        def safe_convert_to_int(value):
            try:
                return int(value)
            except (ValueError, TypeError):
                return np.nan

        for column in merged_df.columns:
            if column != "post" and column != "post_id":
                merged_df[column] = merged_df[column].apply(safe_convert_to_int)
        return merged_df

    def save_data(self, merged_df):
        merged_df = merged_df.drop(columns=["post"])
        merged_df.to_csv(self.output_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Merge human ground truth and multiple model results."
    )
    parser.add_argument(
        "--human_ground_truth",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model_base_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    merger = JoinHumanModel(
        args.human_ground_truth,
        args.model_base_path,
        args.model_names,
        args.dataset,
        args.output_file,
    )

    human_ground_truth_df, model_dfs = merger.load_data()
    merged_df = merger.merge_data(human_ground_truth_df, model_dfs)
    merged_df = merger.labels_to_int(merged_df)
    merger.save_data(merged_df)


if __name__ == "__main__":
    main()
