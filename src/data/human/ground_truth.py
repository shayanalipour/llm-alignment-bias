import numpy as np
import pandas as pd
from collections import Counter
import argparse
import logging
from pathlib import Path
from datetime import datetime


class GroundTruth:
    def __init__(self, input_file, dataset_name, log_dir):
        self.input_file = Path(input_file)
        self.dataset_name = dataset_name
        self.log_dir = Path(log_dir)
        self.logger = self.setup_logging()

    def setup_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = (
            self.log_dir
            / f"ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        return logging.getLogger("GroundTruth")

    def load_data(self):
        self.logger.info(f"Loading data from {self.input_file}")
        df = pd.read_csv(self.input_file)
        self.logger.info(f"Loaded {len(df)} rows of data")
        return df

    @staticmethod
    def detect_majority(numbers):
        count = Counter(numbers)
        most_common = count.most_common()

        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            return most_common[0][0]
        else:
            return -1  # tie

    def compute_ground_truth(self, df):
        self.logger.info("Computing ground truth")
        methods = ["majority", "average"]
        df_ = df.copy()

        ground_truth_dfs = []

        for method in methods:
            self.logger.info(f"Computing ground truth using {method} method")
            if method == "majority":
                agg_func = lambda x: self.detect_majority(x)
            elif method == "average":
                agg_func = lambda x: round(np.mean(x))
            else:
                raise ValueError("Invalid method. Choose 'majority' or 'average'.")

            by_demo = (
                df_[
                    ["post_id", "post", "worker_label", "race", "gender"]
                ]  # Added "post" here
                .melt(id_vars=["post_id", "post", "worker_label"])  # Added "post" here
                .groupby(["post_id", "post", "value"])  # Added "post" here
                .worker_label.agg(agg_func)
                .reset_index()
            )

            overall = (
                df_[["post_id", "post", "worker_label"]]  # Added "post" here
                .groupby(["post_id", "post"])  # Added "post" here
                .worker_label.agg(agg_func)
                .reset_index()
                .rename(columns={"worker_label": "overall"})
            )

            ground_truth = by_demo.pivot(index=["post_id", "post"], columns="value")[
                "worker_label"
            ]  # Added "post" here
            ground_truth = ground_truth.reset_index()

            ground_truth = ground_truth.merge(
                overall, on=["post_id", "post"]
            )  # Added "post" here

            ground_truth.columns = [
                f"{method}_{col}" if col not in ["post_id", "post"] else col
                for col in ground_truth.columns
            ]

            ground_truth_dfs.append(ground_truth)

        # Merge the results from both methods
        final_ground_truth = ground_truth_dfs[0].merge(
            ground_truth_dfs[1], on=["post_id", "post"]  # Added "post" here
        )

        # Reorder columns to have post_id and post first
        cols = final_ground_truth.columns.tolist()
        cols = ["post_id", "post"] + [
            col for col in cols if col not in ["post_id", "post"]
        ]
        final_ground_truth = final_ground_truth[cols]

        self.logger.info("Ground truth computation completed")
        return final_ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--log_dir", type=Path)
    args = parser.parse_args()

    computer = GroundTruth(args.input_file, args.dataset_name, args.log_dir)
    df = computer.load_data()
    result = computer.compute_ground_truth(df)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_file, index=False)
    computer.logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
