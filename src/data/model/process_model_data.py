import pandas as pd
import numpy as np
import argparse
import re
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class ModelResponsesAggregator:
    def __init__(
        self, directory, model_name, dataset_name, prompt_type, label_levels, log_dir
    ):
        self.directory = Path(directory)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.prompt_type = prompt_type

        self.label_levels = label_levels
        self.valid_labels = [str(i) for i in range(1, label_levels + 1)]

        self.log_dir = Path(log_dir)
        self.logger = self.setup_logging()
        self.logger.info(
            f"ModelResponsesAggregator initialized for {model_name} model - {prompt_type} prompt - dir: {directory} - label_levels: {label_levels}"
        )

        self.dataframes = []

    def setup_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        script_name = Path(__file__).stem
        log_file = (
            self.log_dir
            / f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file)],
        )
        return logging.getLogger("ModelResponsesAggregator")

    def extract_final_label(self, text):
        text = str(text)
        text = re.sub(r"\([^)]*\)", "", text)

        if text.strip().split()[0] in self.valid_labels:
            return text.strip().split()[0]

        if re.match(r"\d+\.", text):
            return re.match(r"\d+\.", text).group(0).replace(".", "")

        rating_pattern = rf"(\b\d)\s*(?:out of {self.label_levels})"
        rating_matches = re.findall(rating_pattern, text)
        if rating_matches:
            last_rating = rating_matches[-1]
            if last_rating in self.valid_labels:
                return last_rating

        pattern = r"\b(" + "|".join(self.valid_labels) + r")\b"
        matches = re.findall(pattern, text)
        unique_matches = set(matches)

        if len(unique_matches) == 1:
            return unique_matches.pop()
        else:
            return np.nan

    def load_csv_files(self):
        target_files = list(
            self.directory.glob(f"*{self.model_name}*{self.prompt_type}*.csv")
        )
        self.logger.info(f"Found {len(target_files)} files in {self.directory}")
        for file_path in target_files:
            df = pd.read_csv(file_path, usecols=["post", "response"])
            model_prompt = f"{self.model_name}_{self.prompt_type}"
            df[model_prompt] = df["response"].apply(self.extract_final_label)
            self.dataframes.append(df[["post", model_prompt, "response"]])

    def merge_responses(self):
        if not self.dataframes:
            raise ValueError("No dataframes loaded.")

        merged_df = pd.concat(self.dataframes, axis=0)
        return merged_df

    def handle_duplicates(self, df):
        duplicates = df[df.duplicated(subset=["post"], keep=False)]
        unique_df = df.drop_duplicates(subset=["post"], keep="first")

        model_prompt = f"{self.model_name}_{self.prompt_type}"
        conflicting_labels = defaultdict(set)

        for _, group in duplicates.groupby("post"):
            unique_labels = group[model_prompt].unique()
            if len(unique_labels) > 1:
                conflicting_labels[group["post"].iloc[0]] = set(unique_labels)

        return unique_df, conflicting_labels

    def validate_model_label(self, df):
        df = df.copy()

        initial_rows = len(df)
        column_name = f"{self.model_name}_{self.prompt_type}"

        def convert_and_validate(value):
            try:
                if isinstance(value, str):
                    value = float(value)  # handle cases like '1.0'
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                elif isinstance(value, (float, int)):
                    value = int(value)
                else:
                    return None

                # if valid range
                if 1 <= value <= self.label_levels:
                    return value
                else:
                    return None

            except (ValueError, TypeError):
                return None

        df.loc[:, column_name] = df[column_name].apply(convert_and_validate)

        # log invalid rows
        invalid_rows = df[df[column_name].isnull()]
        if not invalid_rows.empty:
            self.logger.warning(
                f"Found {len(invalid_rows)} invalid rows in {column_name}:"
            )
            for _, row in invalid_rows.iterrows():
                self.logger.warning(f"Invalid row: {row.to_dict()}")

        df = df.dropna(subset=[column_name])
        df.loc[:, column_name] = df[column_name].astype(int)

        dropped_rows = initial_rows - len(df)
        self.logger.info(f"Dropped {dropped_rows} rows. Remaining rows: {len(df)}")

        return df

    def switch_labels(self, df):
        df = df.copy()
        column_name = f"{self.model_name}_{self.prompt_type}"
        if self.dataset_name.lower() in ["sbic", "nlpos", "mhsc"]:
            if self.label_levels != 3:
                self.logger.warning(
                    f"Dataset {self.dataset_name} should have label_levels of 3, but {self.label_levels} was provided."
                )

            self.logger.info(
                f"Switching labels 1 and 3 for dataset: {self.dataset_name}"
            )
            df.loc[:, column_name] = (
                df[column_name].astype(int).replace({1: 3, 3: 1}).astype(int)
            )

        return df

    def report_statistics(self, df, conflicting_labels):
        model_prompt = f"{self.model_name}_{self.prompt_type}"
        total_rows = len(df)
        missing_values = df[model_prompt].isnull().sum()
        valid_labels = df[model_prompt].notna().sum()

        self.logger.info(f"Total unique posts: {total_rows}")
        self.logger.info(f"Posts with valid labels: {valid_labels}")
        self.logger.info(f"Posts with missing labels: {missing_values}")
        self.logger.info(f"Posts with conflicting labels: {len(conflicting_labels)}")

        label_distribution = df[model_prompt].value_counts().sort_index()
        self.logger.info(f"Label distribution:\n{label_distribution}")

        if conflicting_labels:
            self.logger.warning("Posts with conflicting labels:")
            for post, labels in conflicting_labels.items():
                self.logger.warning(f"Post: {post[:50]}... Labels: {labels}")

    def save_merged_responses(self, output_file):
        merged_df = self.merge_responses()
        unique_df, conflicting_labels = self.handle_duplicates(merged_df)
        unique_df = self.validate_model_label(unique_df)
        unique_df = self.switch_labels(unique_df)
        self.report_statistics(unique_df, conflicting_labels)

        output_file = Path(output_file)
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
        unique_df.to_csv(output_file, index=False)
        self.logger.info(f"Saved merged responses to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--label_levels", type=int, choices=[3, 4, 5], default=5)
    parser.add_argument("--log_dir", type=Path, required=True)

    args = parser.parse_args()

    aggregator = ModelResponsesAggregator(
        directory=args.input_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        prompt_type=args.prompt_type,
        label_levels=args.label_levels,
        log_dir=args.log_dir,
    )
    aggregator.load_csv_files()
    aggregator.save_merged_responses(output_file=args.output_file)


if __name__ == "__main__":
    main()
