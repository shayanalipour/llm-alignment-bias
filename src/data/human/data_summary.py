import pandas as pd
from pathlib import Path


def generate_demographic_summary(dataframes: dict, output_dir: Path) -> None:
    demographic_summary = []

    for csv_name, df in dataframes.items():

        total_count = len(df)

        gender_summary = df["gender"].value_counts().reset_index()
        gender_summary.columns = ["demographic", "count"]
        gender_summary["percentage"] = gender_summary["count"] / total_count * 100
        gender_summary["dataset"] = csv_name
        gender_summary["demographic_type"] = "gender"

        race_summary = df["race"].value_counts().reset_index()
        race_summary.columns = ["demographic", "count"]
        race_summary["percentage"] = race_summary["count"] / total_count * 100
        race_summary["dataset"] = csv_name
        race_summary["demographic_type"] = "race"

        demographic_summary.extend([gender_summary, race_summary])

    demographic_df = pd.concat(demographic_summary, ignore_index=True)

    demographic_df = demographic_df[
        ["dataset", "demographic", "demographic_type", "count", "percentage"]
    ]

    output_path = output_dir / "demographic_summary.csv"
    demographic_df.to_csv(output_path, index=False)
    print(f"Demographic summary saved to {output_path}")


def generate_label_distribution(dataframes: dict, output_dir: Path) -> None:
    label_distributions = []

    for csv_name, df in dataframes.items():
        total_count = len(df)
        label_distribution = df["worker_label"].value_counts().reset_index()
        label_distribution.columns = ["label", "count"]
        label_distribution["percentage"] = (
            label_distribution["count"] / total_count * 100
        )
        label_distribution["dataset"] = csv_name
        label_distributions.append(label_distribution)

    label_distribution_df = pd.concat(label_distributions, ignore_index=True)

    output_path = output_dir / "label_distribution.csv"
    label_distribution_df.to_csv(output_path, index=False)
    print(f"Label distribution summary saved to {output_path}")


def generate_dataset_info(dataframes: dict, output_dir: Path) -> None:
    dataset_info = []

    for csv_name, df in dataframes.items():
        info = {
            "dataset": csv_name,
            "unique_posts": df["post"].nunique(),
            "unique_annotators": df["worker_id"].nunique(),
            "avg_annotations_per_post": round(df.groupby("post").size().mean(), 2),
            "max_annotations_per_post": df.groupby("post").size().max(),
            "min_annotations_per_post": df.groupby("post").size().min(),
            "avg_posts_per_annotator": round(df.groupby("worker_id").size().mean(), 2),
            "max_posts_per_annotator": df.groupby("worker_id").size().max(),
            "min_posts_per_annotator": df.groupby("worker_id").size().min(),
            "total_annotations": len(df),
        }
        dataset_info.append(info)

    dataset_info_df = pd.DataFrame(dataset_info)

    output_path = output_dir / "dataset_info.csv"
    dataset_info_df.to_csv(output_path, index=False)
    print(f"Dataset info summary saved to {output_path}")


def get_summary(dataframes: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_demographic_summary(dataframes, output_dir)
    generate_label_distribution(dataframes, output_dir)
    generate_dataset_info(dataframes, output_dir)
