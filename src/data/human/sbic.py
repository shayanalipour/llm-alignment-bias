import os
import pandas as pd
from pathlib import Path
import sanity_check
import datasets

VALID_RACE = ["white", "black", "asian", "hisp"]
VALID_GENDER = ["man", "woman"]
LABEL_MAP = {"0.0": 1, "0.5": 2, "1.0": 3}
TARGET_COLS = ["post", "offensiveYN", "WorkerId", "annotatorRace", "annotatorGender"]


def get_sbic(dataset_name="sbic"):
    dataset = datasets.load_dataset("social_bias_frames")
    df = datasets.concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    ).to_pandas()

    df = df[df["offensiveYN"] != ""]
    df = df[TARGET_COLS].copy()

    new_col_names = {
        "post": "post",
        "offensiveYN": "worker_label",
        "WorkerId": "worker_id",
        "annotatorRace": "race",
        "annotatorGender": "gender",
    }

    df.rename(columns=new_col_names, inplace=True)
    print(df.columns)
    df.dropna(inplace=True)

    df["gender"] = df["gender"].str.lower()
    df["race"] = df["race"].replace({"hisp": "hispanic"}).str.lower()
    df = df[df["race"].isin(VALID_RACE) & df["gender"].isin(VALID_GENDER)]

    df["worker_label"] = df["worker_label"].map(LABEL_MAP)

    post2id = dict(
        (post, f"{dataset_name}_{idx}") for idx, post in enumerate(df["post"].unique())
    )
    df["post_id"] = df["post"].map(post2id)
    worker2id = dict(
        (worker, f"{dataset_name}_{idx}")
        for idx, worker in enumerate(df["worker_id"].unique())
    )
    df["worker_id"] = df["worker_id"].map(worker2id)
    df["dataset"] = dataset_name

    df = sanity_check.check_valid_labels(df, max_label=3)
    df = sanity_check.drop_duplicate_annotations(df)
    df = sanity_check.check_worker_consistency(df)

    assert df.isnull().sum().sum() == 0, "Null values found in the dataset"
    assert (
        df[["post_id", "worker_id"]].duplicated().sum() == 0
    ), "Duplicate annotations detected"

    save_path = Path("../../data/human/processed/sbic.csv")
    os.makedirs(save_path.parent, exist_ok=True)
    df.to_csv(save_path, index=False)

    return df
