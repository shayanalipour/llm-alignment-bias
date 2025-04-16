import os
from pathlib import Path
import pandas as pd

import sanity_check

TARGET_COLS = ["action", "session_id", "litw", "gender", "ethnicity"]
LABELS_MAP = {-1: 1, 0: 2, 1: 3}
VALID_GENDER = ["man", "woman"]
VALID_RACE = ["asian", "black", "hispanic", "white"]
RACE_MAP = {
    "asian": "asian",
    "black": "black",
    "latino/latina": "hispanic",
    "white": "white",
}


def get_nlpos(dataset_name="nlpos"):

    df = pd.read_csv(
        "../../data/human/raw/nlpositionality_toxicity_processed.csv",
        usecols=TARGET_COLS,
    )

    new_col_names = {
        "action": "post",
        "session_id": "worker_id",
        "litw": "worker_label",
        "ethnicity": "race",
    }
    df.rename(columns=new_col_names, inplace=True)
    df.dropna(inplace=True)

    df["gender"] = df["gender"].str.lower()
    df["race"] = df["race"].map(RACE_MAP).str.lower()
    df = df[df["race"].isin(VALID_RACE) & df["gender"].isin(VALID_GENDER)]

    df["worker_label"] = df["worker_label"].map(LABELS_MAP)

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

    save_path = Path("../../data/human/processed/nlpos.csv")
    os.makedirs(save_path.parent, exist_ok=True)
    df.to_csv(save_path, index=False)

    return df
