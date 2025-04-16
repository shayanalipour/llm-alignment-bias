import os
from pathlib import Path
import pandas as pd

import sanity_check

VALID_GENDER = ["man", "woman"]
VALID_RACE = ["asian", "black", "white"]

RACE_MAP = {
    "Black or African American": "black",
    "White": "white",
    "Asian": "asian",
    "Hispanic or Latino": "hispanic",
    "Native American": "native american",
    "Arab American": "arab american",
}

TARGET_COLS = ["text", "user_id", "offensiveness", "race", "gender"]


def get_popq(dataset_name="popq"):

    df = pd.read_csv("../../data/human/raw/popquorn_offensive.csv", usecols=TARGET_COLS)

    new_col_names = {
        "text": "post",
        "user_id": "worker_id",
        "offensiveness": "worker_label",
    }
    df.rename(columns=new_col_names, inplace=True)
    df.dropna(inplace=True)

    df["gender"] = df["gender"].str.lower()
    df["race"] = df["race"].map(RACE_MAP).str.lower()
    df = df[df["race"].isin(VALID_RACE) & df["gender"].isin(VALID_GENDER)]

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

    df = sanity_check.check_valid_labels(df, max_label=5)
    df = sanity_check.drop_duplicate_annotations(df)
    df = sanity_check.check_worker_consistency(df)

    assert df.isnull().sum().sum() == 0, "Null values found in the dataset"
    assert (
        df[["post_id", "worker_id"]].duplicated().sum() == 0
    ), "Duplicate annotations detected"

    save_path = Path("../../data/human/processed/popq.csv")
    os.makedirs(save_path.parent, exist_ok=True)
    df.to_csv(save_path, index=False)

    return df
