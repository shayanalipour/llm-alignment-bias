import os
from pathlib import Path
import datasets

import sanity_check

VALID_GENDER = ["man", "woman"]
VALID_RACE = ["white", "black", "asian", "hispanic"]
LABEL_MAP = {0.0: 1, 1.0: 2, 2.0: 3}
TARGET_COLS = [
    "text",
    "hatespeech",
    "annotator_id",
    "annotator_gender",
    "annotator_race_white",
    "annotator_race_asian",
    "annotator_race_black",
    "annotator_race_latinx",
]


def get_mhsc(dataset_name="mhsc"):

    dataset = datasets.load_dataset("ucberkeley-dlab/measuring-hate-speech")
    df = dataset["train"].to_pandas()
    df = df[TARGET_COLS].copy()

    df["race"] = df.apply(get_race, axis=1)
    df.drop(
        columns=[
            "annotator_race_white",
            "annotator_race_asian",
            "annotator_race_black",
            "annotator_race_latinx",
        ],
        inplace=True,
    )

    new_col_names = {
        "text": "post",
        "hatespeech": "worker_label",
        "annotator_id": "worker_id",
        "annotator_gender": "gender",
    }
    df.rename(columns=new_col_names, inplace=True)

    df.dropna(inplace=True)
    df = df[df["race"] != "unknown"]
    df["gender"] = df["gender"].str.lower().replace({"male": "man", "female": "woman"})
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

    df = drop_outlier(df)

    assert df.isnull().sum().sum() == 0, "Null values found in the dataset"
    assert (
        df[["post_id", "worker_id"]].duplicated().sum() == 0
    ), "Duplicate annotations detected"

    save_path = Path("../../data/human/processed/mhsc.csv")
    os.makedirs(save_path.parent, exist_ok=True)
    df.to_csv(save_path, index=False)

    return df


def get_race(row):
    if row["annotator_race_white"] == 1:
        return "white"
    if row["annotator_race_asian"] == 1:
        return "asian"
    if row["annotator_race_black"] == 1:
        return "black"
    if row["annotator_race_latinx"] == 1:
        return "hispanic"
    return "unknown"


def drop_outlier(df):
    """
    In the MHSC dataset, there are some posts with a large number of annotations (e.g. 100+).
    Less than 0.1% of the posts have more than 5 annotations. There is a big jump from 5 to 3-digit annotations.
    Upon analyzing the distribution of the number of annotations per post, the upper bound was set to 5 annotations per post.
    """
    post_counts = df["post_id"].value_counts()
    outlier_posts = post_counts[post_counts > 5].index
    return df[~df["post_id"].isin(outlier_posts)]
