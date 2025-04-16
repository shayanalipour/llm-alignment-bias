import pandas as pd


def check_valid_labels(df, max_label):
    df["worker_label"] = pd.to_numeric(df["worker_label"], errors="coerce")
    df = df.dropna(subset=["worker_label"])

    df = df[df["worker_label"].apply(lambda x: 1 <= x <= max_label)]
    df["worker_label"] = df["worker_label"].astype(int)

    return df


def drop_duplicate_annotations(df):
    initial_count = len(df)
    df_unique = df.drop_duplicates(subset=["post_id", "worker_id"])
    removed_count = initial_count - len(df_unique)

    if removed_count > 0:
        print(f"Removed {removed_count} duplicate annotations.")

    return df_unique


def check_worker_consistency(df):
    # check if gender and race are consistent
    worker_consistency = df.groupby("worker_id").agg(
        {"gender": "nunique", "race": "nunique"}
    )

    consistent_workers = worker_consistency[
        (worker_consistency["gender"] == 1) & (worker_consistency["race"] == 1)
    ].index

    df_consistent = df[df["worker_id"].isin(consistent_workers)]
    removed_count = len(df) - len(df_consistent)

    if removed_count > 0:
        print(f"Removed {removed_count} rows with inconsistent worker data.")

    return df_consistent
