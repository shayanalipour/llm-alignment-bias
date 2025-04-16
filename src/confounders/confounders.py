import pandas as pd
import scipy as sp
import os
import numpy as np
import argparse

from pathlib import Path
from collections import Counter
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer


class Confounders:
    def __init__(
        self,
        human_data_path,
        model_data_path,
        output_path,
        dataset_names,
        model_name,
        aggregation_method,
    ):
        self.human_data_path = human_data_path
        self.model_data_path = model_data_path
        self.output_path = output_path
        self.dataset_names = dataset_names
        self.model_name = model_name
        self.aggregation_method = aggregation_method

    def prepare_data(self):

        if not os.path.exists(
            self.output_path / f"{self.model_name}_raw_confounders.csv"
        ):
            print("preparing raw data for confounders calculation")
            human_dfs = []
            model_dfs = []

            for dataset in self.dataset_names:
                human_df = pd.read_csv(self.human_data_path / f"{dataset}.csv")
                print(f"dataset: {dataset} shape: {human_df.shape}")
                human_dfs.append(human_df)

                model_df = pd.read_csv(
                    self.model_data_path / f"{dataset}_{self.model_name}.csv"
                )
                model_df["dataset"] = dataset
                model_df["model"] = self.model_name
                model_dfs.append(model_df)

            human_df = pd.concat(human_dfs)
            model_df = pd.concat(model_dfs)

            data = pd.merge(human_df, model_df, on=["post", "dataset"], how="left")
            data = data[
                [
                    "post_id",
                    "worker_id",
                    "gender",
                    "race",
                    "worker_label",
                    f"{self.model_name}_cot",
                    "dataset",
                    "model",
                ]
            ]
            self.output_path.mkdir(parents=True, exist_ok=True)
            data.to_csv(
                self.output_path / f"{self.model_name}_raw_confounders.csv", index=False
            )
            return data
        else:
            print("loading raw data for confounders calculation")
            return pd.read_csv(
                self.output_path / f"{self.model_name}_raw_confounders.csv"
            )

    def calculate_ground_truth(self):
        self.data[f"{self.model_name}_hum_diff"] = self.data.apply(
            lambda row: row[f"{self.model_name}_cot"] - row.worker_label, axis=1
        )
        self.data[f"{self.model_name}_hum_abs_diff"] = self.data[
            f"{self.model_name}_hum_diff"
        ].apply(abs)
        self.data[f"{self.model_name}_hum_bool_diff"] = self.data[
            f"{self.model_name}_hum_diff"
        ].apply(lambda x: x != 0)

    def compute_worker_sensitivity(self):
        self.data["worker_label_rank"] = self.data.groupby(
            "post_id"
        ).worker_label.rank()

    def compute_post_difficulty(self):
        doc_entropies = dict()

        for (dataset, post_id), ratings in self.data.groupby(
            ["dataset", "post_id"]
        ).worker_label:
            if post_id in doc_entropies:
                continue

            dataset_max_labels = dict(self.data.groupby("dataset").worker_label.max())
            reference_dist = (
                np.ones(int(dataset_max_labels[dataset])) / dataset_max_labels[dataset]
            )
            empirical_dist = np.zeros(int(dataset_max_labels[dataset]))
            for i, j in Counter(int(i) for i in ratings).items():
                empirical_dist[i - 1] = j
            empirical_dist /= empirical_dist.sum()
            doc_entropies[post_id] = sp.stats.entropy(empirical_dist, reference_dist)

        self.data["post_label_entropy"] = -self.data.post_id.map(doc_entropies)

    def compute_agreement(self):

        if self.aggregation_method == "majority":
            self.data["post_gender_mode_gt"] = self.data.groupby(
                ["post_id", "gender"]
            ).worker_label.transform(lambda x: pd.Series.mode(x)[0])
            self.data["worker_label_gender_offset"] = (
                self.data.worker_label - self.data.post_gender_mode_gt
            )

            self.data["post_race_mode_gt"] = self.data.groupby(
                ["post_id", "race"]
            ).worker_label.transform(lambda x: pd.Series.mode(x)[0])
            self.data["worker_label_race_offset"] = (
                self.data.worker_label - self.data.post_race_mode_gt
            )

        elif self.aggregation_method == "average":
            self.data["post_gender_mean_gt"] = self.data.groupby(
                ["post_id", "gender"]
            ).worker_label.transform("mean")
            self.data["worker_label_gender_offset"] = (
                self.data.worker_label - self.data.post_gender_mean_gt
            )

            self.data["post_race_mean_gt"] = self.data.groupby(
                ["post_id", "race"]
            ).worker_label.transform("mean")
            self.data["worker_label_race_offset"] = (
                self.data.worker_label - self.data.post_race_mean_gt
            )

    def save_computed_confounders(self):
        final_df = self.data.copy()
        final_df["alignment"] = (
            final_df.worker_label == final_df[f"{self.model_name}_cot"]
        ).astype(int)

        final_df["gender_disagreement"] = final_df.worker_label_gender_offset.abs()
        final_df["ethnicity_disagreement"] = final_df.worker_label_race_offset.abs()

        final_df["gender_agreement"] = -final_df["gender_disagreement"]
        final_df["ethnicity_agreement"] = -final_df["ethnicity_disagreement"]

        final_df = final_df.rename(
            columns={
                "post_label_entropy": "difficulty",
                "worker_label_rank": "sensitivity",
                "worker_label": "label",
                "race": "ethnicity",
            }
        )

        final_df.to_csv(
            self.output_path / f"{self.model_name}_computed_confounders.csv",
            index=False,
        )
        return final_df

    def confounder_logistic_regression(self):

        if not os.path.exists(
            self.output_path / f"{self.model_name}_computed_confounders.csv"
        ):
            print("computing confounders")
            self.data = self.prepare_data()
            self.calculate_ground_truth()
            self.compute_worker_sensitivity()
            self.compute_post_difficulty()
            self.compute_agreement()
            data = self.save_computed_confounders()
        else:
            print("loading computed confounders")
            data = pd.read_csv(
                self.output_path / f"{self.model_name}_computed_confounders.csv"
            )

        ### standardize the data ###
        for col in [
            "label",
            "difficulty",
            "sensitivity",
            "gender_agreement",
            "ethnicity_agreement",
        ]:
            data[col] = data.groupby("dataset")[col].transform(
                lambda group: (group - group.mean()) / (2 * group.std())
            )

        data = pd.merge(
            data,
            pd.get_dummies(
                data.dataset,
                prefix="dataset",
            ).astype(int),
            left_index=True,
            right_index=True,
        )

        for dataset in data.dataset.unique():
            print(dataset)
            data[f"dataset_{dataset}"] -= data[f"dataset_{dataset}"].mean()

        data = pd.merge(
            data,
            pd.get_dummies(
                data.gender,
                prefix="gender",
            ).astype(int),
            left_index=True,
            right_index=True,
        )

        for gender in data.gender.unique():
            data[f"gender_{gender}"] -= data[f"gender_{gender}"].mean()

        data = pd.merge(
            data,
            pd.get_dummies(
                data.ethnicity,
                prefix="ethnicity",
            ).astype(int),
            left_index=True,
            right_index=True,
        )
        for ethnicity in data.ethnicity.unique():
            data[f"ethnicity_{ethnicity}"] -= data[f"ethnicity_{ethnicity}"].mean()

        # demographic model
        demographic_mod = smf.logit(
            formula=f"alignment ~  ethnicity_black+ethnicity_hispanic+ethnicity_asian+gender_woman  + dataset_popq+ dataset_nlpos+ dataset_sbic+ dataset_mhsc",
            data=data,
        )
        demographic_mod_res = demographic_mod.fit()
        print(f"Demographic model: \n {demographic_mod_res.summary()}")

        # confounder model
        confounder_mod = smf.logit(
            formula="alignment ~ label + difficulty + sensitivity + gender_agreement + ethnicity_agreement + ethnicity_black+ethnicity_hispanic+ethnicity_asian+gender_woman+ dataset_popq+ dataset_nlpos+ dataset_sbic+ dataset_mhsc",
            data=data,
        )
        confounder_mod_res = confounder_mod.fit()
        print(f"Confounder model: \n {confounder_mod_res.summary()}")

        stargazer = Stargazer([demographic_mod_res, confounder_mod_res])
        stargazer.significance_levels([0.05, 0.01, 0.001])
        with open(
            self.output_path / f"{self.model_name}_confounder_results.tex", "w"
        ) as f:
            f.write(stargazer.render_latex())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--human_data_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model_data_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--aggregation_method",
        type=str,
        default="average",
    )
    args = parser.parse_args()

    confounders = Confounders(
        args.human_data_path,
        args.model_data_path,
        args.output_path,
        args.dataset_names,
        args.model_name,
        args.aggregation_method,
    )
    confounders.confounder_logistic_regression()


if __name__ == "__main__":
    main()
