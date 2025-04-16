import os
import json
import time
import pandas as pd
import logging
from tqdm import tqdm
import argparse
from pathlib import Path
from datetime import datetime

import google.generativeai as genai


class GoogleGeminiInference:
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        prompt_type: str,
        model_name: str,
        dataset_name: str,
        log_dir: Path,
        config_file: Path,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.prompt_type = prompt_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.log_dir = log_dir
        self.config_file = config_file

        self.logger = self.setup_logging()

        self.batch_size = 100
        self.current_batch = []
        self.processed_posts = set()

        self.system_prompt = json.load(open("../prompts.json"))[
            f"{self.dataset_name}_{self.prompt_type}"
        ]
        self.logger.info(f"System prompt: {self.system_prompt[:50]}...")

        self.setup_gemini_ai()

        self.requests_per_minute = 15
        self.request_interval = 60 / self.requests_per_minute
        self.last_request_time = 0

    def setup_logging(self):
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
        return logging.getLogger("GoogleGeminiInference")

    def setup_gemini_ai(self):
        with open(self.config_file) as f:
            config = json.load(f)
        self.api_key = config["api_key"]
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def load_data(self):
        df = pd.read_csv(os.path.join(self.data_dir, f"{self.dataset_name}.csv"))
        self.logger.info(f"Loaded {len(df)} posts from {self.dataset_name}")
        self.logger.info(f"Number of unique posts: {len(df['post'].unique())}")

        df = df.dropna(subset=["post"])
        self.logger.info(f"shape after dropping NA posts: {df.shape}")
        self.logger.info(f"Number of unique posts: {len(df['post'].unique())}")

        self.processed_posts = self.get_processed_posts()

        df = df[~df["post"].isin(self.processed_posts)]
        self.logger.info(f"shape after removing processed posts: {df.shape}")
        self.logger.info(f"Number of unique posts: {len(df['post'].unique())}")

        return df["post"].unique().tolist()

    def get_processed_posts(self):
        processed_posts = set()

        # check the output directory for any existing batch files
        for file in os.listdir(self.output_dir):
            if file.startswith(
                f"{self.dataset_name}_{self.model_name}_{self.prompt_type}_batch"
            ) and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(self.output_dir, file))
                processed_posts.update(df["post"].tolist())

        self.logger.info(f"Total processed posts: {len(processed_posts)}")
        return processed_posts

    def wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_interval:
            time.sleep(self.request_interval - time_since_last_request)
        self.last_request_time = time.time()

    def run_inference(self, posts):
        for post in tqdm(posts, desc="Google Gemini Inference", unit="posts"):
            try:
                self.wait_for_rate_limit()

                prompt = f"{self.system_prompt}\n\nThe target comment/post is: {post}"

                response = self.model.generate_content(prompt)
                model_output = response.text

                self.current_batch.append(
                    {
                        "post": post,
                        "prompt_type": self.prompt_type,
                        "model_name": self.model_name,
                        "dataset_name": self.dataset_name,
                        "response": model_output,
                    }
                )

                if len(set(i["post"] for i in self.current_batch)) >= self.batch_size:
                    self.save_batch()

            except Exception as e:
                self.logger.error(f"Error processing post: {post}")
                self.logger.error(e)
                continue

        if self.current_batch:
            self.save_batch()

    def save_batch(self):
        if not self.current_batch:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        batch_df = pd.DataFrame(self.current_batch)
        batch_file = os.path.join(
            self.output_dir,
            f"{self.dataset_name}_{self.model_name}_{self.prompt_type}_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        batch_df.to_csv(batch_file, index=False)
        self.logger.info(f"Saved batch to {batch_file}")
        self.current_batch = []

    def run(self):
        self.logger.info(
            f"Running inference for {self.dataset_name} using {self.model_name} on Google Gemini"
        )
        posts = self.load_data()
        self.run_inference(posts)
        self.logger.info("Inference complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--config_file", type=Path, required=True)

    args = parser.parse_args()

    if args.prompt_type not in ["cot", "nocot"]:
        raise ValueError("Invalid prompt type")

    if not os.path.isfile(os.path.join(args.data_dir, f"{args.dataset_name}.csv")):
        raise ValueError("Invalid dataset name")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    inference = GoogleGeminiInference(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        prompt_type=args.prompt_type,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        log_dir=args.log_dir,
        config_file=args.config_file,
    )
    inference.run()


if __name__ == "__main__":
    main()
