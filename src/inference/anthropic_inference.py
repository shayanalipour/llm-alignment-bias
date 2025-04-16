import os
import json
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime
import time
import anthropic


class AnthropicInference:
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        prompt_type: str,
        model_name: str,
        dataset_name: str,
        log_dir: Path,
        config_file: Path,
        prompts_file: Path,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.prompt_type = prompt_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.log_dir = log_dir
        self.config_file = config_file
        self.prompts_file = prompts_file

        self.logger = self.setup_logging()

        self.batch_size = 500
        self.current_batch = []
        self.processed_posts = set()

        self.load_prompts()
        self.logger.info(f"System prompt: {self.system_prompt[:50]}...")

        self.client = self.setup_client()

        self.rate_limit = 50  # requests per minute
        self.request_times = []

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
        return logging.getLogger("AnthropicInference:")

    def setup_client(self):
        with open(self.config_file, "r") as file:
            config = json.load(file)
        api_key = config["api_key"]
        return anthropic.Anthropic(api_key=api_key)

    def load_prompts(self):
        with open(self.prompts_file, "r") as file:
            prompts = json.load(file)
        self.system_prompt = prompts[f"{self.dataset_name}_{self.prompt_type}"]

    def load_data(self):
        df = pd.read_csv(os.path.join(self.data_dir, f"{self.dataset_name}.csv"))
        self.logger.info(
            f"Loaded {df['post'].nunique()} unique posts from {self.dataset_name}"
        )
        df = df.dropna(subset=["post"])
        self.logger.info(f"shape after dropping NA posts: {df['post'].nunique()}")

        self.processed_posts = self.get_processed_posts()

        df = df[~df["post"].isin(self.processed_posts)]
        self.logger.info(
            f"shape after removing processed posts: {df['post'].nunique()}"
        )

        return df["post"].unique().tolist()

    def get_processed_posts(self):
        processed_posts = set()
        for file in os.listdir(self.output_dir):
            if file.startswith(
                f"{self.dataset_name}_{self.model_name}_{self.prompt_type}_batch"
            ) and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(self.output_dir, file))
                processed_posts.update(df["post"].tolist())
        return processed_posts

    def apply_rate_limit(self):
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.request_times.append(time.time())

    def run_inference(self, posts):
        for post in tqdm(posts, desc="Inference", unit="posts"):
            try:
                self.apply_rate_limit()

                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=200,
                    temperature=0,
                    system=self.system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"The target comment/post is: {post}",
                                }
                            ],
                        }
                    ],
                )
                model_output = message.content[0].text

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
            f"Running inference for {self.dataset_name} using {self.model_name}"
        )
        posts = self.load_data()
        self.run_inference(posts)
        self.logger.info("Inference complete")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--config_file", type=Path, required=True)
    parser.add_argument("--prompts_file", type=Path, required=True)

    args = parser.parse_args()

    if args.prompt_type not in ["cot", "nocot"]:
        raise ValueError("Invalid prompt type")

    if not os.path.isfile(os.path.join(args.data_dir, f"{args.dataset_name}.csv")):
        raise ValueError("Invalid dataset name")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    inference = AnthropicInference(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        prompt_type=args.prompt_type,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        log_dir=args.log_dir,
        config_file=args.config_file,
        prompts_file=args.prompts_file,
    )
    inference.run()


if __name__ == "__main__":
    main()
