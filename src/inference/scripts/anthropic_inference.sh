python3 ../anthropic_inference.py \
    --data_dir ../../../data/human/processed \
    --output_dir ../../../data/inference/popq \
    --prompt_type cot \
    --model_name claude-3-5-sonnet-20240620 \
    --dataset_name popq \
    --log_dir ../../../logs \
    --config_file ../../../config/anthropic.json \
    --prompts_file ../prompts.json

if [ $? -eq 0 ]; then
    echo "anthropic_inference.py ran successfully"
else
    echo "anthropic_inference.py failed"
    exit 1
fi