python3 ../gpt_inference.py \
    --data_dir ../../../data/human/processed/ \
    --output_dir ../../../data/inference/nlpos \
    --prompt_type cot \
    --model_name gpt-4o-mini-2024-07-18 \
    --dataset_name nlpos \
    --log_dir ../../../logs \
    --api_key_file ../../../config/gpt_key.json \
    --prompts_file ../prompts.json

if [ $? -eq 0 ]; then
    echo "gpt_inference.py ran successfully"
else
    echo "gpt_inference.py failed"
    exit 1
fi