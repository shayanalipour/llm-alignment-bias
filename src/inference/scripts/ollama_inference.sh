python3 ../ollama_inference.py \
    --data_dir ../../../data/human/processed/ \
    --output_dir ../../../data/inference/ \
    --prompt_type cot \
    --model_name mistral \
    --dataset_name nlpos \
    --log_dir ../../../logs/inference \

if [ $? -eq 0 ]; then
    echo "ollama_inference.py ran successfully"
else
    echo "ollama_inference.py failed"
    exit 1
fi


