python3 ../gemini_inference.py \
    --data_dir ../../../data/human/processed/ \
    --output_dir ../../../data/inference/mhsc \
    --prompt_type cot \
    --model_name gemini-1.5-flash-002 \
    --dataset_name mhsc \
    --log_dir ../../../logs \
    --config_file ../../../config/gemini.json

if [ $? -eq 0 ]; then
    echo "gemini_inference.py ran successfully"
else
    echo "gemini_inference.py failed"
    exit 1
fi