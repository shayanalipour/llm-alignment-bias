DATASETS=("awa" "mhsc" "nlpos" "popq" "sbic")
LABEL_LEVELS=("5" "3" "3" "5" "3")
MODELS=("gpt" "gemini" "solar")


for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    LABEL_LEVEL="${LABEL_LEVELS[$i]}"
    
    for MODEL in "${MODELS[@]}"; do
        echo "Running process_model_data.py for dataset: $DATASET and model: $MODEL"
        python3 ../data/models/process_model_data.py \
        --input_dir "../../data/model/raw/${DATASET}" \
        --model_name $MODEL \
        --dataset_name $DATASET \
        --prompt_type "cot" \
        --output_file "../../data/model/processed/${DATASET}_${MODEL}.csv" \
        --label_levels "${LABEL_LEVEL}" \
        --log_dir "../../logs/model_data/${DATASET}"

        sleep 1

    done
done