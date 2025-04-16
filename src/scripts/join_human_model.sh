DATASETS=("popq" "awa" "nlpos" "sbic" "mhsc")
MODELS=("gpt" "gemini" "solar")
MODEL_BASE_PATH="../../data/model/processed"

for DATASET in "${DATASETS[@]}"; do
    HUMAN_GROUND_TRUTH="../../data/human/ground_truth/${DATASET}_gt.csv"
    OUTPUT_FILE="../../data/joint_final/${DATASET}_human_all_models.csv"
    
    echo "Running join_human_model.py for dataset: $DATASET with all models"
    python3 ../data/join_human_model.py \
        --human_ground_truth "$HUMAN_GROUND_TRUTH" \
        --model_base_path "$MODEL_BASE_PATH" \
        --model_names "${MODELS[@]}" \
        --dataset "$DATASET" \
        --output_file "$OUTPUT_FILE"

    sleep 2
done