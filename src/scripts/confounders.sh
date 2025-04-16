MODEL_DATA_PATH="../../data/model/processed"
HUMAN_DATA_PATH="../../data/human/processed"
OUTPUT_PATH="../../data/confounders/average"
DATASETS=("popq" "awa" "nlpos" "sbic" "mhsc")
MODEL="gpt"
AGGREGATE="average"

python3 ../confounders/confounders.py \
    --human_data_path "$HUMAN_DATA_PATH" \
    --model_data_path "$MODEL_DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --dataset_names "${DATASETS[@]}" \
    --model_name "$MODEL" \
    --aggregation_method "$AGGREGATE"