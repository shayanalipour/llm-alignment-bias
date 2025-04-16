DATASETS=("awa" "mhsc" "nlpos" "popq" "sbic")

for DATASET in "${DATASETS[@]}"
do
    echo "Running ground_truth.py for dataset: $DATASET"

    python3 ../data/human/ground_truth.py \
        --input_file "../../data/human/processed/${DATASET}.csv" \
        --dataset_name "${DATASET}" \
        --output_file "../../data/human/ground_truth/${DATASET}_gt.csv" \
        --log_dir "../../logs/human_gt" \

done