DATA_PATH=../../data/joint_final/
DATASETS="awa mhsc nlpos popq sbic"
MODELS="gpt gemini solar"
OUTPUT_DIR="../../data/alignment/average"

python3 ../alignment/robust_human_vs_model.py \
--models $MODELS \
--dataset_names $DATASETS \
--data_dir $DATA_PATH \
--output_dir $OUTPUT_DIR