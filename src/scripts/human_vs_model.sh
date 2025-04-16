DATA_PATH=../../data/joint_final/
DATASETS="awa mhsc nlpos popq sbic"
MODELS="gpt gemini solar"
METHOD="average"
OUTPUT_DIR="../../data/alignment/average"


python3 ../alignment/human_vs_model.py \
--models $MODELS \
--dataset_names $DATASETS \
--data_dir $DATA_PATH \
--method $METHOD \
--output_dir $OUTPUT_DIR


