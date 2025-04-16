DATA_PATH=../../data/human/processed/
DATASETS="awa mhsc nlpos popq sbic"
LABEL_LEVELS="5 3 3 5 3"
METHOD="average"
OUTPUT_FILE="../../data/alignment/average/human_vs_human.csv"


python3 ../alignment/human_vs_human.py \
--data_path $DATA_PATH \
--datasets $DATASETS \
--max_labels $LABEL_LEVELS \
--method $METHOD \
--output_file $OUTPUT_FILE


