DATA_PATH=../../data/human/processed/
DATASETS="awa mhsc nlpos popq sbic"

python3 ../data/human/process_human_data.py \
    --data_path $DATA_PATH \
    --datasets $DATASETS

if [ $? -eq 0 ]; then
    echo "process_human_data.py ran successfully"
else
    echo "process_human_data.py failed"
    exit 1
fi
