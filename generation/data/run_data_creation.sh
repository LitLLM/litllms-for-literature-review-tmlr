# Also see: https://github.com/shubhamagarwal92/mmd/blob/master/train_and_translate.sh
# Current time
CURRENT_TIME=$(date +"%T")
echo "Current time : $CURRENT_TIME"
export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR
export VERSION=1
export DATA_DIR=results/auto_review/dataset/full_dataset/
export SAVE_DATA_DIR=results/auto_review/dataset/new_dataset/

mkdir -p "${DATA_DIR}"
echo "Results are saved in directory: ${DATA_DIR}"

PYTHONPATH=. python data/download_s2orc_full_data.py

# Run as python -m data.filter_s2_data
PYTHONPATH=. python data/filter_s2_data.py # > $MODEL_OUT_DIR/logs.txt

PYTHONPATH=. python data/save_as_hf_dataset.py