# Also see: https://github.com/shubhamagarwal92/mmd/blob/master/train_and_translate.sh
# Current time
CURRENT_TIME=$(date +"%T")
echo "Current time : $CURRENT_TIME"
export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR
export RESULTS_BASE_DIR=results/auto_review/multixscience/

export MODEL_NAME="gpt-3.5-turbo"

# export MODEL_OUT_DIR="${RESULTS_BASE_DIR}"/"${MODEL_NAME}"/"${VERSION}"
export MODEL_OUT_DIR="${RESULTS_BASE_DIR}"/"${MODEL_NAME}"/
mkdir -p "${MODEL_OUT_DIR}"
echo "Results are saved in directory: ${MODEL_OUT_DIR}"

PYTHONPATH=. python autoreview/models/plan_based_generation.py \
-m "gpt-3.5-turbo" \
-d "shubhamagarwal92/rw_2308_filtered" \
-s "MODEL_OUT_DIR" \
-p "plan_template" > $MODEL_OUT_DIR/logs.txt
