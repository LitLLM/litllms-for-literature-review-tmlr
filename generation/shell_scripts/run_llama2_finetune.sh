# Also see: https://github.com/shubhamagarwal92/mmd/blob/master/train_and_translate.sh
# Current time
CURRENT_TIME=$(date +"%T")
echo "Current time : $CURRENT_TIME"
export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR
export RESULTS_BASE_DIR=/results/auto_review/multixscience/

export MODEL_BASE_NAME="Llama-2-7b-chat-hf"
export MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
# export MODEL_NAME="meta-llama/Llama-2-7b-hf"
# export MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
# export MODEL_NAME="meta-llama/Llama-2-13b-hf"
# export MODEL_NAME="meta-llama/Llama-2-70b-chat-hf"
# export MODEL_NAME="meta-llama/Llama-2-70b-hf"
export VERSION=1

export MODEL_OUT_DIR="${RESULTS_BASE_DIR}"/"${MODEL_BASE_NAME}"/

mkdir -p "${MODEL_OUT_DIR}"
echo "Results are saved in directory: ${MODEL_OUT_DIR}"
PYTHONPATH=. python autoreview/models/llama2_finetune.py # > $MODEL_OUT_DIR/logs.txt
# PYTHONPATH=. python autoreview/models/llamav2_finetune_flash.py # > $MODEL_OUT_DIR/logs.txt