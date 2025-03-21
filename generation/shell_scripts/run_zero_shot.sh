# Also see: https://github.com/shubhamagarwal92/mmd/blob/master/train_and_translate.sh
# Current time
CURRENT_TIME=$(date +"%T")
echo "Current time : $CURRENT_TIME"
export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR
export RESULTS_BASE_DIR=/results/auto_review/multixscience/

export MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
# export MODEL_NAME="meta-llama/Llama-2-7b-hf"
# export MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
# export MODEL_NAME="meta-llama/Llama-2-13b-hf"
# export MODEL_NAME="meta-llama/Llama-2-70b-chat-hf"
# export MODEL_NAME="meta-llama/Llama-2-70b-hf"
# export VERSION=1

# export MODEL_OUT_DIR="${RESULTS_BASE_DIR}"/"${MODEL_NAME}"/"${VERSION}"

# mkdir -p "${MODEL_OUT_DIR}"
# echo "Results are saved in directory: ${MODEL_OUT_DIR}"

# conda activate autoreview

# PYTHONPATH=. python autoreview/models/llamav2_finetune_flash.py # > $MODEL_OUT_DIR/logs.txt
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python autoreview/models/zero_shot_longer_context.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python autoreview/models/zero_shot_longer_context.py \
-s outputs

# https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
