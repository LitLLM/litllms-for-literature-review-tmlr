# pip install text-generation

model=meta-llama/Llama-2-70b-chat-hf; 
volume=/home/shubham/cache/data;
echo $HF_TOKEN

docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN -p 80:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest \
--model-id $model --quantize bitsandbytes-nf4 \
--max-total-tokens 22384 --max-input-length 20500 --max-batch-prefill-tokens 20500 --rope-scaling dynamic --rope-factor 2 \
--sharded true --num-shard 4

# --gpus '"device=0,1,2,3"'
# -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.3 \

# This works
# curl 172.17.0.1:80/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' -H 'Content-Type: application/json'