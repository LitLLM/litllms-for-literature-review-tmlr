#!/usr/bin/env bash

conda create -n litllm-generation python=3.11 -y
conda activate litllm-generation

pip install -r requirements.txt