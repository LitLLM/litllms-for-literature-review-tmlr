#!/bin/bash
# This script submits FAISS index creation jobs in batches.
# 
# Parameters:
# - num_jobs: Total number of jobs.
# - total_files: Total files to process.
# - input_dir: Directory with input embedding files.
# - output_dir_base: Base directory for output FAISS indexes.
# - dimension: Dimensionality of the embeddings.
# 
# For each job, the script computes the file range, creates an output subdirectory, 
# and submits a job to create FAISS indexes for the specified file range.

num_jobs=10
total_files=908
files_per_job=$((total_files / num_jobs))
input_dir="/path/to/input/dir"
output_dir_base="/path/to/output/dir"
dimension=768

for ((i=0; i<num_jobs; i++)); do
    batch_start=$((i * files_per_job))
    
    if [ $i -eq $((num_jobs - 1)) ]; then
        batch_end=$total_files
    else
        batch_end=$((batch_start + files_per_job))
    fi

    # Create subdirectory for each job's file range
    output_dir="${output_dir_base}/${batch_start}-${batch_end}"
    mkdir -p "$output_dir"

    # Generate the list of files for this specific batch
    input_files=()
    for ((j=batch_start; j<batch_end; j++)); do
        input_files+=("${input_dir}/embeddings-specter_v2-part${j}.jsonl.gz")
    done

    echo "Submitting job for files from $batch_start to $batch_end, saving to $output_dir"

    python /mnt/home/lit-rank/data/faiss_gpu/main.py \
        --action index \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        --num_files $((batch_end - batch_start)) \
        --dimension $dimension \
        --file_range "${input_files[@]}"
done
