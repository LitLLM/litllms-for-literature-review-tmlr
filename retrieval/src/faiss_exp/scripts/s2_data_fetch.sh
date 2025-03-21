num_jobs=50
total_queries=500
queries_per_job=$((total_queries / num_jobs))

api_key="YOUR_S2_API_KEY" #Put your api key from semantic scholar.
directory_path="/path/to/dir"
output_dir="/path/to/output/dir"

for ((i=0; i<num_jobs; i++)); do
    batch_start=$((i * queries_per_job))
    
    if [ $i -eq $((num_jobs - 1)) ]; then
        batch_end=$((total_queries - 1))
    else
        batch_end=$((batch_start + queries_per_job - 1))
    fi

    output_file="${output_dir}/aggregated_paper_data_part${i}.csv"

    echo "Submitting job for queries from $batch_start to $batch_end"

    python /mnt/home/lit-rank/data/faiss_gpu/s2_data.py \
        --batch_start $batch_start \
        --batch_end $batch_end \
        --api_key $api_key \
        --directory_path "$directory_path" \
        --output_file "$output_file"
done
