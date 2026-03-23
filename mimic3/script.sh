
python process_mimic3.py \
    --model /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/medgemma1.5-4b-instruct_bs8_lr3e-5 \
    --input-pattern "/data/bwh-comppath-img2/MGH_CID/LLM_Proc/results_mimic3/shards/*.parquet" \
    --output-dir /data/bwh-comppath-img2/MGH_CID/LLM_Proc/results_mimic3/medgemma1.5-4b-instruct_bs8_lr3e-5 \
    --host 127.0.0.1 \
    --num-servers 1 \
    --max-concurrent 512 \
    --max-parallel-files 1 \
    --unique-notes-only

python process_mimic3.py \
    --model /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/qwen3-4b-instruct_bs8_lr3e-5 \
    --input-pattern "/data/bwh-comppath-img2/MGH_CID/LLM_Proc/results_mimic3/shards/*.parquet" \
    --output-dir /data/bwh-comppath-img2/MGH_CID/LLM_Proc/results_mimic3/qwen3-4b-instruct_bs8_lr3e-5_date_instruct \
    --host 127.0.0.1 \
    --num-servers 8 \
    --max-concurrent 512 \
    --max-parallel-files 1 \
    --unique-notes-only



