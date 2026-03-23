
# conda activate qwen35
cd /data/bwh-comppath-img2/MGH_CID/LLM_Proc/mimic3
# MODEL='medgemma1.5-4b-instruct_bs8_lr3e-5'
MODEL='qwen3-4b-instruct_bs8_lr3e-5'
# python process_mimic3.py \
#     --model /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/$MODEL \
#     --input-pattern "/data/bwh-comppath-img2/MGH_CID/LLM_Proc/benchmark/ehrcon_all_notes.parquet" \
#     --output-dir /data/bwh-comppath-img2/MGH_CID/LLM_Proc/benchmark/$MODEL/init_try \
#     --host 127.0.0.1 \
#     --num-servers 8 \
#     --max-concurrent 256 \
#     --max-parallel-files 1 \
#     --unique-notes-only

python retry_truncate.py \
    --model /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/$MODEL \
    --input-pattern /data/bwh-comppath-img2/MGH_CID/LLM_Proc/benchmark/$MODEL/init_try/ehrcon_all_notes.parquet \
    --output-dir /data/bwh-comppath-img2/MGH_CID/LLM_Proc/benchmark/$MODEL/retry_truncate \
    --host 127.0.0.1 \
    --max-tokens 12288