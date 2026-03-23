python process_sharded_parquets.py \
    --model /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/medgemma1.5-4b-instruct_bs8_lr3e-5 \
    --input-pattern "/data/bwh-comppath-img/concept_extraction/data/all_patient_notes_multilevel_sharded/0*/*.parquet" \
    --output-dir /data/bwh-comppath-img2/MGH_CID/LLM_Proc/results/medgemma1.5-4b-instruct_bs8_lr3e-5/eval \
    --host 127.0.0.1 \
    --num-servers 1 \
    --max-concurrent 128 \
    --max-parallel-files 1 \
    --unique-notes-only


python process_sharded_parquets.py \
    --model /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/qwen3-4b-instruct_bs8_lr3e-5 \
    --input-pattern "/data/bwh-comppath-img3/concept_extraction/data/all_patient_notes_multilevel_sharded/0*/*.parquet" \
    --output-dir /data/bwh-comppath-img2/MGH_CID/LLM_Proc/results/qwen3-4b-instruct_bs8_lr3e-5/eval \
    --host 127.0.0.1 \
    --num-servers 8 \
    --max-concurrent 256 \
    --max-parallel-files 1 \
    --unique-notes-only