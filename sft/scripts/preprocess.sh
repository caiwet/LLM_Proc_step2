# Process training set
ROOT='/data/bwh-comppath-img2/MGH_CID/LLM_Proc/step2_gemini_output'
python preprocess_dataset.py \
    --input-path $ROOT/train.jsonl \
    --output-path $ROOT/train_tokenized_qwen3_4b.parquet \
    --model-name /data/bwh-comppath-img2/MGH_CID/hf_cache/Qwen3-4B-Instruct-2507  \
    --max-seq-length 16384

# Process eval set
python preprocess_dataset.py \
    --input-path $ROOT/eval.jsonl \
    --output-path $ROOT/eval_tokenized_qwen3_4b.parquet \
    --model-name /data/bwh-comppath-img2/MGH_CID/hf_cache/Qwen3-4B-Instruct-2507  \
    --max-seq-length 16384