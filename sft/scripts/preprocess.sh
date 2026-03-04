# Process training set
ROOT='/data/bwh-comppath-img2/MGH_CID/LLM_Proc/step2_gemini_output'
python preprocess_dataset.py \
    --input-path $ROOT/train.jsonl \
    --output-path $ROOT/train_tokenized_lfm.parquet \
    --model-name LiquidAI/LFM2.5-1.2B-Instruct \
    --max-seq-length 16384

# Process eval set
python preprocess_dataset.py \
    --input-path $ROOT/eval.jsonl \
    --output-path $ROOT/eval_tokenized_lfm.parquet \
    --model-name LiquidAI/LFM2.5-1.2B-Instruct \
    --max-seq-length 16384