export HF_HOME=/data/bwh-comppath-img/concept_extraction/.hf_cache
ROOT='/data/bwh-comppath-img2/MGH_CID/LLM_Proc/step2_gemini_output'
accelerate launch --num_processes=1 train_pretokenized.py \
    --model-name LiquidAI/LFM2.5-1.2B-Instruct \
    --dataset-path $ROOT/train_tokenized_lfm.parquet \
    --eval-dataset-path $ROOT/eval_tokenized_lfm.parquet \
    --output-dir /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/lfm2.5-1.2b-instruct_bs8_lr3e-5 \
    --num-train-epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 2000 \
    --logging_steps 10 \
    --save_steps 5000 \
    --eval_steps 5000 \
    --save_total_limit 3 \
    --report_to tensorboard