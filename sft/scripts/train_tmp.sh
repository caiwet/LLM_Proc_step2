export HF_HOME=/data/bwh-comppath-img/concept_extraction/.hf_cache
ROOT='/data/bwh-comppath-img2/MGH_CID/LLM_Proc/step2_gemini_output'
accelerate launch --num_processes=8 train_pretokenized.py \
    --model-name /data/bwh-comppath-img2/MGH_CID/hf_cache/Qwen3-4B-Instruct-2507 \
    --dataset-path $ROOT/train_tokenized_qwen4b.parquet \
    --eval-dataset-path $ROOT/eval_tokenized_qwen4b.parquet \
    --output-dir /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/tmp \
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
    --report_to tensorboard \
    --resume_from_checkpoint /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/qwen3-4b-instruct_bs8_lr3e-5/checkpoint-10000
