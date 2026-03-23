export HF_HOME=/data/bwh-comppath-img/concept_extraction/.hf_cache
ROOT='/data/bwh-comppath-img2/MGH_CID/LLM_Proc/step2_gemini_output'
accelerate launch --num_processes=8 train_pretokenized.py \
    --model-name /data/bwh-comppath-img2/MGH_CID/hf_cache/Qwen3-4B-Instruct-2507 \
    --dataset-path $ROOT/train_tokenized_qwen3_4b.parquet \
    --eval-dataset-path $ROOT/eval_tokenized_qwen3_4b.parquet \
    --output-dir /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/qwen3-4b-instruct_bs8_lr3e-5 \
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

# accelerate launch --num_processes=8 train_pretokenized.py \
#     --model-name /data/bwh-comppath-img3/concept_extraction/sft/models/medgemma_llm \
#     --dataset-path $ROOT/train_tokenized_medgemma1.5.parquet \
#     --eval-dataset-path $ROOT/eval_tokenized_medgemma1.5.parquet \
#     --output-dir /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/medgemma1.5-4b-instruct_bs8_lr3e-5_all_data \
#     --num-train-epochs 2 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 3e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_steps 2000 \
#     --logging_steps 10 \
#     --save_steps 5000 \
#     --eval_steps 5000 \
#     --save_total_limit 3 \
#     --report_to tensorboard 