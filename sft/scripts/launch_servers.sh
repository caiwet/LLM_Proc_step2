mkdir -p logs

MODEL=medgemma1.5-4b-instruct_bs8_lr3e-5

echo "Launching 8 vLLM servers..."

for i in 1 2 3 4 6 7; do
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
        --model /data/bwh-comppath-img2/MGH_CID/LLM_Proc/sft_out/$MODEL \
        --port $((8000 + i)) \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --dtype bfloat16 \
        --attention-backend FLASH_ATTN \
        > logs/{$MODEL}_vllm_gpu${i}.log 2>&1 &
    
    echo "Started vLLM server on GPU $i at port $((8000 + i))"
    sleep 3
done

echo "All vLLM servers started on ports 8000-8007"

