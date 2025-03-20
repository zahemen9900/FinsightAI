#!/bin/bash
# Script to run optimized training for the SmolLM2 model

# Set environment variables for better performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1

# Clean CUDA cache
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Set default model
MODEL_NAME="HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Set output directory
OUTPUT_DIR="optimized_output"
mkdir -p "$OUTPUT_DIR"

# Check for resumption
RESUME_ARG=""
if [ -d "${OUTPUT_DIR}/checkpoint-*" ]; then
    echo "Found checkpoints in ${OUTPUT_DIR}"
    echo "Would you like to resume training? (y/n)"
    read resume_choice
    
    if [[ "$resume_choice" == "y" || "$resume_choice" == "Y" ]]; then
        latest_checkpoint=$(ls -d ${OUTPUT_DIR}/checkpoint-* | sort -V | tail -n 1)
        echo "Resuming from $latest_checkpoint"
        RESUME_ARG="--resume_from_checkpoint $latest_checkpoint"
    fi
fi

# Check if we should use deepspeed or not
DEEPSPEED_ARG=""
if [ -f "ds_zero2_config.json" ]; then
    DEEPSPEED_ARG="--deepspeed ds_zero2_config.json"
    echo "Using DeepSpeed configuration: ds_zero2_config.json"
fi

# Always use SDPA attention implementation since FlashAttention isn't available
ATTN_ARG="--attn_implementation sdpa"
echo "Using SDPA attention implementation (doesn't require flash-attn)"

# Display configuration
echo "Starting optimized training with:"
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Resuming: ${RESUME_ARG:-No}"
echo "DeepSpeed: ${DEEPSPEED_ARG:-Disabled}"
echo "Attention: SDPA (PyTorch native)"

# Run the training with optimized settings
CUDA_VISIBLE_DEVICES=0 python src/main/train_optimized.py \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_length 4096 \
    --logging_steps 10 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --num_train_epochs 3 \
    --save_total_limit 5 \
    --load_best_model_at_end true \
    --optim "adamw_8bit" \
    --lora_r 64 \
    --bf16 true \
    --gradient_checkpointing true \
    --pause_minutes 30 \
    $RESUME_ARG \
    $DEEPSPEED_ARG \
    $ATTN_ARG