#!/bin/bash
# Script to run Unsloth training with optimized settings

# Set environment variables for better performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Enable TF32 for better performance on newer GPUs
export NVIDIA_TF32_OVERRIDE=1

# Clean up temporary files
rm -rf ~/tmp/*
rm -rf /tmp/unsloth_*

# Make sure the output directory exists
mkdir -p unsloth_output

# Install requirements if needed
if [ "$1" == "--install-deps" ]; then
    echo "Installing/updating required packages..."
    pip install -q -r src/main/unsloth_requirements.txt
    
    # Specifically update Unsloth to the latest version
    echo "Updating Unsloth to latest version..."
    pip install -q -U unsloth
fi

# Print Unsloth version for debugging
python -c "import unsloth; print(f'Using Unsloth version: {unsloth.__version__}')"

# Run training with Unsloth
echo "Starting training..."
python src/main/train_unsloth.py \
    --model_name_or_path "HuggingFaceTB/SmolLM2-1.7B-Instruct" \
    --output_dir "unsloth_output" \
    --batch_size 2 \
    --max_seq_length 8192 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --bf16 \
    --sample_packing
