#!/bin/bash
# Script to run Unsloth training with a completely fresh environment
# This clears caches and forces the correct model to be used

# Set environment variables for better performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1

# Explicitly disable offline mode to force re-download
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HUGGINGFACE_HUB_CACHE="/tmp/hf_cache_tmp"

# Clean up temporary files
rm -rf ~/tmp/*
rm -rf /tmp/unsloth_*
rm -rf ${HUGGINGFACE_HUB_CACHE}
mkdir -p ${HUGGINGFACE_HUB_CACHE}

# Make sure the output directory exists
mkdir -p unsloth_output

# Print Unsloth version for debugging
python -c "import unsloth; print(f'Using Unsloth version: {unsloth.__version__}')"

# Clear Huggingface cache for problematic models
echo "Clearing Huggingface cache for Unsloth models..."
python src/main/clear_hf_cache.py --clear_all_unsloth

# Create a proper flat JSON config file that will work with HfArgumentParser
# Remove force_download and other invalid arguments
cat > unsloth_config.json << EOL
{
    "model_name_or_path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "trust_remote_code": true,
    "output_dir": "unsloth_output",
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "max_seq_length": 8192,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "gradient_accumulation_steps": 4,
    "logging_steps": 10,
    "eval_steps": 1000,
    "save_steps": 1000,
    "save_total_limit": 3,
    "bf16": true,
    "sample_packing": true,
    "eval_strategy": "steps"
}
EOL

echo "Created configuration file: unsloth_config.json"

# Run training with the correct config format
echo "Starting training..."
python src/main/train_unsloth.py unsloth_config.json
