#!/usr/bin/env python3
"""
Helper script to run Unsloth training with a fixed configuration.
This avoids the command-line argument parsing issues.
"""
import os
import json
import sys
import subprocess
from pathlib import Path

def main():
    # Set environment variables for better performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    
    # Make sure the output directory exists
    os.makedirs("unsloth_output", exist_ok=True)
    
    # Create a JSON config file with the correct structure for HfArgumentParser
    config = {
        # ModelArguments parameters
        "model_name_or_path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "trust_remote_code": True,
        
        # UnslothTrainingArguments parameters
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
        "bf16": True,
        "sample_packing": True
    }
    
    # Save the config to a file
    config_path = Path("unsloth_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created configuration file: {config_path}")
    
    # Print Unsloth version for debugging
    try:
        import unsloth
        print(f"Using Unsloth version: {unsloth.__version__}")
    except (ImportError, AttributeError):
        print("Could not determine Unsloth version.")
    
    # Run training with Unsloth using the JSON config file
    print("Starting training...")
    subprocess.run([sys.executable, "src/main/train_unsloth.py", str(config_path)])

if __name__ == "__main__":
    main()
