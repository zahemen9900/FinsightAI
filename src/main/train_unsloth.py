#!/usr/bin/env python3
"""
Fine-tune FinSight AI models using Unsloth for faster and more memory-efficient training
compared to DeepSpeed. Unsloth is specialized for Llama and Mistral models with optimized
kernels and more efficient training approaches.
"""

import argparse
import os
import gc
import sys
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional
from dataclasses import dataclass, field

import torch
import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig
)

from rich.logging import RichHandler
from rich.console import Console
from rich.progress import track

# Import from local scripts
from train import ModelArguments, prepare_dataset

# Import Unsloth - Fixed imports to match current API
from unsloth import FastLanguageModel
# Remove the problematic import
# Import patch functionality only if needed later in the code

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

# Print Unsloth version for debugging
try:
    import unsloth
    logger.info(f"Unsloth version: {unsloth.__version__}")
except (ImportError, AttributeError):
    logger.warning("Could not determine Unsloth version")

@dataclass
class UnslothTrainingArguments(TrainingArguments):
    """Arguments for Unsloth training"""
    # Unsloth specific parameters
    max_seq_length: int = 8192
    sample_packing: bool = True  # Enables packing multiple samples in a single sequence
    bf16: bool = True  # Use bfloat16 precision
    fp16: bool = False  # Don't use fp16 when bf16 is enabled
    
    # LoRA parameters
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Basic training parameters
    output_dir: str = "unsloth_output"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    optim: str = "adamw_torch"  # Unsloth works well with standard optimizers
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine_with_restarts"
    warmup_ratio: float = 0.03
    
    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Evaluation strategy
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict = field(default_factory=lambda: {"use_reentrant": False})
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Other parameters
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int = 4
    
    def __post_init__(self):
        """Override post init to convert relative eval steps to absolute if needed"""
        super().__post_init__()
        
        # If eval_steps or save_steps is a float, interpret as a fraction of training
        if isinstance(self.eval_steps, float) and self.eval_steps < 1.0:
            if self.max_steps > 0:
                self.eval_steps = int(self.max_steps * self.eval_steps)
            else:
                # Estimate total steps
                estimated_steps = self.num_train_epochs * 1000  # just a guess
                self.eval_steps = max(10, int(estimated_steps * self.eval_steps))
                logger.info(f"Setting eval_steps to {self.eval_steps} (estimated from epochs)")
                
        if isinstance(self.save_steps, float) and self.save_steps < 1.0:
            if self.max_steps > 0:
                self.save_steps = int(self.max_steps * self.save_steps)
            else:
                # Use same as eval steps
                self.save_steps = self.eval_steps
                logger.info(f"Setting save_steps to {self.save_steps}")

def merge_datasets(dataset_paths: List[Dict[str, Union[str, float]]], tokenizer, num_proc: int = 4) -> DatasetDict:
    """Load and merge multiple datasets with specified proportions"""
    logger.info("Loading and merging datasets...")
    
    all_datasets = []
    for dataset_info in dataset_paths:
        try:
            path = dataset_info['path']
            name = dataset_info['name']
            proportion = dataset_info.get('proportion', 1.0)  # Default to using full dataset
            
            # Load dataset
            dataset = prepare_dataset(path, tokenizer, num_proc=num_proc)
            
            # Apply proportion if less than 1.0
            if proportion < 1.0:
                for split in dataset:
                    num_samples = int(len(dataset[split]) * proportion)
                    dataset[split] = dataset[split].shuffle(seed=42).select(range(num_samples))
            
            all_datasets.append(dataset)
            
            # Log sample and size info
            logger.info(f"\nDataset: {name}")
            logger.info(f"Using {proportion*100:.1f}% of data")
            logger.info(f"Size: {len(dataset['train'])} training samples")
            
            # Log sample
            sample_idx = random.randint(0, len(dataset['train'])-1)
            logger.info(f"Sample from {name}:")
            logger.info(f"{dataset['train'][sample_idx]['text'][:500]}...")
            
        except Exception as e:
            logger.warning(f"Failed to load dataset {name} from {path}: {e}")
            continue
    
    if not all_datasets:
        raise ValueError("No datasets were successfully loaded")
    
    # Merge splits
    merged_train = concatenate_datasets([d["train"] for d in all_datasets])
    merged_test = concatenate_datasets([d["test"] for d in all_datasets])
    
    # Shuffle
    merged_train = merged_train.shuffle(seed=42)
    merged_test = merged_test.shuffle(seed=42)
    
    logger.info(f"\nFinal merged dataset sizes:")
    logger.info(f"Train: {len(merged_train)}, Test: {len(merged_test)}")
    
    return DatasetDict({
        "train": merged_train,
        "test": merged_test
    })

def setup_unsloth_model(model_name_or_path: str, training_args: UnslothTrainingArguments):
    """Set up a model using Unsloth for faster training"""
    # Clear CUDA cache before model setup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Check model name - print warning if it's not what we expected
    expected_model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    if model_name_or_path != expected_model:
        logger.warning(f"Model path is {model_name_or_path}, not the expected {expected_model}")
        override = input(f"Do you want to override and use {expected_model} instead? (y/n): ")
        if override.lower() in ["y", "yes"]:
            model_name_or_path = expected_model
            logger.info(f"Using {model_name_or_path} instead")
    
    logger.info(f"Using model path: {model_name_or_path}")
    
    # Set up compute dtype
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    
    # Configure quantization settings
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    logger.info(f"Loading model from {model_name_or_path}...")
    
    # Set temporary environment variables instead of using unavailable args
    original_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    original_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    
    # Force download by disabling offline mode
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    
    try:
        # Try to use more specific parameters to force correct model loading
        logger.info("Attempting to load model with explicit parameters...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name_or_path=model_name_or_path,
            max_seq_length=training_args.max_seq_length,
            dtype=compute_dtype,
            load_in_4bit=True,
            quantization_config=quantization_config,
            trust_remote_code=True,
            # Don't use parameters that aren't accepted by the function
            # use_auth_token, force_download etc. are passed via environment variables
        )
        
    except Exception as e:
        logger.warning(f"Error with initial loading method: {e}")
        logger.info("Trying alternative loading method...")
        
        # Try to clean HF cache for this model to force a fresh download
        try:
            import shutil
            from huggingface_hub import HfFolder, HfApi
            
            # Create API object
            api = HfApi()
            
            # Get cache directory
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            model_cache = os.path.join(cache_dir, "models--" + model_name_or_path.replace("/", "--"))
            
            if os.path.exists(model_cache):
                logger.info(f"Removing cached model from {model_cache}")
                shutil.rmtree(model_cache, ignore_errors=True)
        except Exception as cache_error:
            logger.warning(f"Failed to clean cache: {cache_error}")
        
        # Try fallback loading
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name_or_path=model_name_or_path,
            max_seq_length=training_args.max_seq_length,
            load_in_4bit=True,
            trust_remote_code=True
        )
    finally:
        # Restore original environment variables
        if original_offline is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = original_offline
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            
        if original_hf_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = original_hf_offline
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
    
    # Verify loaded model details
    try:
        config_name = getattr(model.config, "_name_or_path", "unknown")
        if "llama" in config_name.lower() and "SmolLM2" in model_name_or_path:
            logger.warning(f"Loaded model appears to be {config_name}, but requested {model_name_or_path}")
            logger.warning("Training will continue but may not use the expected model!")
    except:
        pass
    
    # Apply Unsloth's LoRA optimization
    # Updated API usage for newer Unsloth versions
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=training_args.lora_r,
        target_modules=training_args.target_modules,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        # Removed arguments that might not be in newer API versions
    )
    
    # Log trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer

def main():
    """Main training function"""
    # Parse arguments
    parser = HfArgumentParser([ModelArguments, UnslothTrainingArguments])
    
    # We need to handle argument parsing differently to avoid conflicts
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass a JSON file, parse it with better error handling
        json_file = os.path.abspath(sys.argv[1])
        try:
            # Load and check the JSON data
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Print JSON keys for debugging
            logger.info(f"Loaded config with keys: {list(data.keys())}")
            
            # Remove keys not recognized by the parser
            invalid_keys = [
                'force_download', 'resume_download', 'local_files_only', 
                'use_auth_token', 'use_cache', 'cache_dir'
            ]
            
            for key in invalid_keys:
                if key in data:
                    # Store in environment variable if possible
                    if key == 'force_download' and data[key]:
                        os.environ["TRANSFORMERS_OFFLINE"] = "0"
                        os.environ["HF_HUB_OFFLINE"] = "0"
                    elif key == 'cache_dir':
                        os.environ["HUGGINGFACE_HUB_CACHE"] = data[key]
                        
                    logger.info(f"Removing unsupported key '{key}' from config")
                    data.pop(key)
            
            # Fix nested model_args if present
            if "model_args" in data:
                model_args_dict = data.pop("model_args")
                data.update(model_args_dict)
                logger.info("Flattened 'model_args' into main config")
                
            # Handle evaluation_strategy/eval_strategy confusion 
            if "evaluation_strategy" in data and "eval_strategy" not in data:
                data["eval_strategy"] = data["evaluation_strategy"]
                logger.info("Mapped 'evaluation_strategy' to 'eval_strategy'")
            
            # Parse the corrected data
            model_args, training_args = parser.parse_dict(data)
            logger.info(f"Successfully parsed config from {json_file}")
            
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            logger.error(f"Please ensure all arguments are at the top level (not nested under model_args)")
            
            # List valid arguments for ModelArguments
            try:
                logger.error(f"Valid arguments for ModelArguments: {list(ModelArguments.__dataclass_fields__.keys())}")
            except:
                logger.error("Could not list valid model arguments")
                
            # List valid arguments for UnslothTrainingArguments
            try:
                trainer_args = [f for f in UnslothTrainingArguments.__dataclass_fields__.keys() 
                               if not f.startswith("_")]
                logger.error(f"Valid arguments for UnslothTrainingArguments include: {', '.join(trainer_args[:10])}, ...")
            except:
                logger.error("Could not list valid training arguments")
                
            sys.exit(1)
    else:
        # Parse directly into dataclasses - this avoids adding duplicate arguments
        model_args, training_args = parser.parse_args_into_dataclasses()
        
        # Handle special command-line arguments that might not be in the dataclass
        # without creating duplicates
        cmd_parser = argparse.ArgumentParser(add_help=False)
        cmd_parser.add_argument("--batch_size", type=int, help="Override batch size")
        cmd_parser.add_argument("--max_seq_length", type=int, help="Override sequence length")
        
        # Only parse known args for these special overrides
        cmd_args, _ = cmd_parser.parse_known_args()
        
        # Apply overrides from command line if specified
        if hasattr(cmd_args, 'batch_size') and cmd_args.batch_size is not None:
            logger.info(f"Overriding batch size with command line value: {cmd_args.batch_size}")
            training_args.per_device_train_batch_size = cmd_args.batch_size
            training_args.per_device_eval_batch_size = cmd_args.batch_size
            
        if hasattr(cmd_args, 'max_seq_length') and cmd_args.max_seq_length is not None:
            logger.info(f"Overriding max_seq_length with command line value: {cmd_args.max_seq_length}")
            training_args.max_seq_length = cmd_args.max_seq_length
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Set up logging
    logging.getLogger("transformers").setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_info()
    transformers.utils.logging.set_verbosity_info()
    
    # Log loaded arguments
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Output directory: {training_args.output_dir}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Sequence length: {training_args.max_seq_length}")
    
    # Load model and tokenizer using Unsloth
    model, tokenizer = setup_unsloth_model(model_args.model_name_or_path, training_args)
    
    # Updated dataset paths with names and proportions - same as original script
    dataset_paths = [
        {
            "path": "/home/zahemen/datasets/sft_datasets/intro_conversations.jsonl",
            "name": "finsight_intro",
            "proportion": 1.0
        },
        {
            "path": "/home/zahemen/datasets/sft_datasets/reddit_finance_conversations.jsonl",
            "name": "reddit_finance",
            "proportion": 1.0
        },
        {
            "path": "/home/zahemen/datasets/sft_datasets/company_conversations.jsonl",
            "name": "finance_qa",
            "proportion": 1.0
        },
        {
            "path": "/home/zahemen/datasets/sft_datasets/financial_definitions_dataset.jsonl",
            "name": "financial_definitions",
            "proportion": 1.0
        },
        {
            "path": "/home/zahemen/datasets/sft_datasets/finance_conversations.jsonl",
            "name": "finance_conversations",
            "proportion": 1.0
        },
        {
            "path": "/home/zahemen/datasets/sft_datasets/advanced_finance_conversations.jsonl",
            "name": "advanced_finance_questions",
            "proportion": 1.0
        }
    ]
    
    # Load and merge datasets
    try:
        dataset = merge_datasets(dataset_paths, tokenizer, num_proc=training_args.dataloader_num_workers)
    except Exception as e:
        logger.error(f"Failed to load and merge datasets: {e}")
        sys.exit(1)
    
    # Log samples from merged dataset
    for index in random.sample(range(len(dataset["train"])), min(3, len(dataset["train"]))):
        logger.info(f"Sample {index} of the merged training set: \n\n{dataset['train'][index]['text'][:1000]}...")
    
    # Prepare data collator - Unsloth handles this efficiently
    # Updated for newer API versions
    try:
        data_collator = FastLanguageModel.get_packing_data_collator(
            tokenizer, training_args.max_seq_length, training_args.sample_packing
        )
    except (TypeError, AttributeError):
        logger.warning("Error with standard data collator, trying alternative approach")
        # Alternative for newer Unsloth versions
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        logger.info("Using standard DataCollatorForLanguageModeling")
    
    # Initialize the trainer - check for UnslothTrainer first, fall back to standard Trainer
    try:
        from unsloth.trainer import UnslothTrainer
        trainer_class = UnslothTrainer
        logger.info("Using UnslothTrainer")
    except ImportError:
        from transformers import Trainer
        trainer_class = Trainer
        logger.warning("UnslothTrainer not found, falling back to standard Trainer")
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Clear memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Train the model
    logger.info("Starting training with Unsloth...")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # Save the model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Run evaluation
    logger.info("Running evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # Save model metadata
    metadata = {
        "framework": "unsloth",
        "base_model": model_args.model_name_or_path,
        "training_parameters": {
            "lora_r": training_args.lora_r,
            "lora_alpha": training_args.lora_alpha,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "max_seq_length": training_args.max_seq_length,
            "sample_packing": training_args.sample_packing,
        },
        "dataset_stats": {
            "train_samples": len(dataset["train"]),
            "eval_samples": len(dataset["test"]),
            "sources": [info["name"] for info in dataset_paths]
        },
    }
    
    with open(os.path.join(training_args.output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Training complete! Model saved to {training_args.output_dir}")
    return

if __name__ == "__main__":
    main()
