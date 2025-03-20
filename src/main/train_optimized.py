#!/usr/bin/env python3
"""
Optimized training script for FinSight AI that combines the best of QLoRA with
memory-efficient training techniques. This script is designed specifically for
SmolLM2 and similar models, providing better performance than standard QLoRA
while being compatible with consumer hardware.
"""

import os
import gc
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union, Any

import torch
import datasets
import transformers
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from trl import SFTTrainer, SFTConfig
from train import ModelArguments, prepare_dataset
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import track
from transformers.trainer_callback import EarlyStoppingCallback
from callbacks import PauseResumeCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class OptimizedTrainingArguments(TrainingArguments):
    """
    Enhanced training arguments for optimal performance with SmolLM2 models
    """
    # LoRA specific parameters
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj",
        "mixer_self_attention", "mixer_cross_attention", "mixer_mlp"
    ])
    
    # Advanced memory optimization
    use_reentrant: bool = False
    use_gradient_scaling: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict = field(default_factory=lambda: {
        "use_reentrant": False
    })
    double_quant: bool = True
    quant_type: str = "nf4"
    bf16: bool = True
    fp16: bool = False
    torch_compile: bool = True
    
    # Training parameters
    output_dir: str = "optimized_output"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    logging_steps: int = 40
    warmup_ratio: float = 0.15
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine_with_restarts"
    max_grad_norm: float = 0.3
    
    # Evaluation and saving
    save_strategy: str = "steps"
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    save_steps: int = 1000
    save_total_limit: int = 4
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # DeepSpeed configuration for Stage 2 with memory offloading
    deepspeed_config: Optional[Dict] = field(default_factory=lambda: {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "gradient_clipping": 0.3,
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto"
    })
    
    # Resume training parameters
    resume_from_checkpoint: Optional[str] = None
    save_safetensors: bool = True
    
    # Advanced LLM training parameters
    max_length: int = 8192
    
    # Remove this conflicting parameter
    # attn_implementation: str = "flash_attention_2"
    
    rope_scaling: Optional[Dict] = None
    
    # Other parameters
    dataset_num_proc: int = 4
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Pause during training for thermal management
    pause_minutes: int = 30
    
    def __post_init__(self):
        super().__post_init__()
        
        # DeepSpeed specific handling
        if self.deepspeed and isinstance(self.deepspeed, str) and self.deepspeed.endswith(".json"):
            with open(self.deepspeed, "r") as f:
                self.deepspeed_config = json.load(f)
            
            # Override with our settings if file was provided
            self.deepspeed = self.deepspeed_config

def setup_optimized_model(model_args, training_args):
    """Set up a model with optimal memory settings for training"""
    
    # Compute dtype setup
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    
    # Enhanced quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=training_args.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=training_args.double_quant,
    )
    
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=model_args.trust_remote_code,
        # Use attention implementation from model_args instead
        attn_implementation=model_args.attn_implementation,
        rope_scaling=training_args.rope_scaling,
    )
    
    # Enable gradient checkpointing and other optimizations
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training_args.gradient_checkpointing
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            use_reentrant=training_args.use_reentrant
        )
    
    # Enhanced LoRA config for better training
    peft_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=training_args.target_modules,
        init_lora_weights="gaussian",
    )
    
    # Apply PEFT modifications
    model = get_peft_model(model, peft_config)
    
    # Memory optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Enable gradient checkpointing again after PEFT to ensure it's active
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            use_reentrant=training_args.use_reentrant
        )
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Torch compile if available and enabled
    if training_args.torch_compile:
        try:
            model = torch.compile(model)
            logger.info("Successfully compiled model with torch.compile")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
    
    return model

def merge_datasets(dataset_paths: List[Dict[str, Union[str, float]]], tokenizer, num_proc: int = 4) -> DatasetDict:
    """Load and merge multiple datasets with their respective proportions"""
    logger.info("Loading and merging datasets...")
    
    all_datasets = []
    for dataset_info in dataset_paths:
        try:
            path = dataset_info['path']
            name = dataset_info['name']
            proportion = dataset_info.get('proportion', 1.0)
            
            # Load dataset
            dataset = prepare_dataset(path, tokenizer, num_proc=num_proc)
            
            # Apply proportion if needed
            if proportion < 1.0:
                for split in dataset:
                    num_samples = int(len(dataset[split]) * proportion)
                    dataset[split] = dataset[split].shuffle(seed=42).select(range(num_samples))
            
            all_datasets.append(dataset)
            
            # Log info
            logger.info(f"\nDataset: {name}")
            logger.info(f"Using {proportion*100:.1f}% of data")
            logger.info(f"Size: {len(dataset['train'])} training samples")
            
            # Show sample
            sample_idx = random.randint(0, len(dataset['train'])-1)
            logger.info(f"Sample from {name}:")
            logger.info(f"{dataset['train'][sample_idx]['text'][:500]}...")
            
        except Exception as e:
            logger.warning(f"Failed to load dataset {name} from {path}: {e}")
            continue
    
    if not all_datasets:
        raise ValueError("No datasets were successfully loaded")
    
    # Merge datasets
    merged_train = datasets.concatenate_datasets([d["train"] for d in all_datasets])
    merged_test = datasets.concatenate_datasets([d["test"] for d in all_datasets])
    
    # Shuffle datasets
    merged_train = merged_train.shuffle(seed=42)
    merged_test = merged_test.shuffle(seed=42)
    
    logger.info(f"\nFinal merged dataset sizes:")
    logger.info(f"Train: {len(merged_train)}, Test: {len(merged_test)}")
    
    return DatasetDict({
        "train": merged_train,
        "test": merged_test
    })

def train():
    """Main training function with optimized settings"""
    # Parse arguments
    parser = HfArgumentParser([ModelArguments, OptimizedTrainingArguments])
    
    # Handle JSON config if provided
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Add CLI argument parsing for resume and pause
        parser_args = parser.parse_args_into_dataclasses()
        model_args, training_args = parser_args
        
        # Parse additional arguments
        cmd_parser = argparse.ArgumentParser(add_help=False)
        cmd_parser.add_argument("--resume_from_checkpoint", type=str, default=None)
        cmd_parser.add_argument("--pause_minutes", type=int, default=30)
        cmd_args, _ = cmd_parser.parse_known_args()
        
        # Override from command line if specified
        if cmd_args.resume_from_checkpoint:
            training_args.resume_from_checkpoint = cmd_args.resume_from_checkpoint
            logger.info(f"Resuming from checkpoint: {cmd_args.resume_from_checkpoint}")
            
        if cmd_args.pause_minutes != 30:  # Default is 30
            training_args.pause_minutes = cmd_args.pause_minutes
            logger.info(f"Pause duration set to {cmd_args.pause_minutes} minutes")
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Set up logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    # Set up deepspeed config properly for ZeRO-2
    if training_args.deepspeed:
        logger.info("DeepSpeed enabled - configuring for ZeRO-2")
        # Use file config if string, otherwise use the default dict
        if isinstance(training_args.deepspeed, str):
            logger.info(f"Using DeepSpeed config from: {training_args.deepspeed}")
        else:
            logger.info("Using default DeepSpeed ZeRO-2 configuration")
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Enable padding and EOS tokens for proper training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset paths with proportions
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
        dataset = merge_datasets(dataset_paths, tokenizer, training_args.dataset_num_proc)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)
    
    # Log examples from merged dataset
    for index in random.sample(range(len(dataset["train"])), min(3, len(dataset["train"]))):
        logger.info(f"Sample {index} of merged training set: \n{dataset['train'][index]['text'][:1000]}...")
    
    # Set up model with optimized settings
    model = setup_optimized_model(model_args, training_args)
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Set up trainer with callbacks
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.05
            ),
            PauseResumeCallback(pause_minutes=training_args.pause_minutes)
        ],
    )
    
    # Clear memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run training
    logger.info(f"Starting training with optimized settings...")
    
    try:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        # Log and save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Save model
        logger.info(f"Saving model to {training_args.output_dir}")
        trainer.save_model()
        trainer.save_state()
        
        # Run evaluation
        if training_args.do_eval:
            logger.info("Running evaluation...")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        
        # Save metadata
        metadata = {
            "base_model": model_args.model_name_or_path,
            "optimized_params": {
                "lora_r": training_args.lora_r,
                "lora_alpha": training_args.lora_alpha,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "warmup_ratio": training_args.warmup_ratio,
                "opt_level": "ZeRO-2" if training_args.deepspeed else "Standard",
                "quantization": f"4-bit {training_args.quant_type}"
            },
            "dataset_stats": {
                "train_samples": len(dataset["train"]),
                "eval_samples": len(dataset["test"]),
                "sources": [info["name"] for info in dataset_paths]
            }
        }
        
        with open(os.path.join(training_args.output_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
