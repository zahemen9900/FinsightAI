import os
import logging
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import torch
import datasets
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from trl import SFTTrainer, SFTConfig
from train import ModelArguments, prepare_dataset
from rich.logging import RichHandler
import wandb
from transformers.trainer_callback import EarlyStoppingCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')



@dataclass 
class QLoRAConfig(SFTConfig):
    # LoRA specific parameters
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Training specific parameters 
    num_train_epochs: int = 2
    learning_rate: float = 2e-5
    output_dir: str = "qlora_output"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    logging_steps: int = 30
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = 'cosine' 
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 4
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"  # Added for early stopping
    greater_is_better: bool = False  # Added for early stopping - we want loss to decrease

    # DeepSpeed configs
    deepspeed = {
        "zero_optimization": {
            "stage": 2, #ZeRO Stage 2 is often the best performance / memory configuration
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "gradient_checkpointing": False
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto"
        },
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "steps_per_print": 50,
        "wall_clock_breakdown": False
    }

    # Model specific
    bf16: bool = True
    fp16: bool = False
    double_quant: bool = True
    quant_type: str = "nf4"
    dataset_num_proc: int = 1
    use_cache: bool = False # Set to false to avoid gradient checkpointing warning

    # Add gradient checkpointing settings
    gradient_checkpointing_kwargs: dict = None
    
    def __post_init__(self):
        super().__post_init__()
        # Set default gradient checkpointing kwargs if not provided
        if self.gradient_checkpointing_kwargs is None:
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}

def setup_quantized_model(model_args, training_args):
    """Set up quantized model with LoRA configuration"""
    
    # Quantization configuration
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=training_args.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=training_args.double_quant,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Set static graph for DDP
    if hasattr(model, "_set_static_graph"):
        model._set_static_graph()
        
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Set up LoRA configuration
    peft_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    return model

def merge_datasets(dataset_paths: List[str], tokenizer, num_proc: int = 4) -> Dataset:
    """Load and merge multiple datasets with improved chat template handling"""
    logger.info("Loading and merging datasets...")
    
    all_datasets = []
    for path in dataset_paths:
        try:
            dataset = prepare_dataset(path, tokenizer, num_proc=num_proc)
            all_datasets.append(dataset)
            
            # Log sample from dataset
            sample_idx = random.randint(0, len(dataset['train'])-1)
            logger.info(f"\nSample from {path}:")
            logger.info(f"{dataset['train'][sample_idx]['text'][:500]}...")
            
        except Exception as e:
            logger.warning(f"Failed to load dataset from {path}: {e}")
            continue
    
    if not all_datasets:
        raise ValueError("No datasets were successfully loaded")
    
    # Merge splits
    merged_train = datasets.concatenate_datasets([d["train"] for d in all_datasets])
    merged_test = datasets.concatenate_datasets([d["test"] for d in all_datasets])
    
    # Shuffle
    merged_train = merged_train.shuffle(seed=42)
    merged_test = merged_test.shuffle(seed=42)
    
    logger.info(f"\nFinal merged dataset sizes:")
    logger.info(f"Train: {len(merged_train)}, Test: {len(merged_test)}")
    
    return datasets.DatasetDict({
        "train": merged_train,
        "test": merged_test
    })

def train():
    # Initialize arguments
    model_args = ModelArguments()
    training_args = QLoRAConfig()
    
    # Add support for multiple dataset paths
    dataset_paths = [
        "/home/zahemen/datasets/reddit-finance-250k/sft_cleaned_data.jsonl",
        "/home/zahemen/datasets/finance_qa_conversations.jsonl",
        "/home/zahemen/datasets/intro_conversations.jsonl"
    ]

    # Set seed for reproducibility 
    set_seed(training_args.seed)

    # Set up logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=model_args.trust_remote_code
    )

    # Load and merge datasets
    try:
        dataset = merge_datasets(dataset_paths, tokenizer)
    except Exception as e:
        logger.error(f"Failed to load and merge datasets: {e}")
        sys.exit(1)

    # Log samples from merged dataset
    with training_args.main_process_first(desc="Log samples from merged training set"):
        for index in random.sample(range(len(dataset["train"])), 3):
            logger.info(f"Sample {index} of the merged training set: \n\n{dataset['train'][index]['text']}")

    # Initialize model with QLoRA
    model = setup_quantized_model(model_args, training_args)
    model.print_trainable_parameters()  # Log trainable parameters

    # Initialize trainer with early stopping callback
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.1
            )
        ]
    )

    # Train
    logger.info("Starting training")
    try:
        train_result = trainer.train()
        
        # Log and save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Save model
        logger.info(f"Saving model to {training_args.output_dir}")
        trainer.save_model()
        trainer.save_state()
        
        # Evaluate
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    try:
        train()
        # wandb.finish()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
