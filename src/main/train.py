import os
import logging
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import datasets
import transformers
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from trl import SFTTrainer, SFTConfig
from rich.logging import RichHandler

# Configure logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')

@dataclass
class ModelArguments:
    model_name_or_path: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    use_fast_tokenizer: bool = True
    torch_dtype: str = "auto"
    model_revision: str = "main"
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"

def apply_chat_template(example: Dict, tokenizer) -> Dict:
    """Apply chat template to messages and preserve metadata"""
    try:
        messages = example.get("messages", [])
        # Ensure all message contents are strings
        for msg in messages:
            if msg.get("content") is None:
                msg["content"] = ""
            
        # Apply template
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Preserve metadata in the processed example
        if "metadata" in example:
            example["source"] = example["metadata"].get("source", "")
            example["score"] = example["metadata"].get("score", 0.0)
            example["prompt_id"] = example["metadata"].get("prompt_id", "")
            example["has_context"] = example["metadata"].get("has_context", False)
            example["ticker"] = example["metadata"].get("ticker", "")
            example["filing_year"] = example["metadata"].get("filing_year", "")
            
        return example
    except Exception as e:
        logger.warning(f"Error applying chat template: {e}")
        # Provide fallback with metadata
        return {
            "text": "Error processing conversation",
            "source": "error",
            "score": 0.0,
            "prompt_id": "",
            "has_context": False,
            "ticker": "",
            "filing_year": ""
        }

def prepare_dataset(dataset_path: str, tokenizer) -> Dataset:
    """Load and prepare dataset for training with metadata handling"""
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        # Load dataset
        dataset = datasets.load_dataset('json', data_files=dataset_path)['train']
        
        # Apply chat template and preserve metadata
        logger.info("Applying chat template and processing metadata")
        
        # Keep metadata fields when mapping
        keep_columns = ['source', 'score', 'prompt_id', 'has_context', 'ticker', 'filing_year']
        
        dataset = dataset.map(
            lambda x: apply_chat_template(x, tokenizer),
            remove_columns=[col for col in dataset.column_names if col not in keep_columns],
            desc="Applying chat template"
        )
        
        # Log metadata statistics
        if 'source' in dataset.column_names:
            sources = dataset.unique('source')
            logger.info(f"Dataset sources: {sources}")
        
        if 'has_context' in dataset.column_names:
            context_ratio = sum(dataset['has_context']) / len(dataset)
            logger.info(f"Samples with context: {context_ratio:.2%}")
            
        # Split into train/test preserving metadata distribution
        dataset = dataset.train_test_split(
            test_size=0.3,
            seed=42,
            # Stratify by source if available to maintain distribution
            stratify_by_column='source' if 'source' in dataset.column_names else None
        )
        
        logger.info(f"Final dataset sizes - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load or process dataset: {e}")
        raise

def train():
    # Initialize arguments
    model_args = ModelArguments() 
    training_args = SFTConfig(
        output_dir="output",
        run_name="sft_training_run",  # Add unique run name to fix wandb warning
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_grad_norm=0.3,
        weight_decay=0.001,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        max_seq_length=8192,  # Reduce from default to match model's context window
        packing=True,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        do_eval=True,
        dataset_num_proc=1,
        use_cache=False  # Add this to fix gradient checkpointing warning
    )

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Set logging level
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.tokenizer_name or model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=model_args.trust_remote_code
    )

    # Load and prepare dataset
    dataset = prepare_dataset("/home/zahemen/datasets/reddit-finance-250k/sft_format_data.jsonl", tokenizer)

    # Log samples
    with training_args.main_process_first(desc="Log samples from training set"):
        for index in random.sample(range(len(dataset["train"])), 3):
            logger.info(f"Sample {index} of the training set: \n\n{dataset['train'][index]['text']}")

    # Initialize trainer with correct parameters and model configs
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_cache=False,  # Also set use_cache=False in model config
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
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
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
