import os
import logging
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Literal

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

def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.get_chat_template()

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})

def apply_chat_template(
    example: Dict,
    tokenizer,
    task: Literal["sft", "generation"] = "sft",
    auto_insert_empty_system_msg: bool = True,
    use_metadata: bool = True,
) -> Dict:
    """Apply chat template following SmolLM2's implementation"""
    try:
        messages = example["messages"]
        
        # Add empty system message if none exists
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
            
        # Apply template
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
        
        if use_metadata and "metadata" in example:
            # Copy ALL metadata fields directly to the example
            for key, value in example["metadata"].items():
                example[key] = value

        return example
    except Exception as e:
        logger.warning(f"Error applying chat template: {e}")
        return {
            "text": "Error processing conversation",
            "source": "error",
            "conversation_id": "",
            "type": "error"
        }

def prepare_dataset(dataset_path: str, tokenizer, num_proc: int = 4) -> Dataset:
    """Load and prepare dataset with improved chat template handling"""
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        # Load dataset
        dataset = datasets.load_dataset('json', data_files=dataset_path)['train']
        
        # Get columns to preserve
        metadata_columns = ['source', 'conversation_id', 'type']
        columns_to_remove = [col for col in dataset.column_names if col not in metadata_columns]
        
        # Apply chat template with parallel processing
        dataset = dataset.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "sft",
                "auto_insert_empty_system_msg": False,
            },
            num_proc=num_proc,
            remove_columns=columns_to_remove,
            desc="Applying chat template"
        )
        
        # Log dataset statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total examples: {len(dataset)}")
        if 'source' in dataset.column_names:
            sources = dataset.unique('source')
            source_counts = {src: len(dataset.filter(lambda x: x['source'] == src)) for src in sources}
            logger.info("Source distribution:")
            for src, count in source_counts.items():
                logger.info(f"  {src}: {count} ({count/len(dataset)*100:.1f}%)")
        
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.25, seed=42)
        logger.info(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
        
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
    dataset = prepare_dataset("/home/zahemen/datasets/sft_datasets/finsight_combined_dataset_20250304_121951.jsonl", tokenizer)

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
