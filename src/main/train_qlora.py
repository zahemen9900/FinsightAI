import os
import logging
import random
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import numpy as np
from sympy import comp
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
import argparse

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
    # LoRA specific parameters - optimized for speed
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05 
    
    # Training parameters optimized for speed
    num_train_epochs: int = 4
    learning_rate: float = 2e-4
    output_dir: str = "qlora_output"
    per_device_train_batch_size: int = 2   # Adjusted for memory
    per_device_eval_batch_size: int = 3
    gradient_accumulation_steps: int = 4    # Reduced for faster updates
    logging_steps: int = 50
    warmup_ratio: float = 0.15
    logging_dir: str = "logs"
    lr_scheduler_type: str = 'cosine_with_restarts'
    do_eval: bool = True
    eval_steps: int = 750      
    save_steps: int = 750
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 5   # Keep more checkpoints for resuming
    load_best_model_at_end: bool = True
    # lower_is_better: bool = True # minimize loss
    metric_for_best_model: str = "combined_score"
    greater_is_better: bool = True
    # Optimized DeepSpeed config for faster training
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

    # Model settings optimized for speed
    bf16: bool = True
    # fp16: bool = True
    double_quant: bool = True
    quant_type: str = "nf4"
    dataset_num_proc: int = 6
    use_cache: bool = True
    
    # Memory optimizations
    max_grad_norm: float = 0.2
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Add resume training parameters
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint directory
    save_safetensors: bool = True  # Better format for saving checkpoints
    
    # Enhanced training parameters for consistency
    weight_decay: float = 0.05             # Added weight decay
    num_cycles: int = 3                    # Number of LR cycles
    
    # Added focused attention params
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    
    # Memory optimization
    gradient_checkpointing: bool = True
    torch_compile: bool = True             # Use torch.compile for speed
    optim: str = "paged_adamw_32bit"      # Memory efficient optimizer
    
    # Add evaluation optimization params
    
    # Memory optimizations
    max_grad_norm: float = 0.2
    max_eval_samples: int = 1000  # Limit eval samples
    eval_accumulation_steps: int = 4
    
    def __post_init__(self):
        super().__post_init__()
        self.gradient_checkpointing_kwargs = {
            "use_reentrant": False,
            # 'use_cache': False,
            "use_gradient_scaling": True    # Added for better stability
        }
        # If resuming, ensure we load the best model
        if self.resume_from_checkpoint:
            self.load_best_model_at_end = True
        self.evaluation_strategy = "steps"
        self.eval_steps = 500
        self.per_device_eval_batch_size = self.eval_batch_size

def setup_quantized_model(model_args, training_args):
    """Enhanced model setup with consistency improvements"""
    
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
    
    # Enhanced LoRA configuration
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
            "down_proj",
            "mixer_self_attention",  # Added for better attention
            "mixer_cross_attention", # Added for better attention
            "mixer_mlp",            # Added for better feature mixing
        ],
        init_lora_weights="gaussian",  # Changed from default
    )
    
    # Get PEFT model with enhanced settings
    model = get_peft_model(model, peft_config)
    
    # Add more aggressive memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    # Add focused attention dropouts
    if hasattr(model, "config"):
        model.config.attention_dropout = training_args.attn_dropout
        model.config.resid_dropout = training_args.resid_dropout
        model.config.embed_dropout = training_args.embed_dropout
    
    # Enable more efficient training
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Enable flash attention if available
    if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "flash_attention_2"
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model

def merge_datasets(dataset_paths: List[Dict[str, Union[str, float]]], tokenizer, num_proc: int = 4) -> Dataset:
    """Load and merge multiple datasets with specified proportions
    
    Args:
        dataset_paths: List of dicts containing {'path': str, 'name': str, 'proportion': float}
        tokenizer: Tokenizer instance
        num_proc: Number of processes for parallel processing
    """
    logger.info("Loading and merging datasets...")
    
    all_datasets = []
    for dataset_info in dataset_paths:
        try:
            path = dataset_info['path']
            name = dataset_info['name']
            proportion = dataset_info.get('proportion', 1.0)  # Default to using full dataset
            
            # Load dataset
            dataset = prepare_dataset(path, tokenizer, num_proc=num_proc)
            
            # Apply proportion if less than 1
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



def compute_consistency_metrics(tokenizer, eval_pred):
    """Optimized metrics computation with reduced memory usage"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Process in smaller batches to reduce memory usage
    batch_size = 32  # Adjust based on your GPU memory
    metrics_sum = {"response_consistency": 0.0, "topic_adherence": 0.0}
    total_batches = 0
    
    try:
        # Process predictions in batches
        for i in range(0, len(predictions), batch_size):
            batch_preds = predictions[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            # Filter padding tokens
            mask = batch_labels != -100
            filtered_preds = batch_preds[mask]
            filtered_labels = batch_labels[mask]
            
            if len(filtered_preds) == 0:
                continue
                
            # Calculate metrics for batch
            consistency = calculate_response_consistency(filtered_preds, filtered_labels)
            
            # Use CPU for text decoding to save GPU memory
            with torch.cuda.device('cpu'):
                topic_score = calculate_topic_adherence(
                    tokenizer, 
                    filtered_preds, 
                    filtered_labels
                )
            
            metrics_sum["response_consistency"] += consistency
            metrics_sum["topic_adherence"] += topic_score
            total_batches += 1
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate averages
        if total_batches > 0:
            metrics = {
                "response_consistency": metrics_sum["response_consistency"] / total_batches,
                "topic_adherence": metrics_sum["topic_adherence"] / total_batches,
            }
            metrics["combined_score"] = (metrics["response_consistency"] + 
                                       metrics["topic_adherence"]) / 2
        else:
            metrics = {
                "response_consistency": 0.0,
                "topic_adherence": 0.0,
                "combined_score": 0.0
            }
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Error computing metrics: {e}")
        return {
            "response_consistency": 0.0,
            "topic_adherence": 0.0,
            "combined_score": 0.0
        }

# Add these helper functions at the end of the file
def calculate_response_consistency(predictions, labels):
    """Calculate how consistent the responses are with the questions"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        from scipy.stats import entropy
        
        # Convert to numpy arrays if needed
        pred_array = np.array(predictions)
        label_array = np.array(labels)
        
        # Calculate multiple consistency metrics
        scores = []
        
        # 1. Distribution similarity
        pred_dist = np.bincount(pred_array.flatten()) / len(pred_array.flatten())
        label_dist = np.bincount(label_array.flatten()) / len(label_array.flatten())
        
        # Pad distributions to same length
        max_len = max(len(pred_dist), len(label_dist))
        pred_dist = np.pad(pred_dist, (0, max_len - len(pred_dist)))
        label_dist = np.pad(label_dist, (0, max_len - len(label_dist)))
        
        # KL divergence (lower is better)
        kl_div = entropy(pred_dist + 1e-10, label_dist + 1e-10)
        distribution_score = 1 / (1 + kl_div)  # Convert to 0-1 scale
        
        # 2. Sequential consistency
        # Check if predictions maintain similar patterns as labels
        pred_diffs = np.diff(pred_array, axis=-1)
        label_diffs = np.diff(label_array, axis=-1)
        sequence_score = np.mean(np.sign(pred_diffs) == np.sign(label_diffs))
        
        # 3. Response length consistency
        pred_lengths = (pred_array != 0).sum(axis=-1)  # Assuming 0 is padding
        label_lengths = (label_array != 0).sum(axis=-1)
        length_ratio = np.minimum(pred_lengths, label_lengths) / np.maximum(pred_lengths, label_lengths)
        length_score = np.mean(length_ratio)
        
        # Combine scores with weights
        final_score = (
            0.4 * distribution_score +
            0.4 * sequence_score +
            0.2 * length_score
        )
        
        return float(final_score)
    except Exception as e:
        logger.warning(f"Error calculating response consistency: {e}")
        return 0.0

def calculate_topic_adherence(tokenizer, predictions, labels):
    """Calculate how well responses stick to the given topic using finance-specific metrics"""
    try:
        import numpy as np
        
        # Financial domain keywords (subset for efficiency)
        finance_keywords = {
            'high_relevance': {
            'investment', 'stock', 'market', 'fund', 'portfolio', 'risk',
            'return', 'asset', 'equity', 'bond', 'dividend', 'trading',
            'financial', 'bank', 'interest', 'rate', 'profit', 'loss',
            'capital', 'debt', 'credit', 'money', 'price', 'value',
            'volatility', 'liquidity', 'derivative', 'futures', 'option',
            'hedge', 'leverage', 'yield', 'commodity', 'forex', 'exchange',
            'securities', 'inflation', 'bear', 'bull', 'margin', 'broker',
            'valuation', 'arbitrage', 'collateral', 'mortgage', 'sector',
            'etf', 'reit', 'mutual', 'index', 'shares', 'treasury'
            },
            'medium_relevance': {
            'growth', 'performance', 'strategy', 'analysis', 'management',
            'income', 'revenue', 'cost', 'expense', 'cash', 'flow', 'tax',
            'fee', 'charge', 'account', 'balance', 'margin', 'trend',
            'allocation', 'diversification', 'beta', 'alpha', 'ratio',
            'earnings', 'capitalization', 'benchmark', 'correlation',
            'maturity', 'premium', 'spread', 'portfolio', 'fundamental',
            'technical', 'momentum', 'resistance', 'support', 'volume',
            'fiscal', 'budget', 'quarter', 'annual', 'dividend', 'payout',
            'sustainable', 'compliance', 'regulatory', 'audit', 'liability'
            }
        }
        
        def calculate_keyword_density(text, keywords):
            """Calculate weighted keyword density in text"""
            words = set(text.lower().split())
            high_matches = len(words.intersection(keywords['high_relevance']))
            med_matches = len(words.intersection(keywords['medium_relevance']))
            total_words = len(words)
            
            if total_words == 0:
                return 0.0
                
            return (high_matches * 1.5 + med_matches) / total_words
        
        # Convert token IDs to text using tokenizer
        pred_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
        label_texts = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
        
        # Calculate topic adherence scores
        pred_scores = np.mean([calculate_keyword_density(text, finance_keywords) 
                             for text in pred_texts])
        label_scores = np.mean([calculate_keyword_density(text, finance_keywords) 
                              for text in label_texts])
        
        # Compare prediction adherence to label adherence
        relative_adherence = pred_scores / max(label_scores, 1e-5)
        
        # Additional checks for response structure
        def check_response_structure(text):
            """Check if response follows expected financial discussion structure"""
            # Convert text to lowercase for checking
            text = text.lower()
            
            # Check for common financial discussion patterns
            has_numbers = bool(re.search(r'\d', text))
            has_percentages = '%' in text
            has_currency = bool(re.search(r'[\$€£¥]', text))
            has_comparison = bool(re.search(r'(increase|decrease|higher|lower|more|less)', text))
            
            # Calculate structure score
            structure_score = np.mean([
                has_numbers,
                has_percentages,
                has_currency,
                has_comparison
            ])
            
            return structure_score
        
        # Calculate structure adherence
        pred_structure = np.mean([check_response_structure(text) for text in pred_texts])
        label_structure = np.mean([check_response_structure(text) for text in label_texts])
        
        structure_score = pred_structure / max(label_structure, 1e-5)
        
        # Combine scores with weights
        final_score = (
            0.6 * relative_adherence +
            0.4 * structure_score
        )
        
        # Clip to ensure score is between 0 and 1
        return float(np.clip(final_score, 0, 1))
        
    except Exception as e:
        logger.warning(f"Error calculating topic adherence: {e}")
        return 0.0

def train():
    # Initialize arguments
    model_args = ModelArguments()
    training_args = QLoRAConfig()
    
    # Add argument parsing for resume training
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Path to checkpoint directory to resume training from"
    )
    args, _ = parser.parse_known_args()
    
    if args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = args.resume_from_checkpoint
        logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    
    # Updated dataset paths with names and proportions
    dataset_paths = [
        {
            "path": "/home/zahemen/datasets/reddit-finance-250k/sft_cleaned_data.jsonl",
            "name": "reddit_finance",
            "proportion": 1.0
        },
        {
            "path": "/home/zahemen/datasets/finance_qa_conversations.jsonl",
            "name": "finance_qa",
            "proportion": 1.0
        },
        {
            "path": "/home/zahemen/datasets/intro_conversations.jsonl",
            "name": "intro_conversations",
            "proportion": 1.0
        }
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

    # Add memory optimization before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Add gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Initialize trainer with optimized settings
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"].select(range(min(
            len(dataset["test"]), 
            training_args.max_eval_samples
        ))),  # Limit eval dataset size
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.05
            )
        ],
        compute_metrics=lambda eval_pred: compute_consistency_metrics(tokenizer, eval_pred),
        data_collator=None,
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
