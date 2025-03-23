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
from rich.console import Console
import wandb
from transformers.trainer_callback import EarlyStoppingCallback
import argparse
from callbacks import PauseResumeCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')
console = Console()

# Rest of the file remains unchanged
@dataclass 
class QLoRAConfig(SFTConfig):
    # LoRA specific parameters - optimized for speed
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training parameters optimized for speed
    num_train_epochs: int = 3
    learning_rate: float = 5e-6
    output_dir: str = "qlora_output"
    per_device_train_batch_size: int = 2   # Adjusted for memory
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4    # Reduced for faster updates
    logging_steps: int = 40
    warmup_ratio: float = 0.03
    logging_dir: str = "logs"
    lr_scheduler_type: str = 'cosine_with_restarts'
    do_eval: bool = True
    eval_steps: int = 1100      
    save_steps: int = 1100
    # max_steps: int = 3999           #Latest run yielded ~4100 steps, I rounded down for consistency
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 4   # Keep more checkpoints for resuming
    load_best_model_at_end: bool = True
    lower_is_better: bool = True # minimize loss
    # Pause duration in minutes
    pause_minutes: int = 30    
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
    double_quant: bool = False  #for better quality on larger datasets 
    quant_type: str = "nf4"
    dataset_num_proc: int = 4
    use_cache: bool = True
    
    # Memory optimizations
    max_grad_norm: float = 0.3
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Add resume training parameters
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint directory
    save_safetensors: bool = True  # Better format for saving checkpoints
    
    # Enhanced training parameters for consistency
    weight_decay: float = 0.03             # Added weight decay
    num_cycles: int = 4                    # Number of LR cycles
    
    # Added focused attention params
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    
    # Memory optimization
    gradient_checkpointing: bool = True
    torch_compile: bool = True             # Use torch.compile for speed
    optim: str = "adamw_8bit"      # Memory efficient optimizer



    def __post_init__(self):
        super().__post_init__()
        self.gradient_checkpointing_kwargs = {
            "use_reentrant": False,
            "use_gradient_scaling": True    # Added for better stability
        }
        # If resuming, ensure we load the best model
        if self.resume_from_checkpoint:
            self.load_best_model_at_end = True

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
            "mixer_self_attention",  # Added for better attention
            "mixer_cross_attention", # Added for better attention
            "mixer_mlp",            # Added for better feature mixing
            "norm",
            "ln_f",
            "ln_1",
            "ln_2"
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


def train():
    # Initialize arguments
    model_args = ModelArguments()
    training_args = QLoRAConfig()
    
    # Add argument parsing for resume training and pause duration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Path to checkpoint directory to resume training from"
    )
    parser.add_argument(
        "--pause_minutes", type=int, default=45,
        help="Number of minutes to pause training at halfway point"
    )
    args, _ = parser.parse_known_args()
    
    if args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = args.resume_from_checkpoint
        logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    
    training_args.pause_minutes = args.pause_minutes
    logger.info(f"Training will pause for {training_args.pause_minutes} minutes halfway through.")
    
    # Updated dataset paths with names and proportions
    dataset_paths = [
        {
            "path": "/home/zahemen/datasets/sft_datasets/intro_conversations.jsonl",
            "name": "finsight_intro",
            "proportion": 1.0
        },
        # {
        #     "path": "/home/zahemen/datasets/sft_datasets/reddit_finance_conversations.jsonl",
        #     "name": "reddit_finance",
        #     "proportion": 1.0
        # },
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
        # {
        #     "path": "/home/zahemen/datasets/sft_datasets/finsight_combined_dataset_20250303_210949.jsonl",
        #     "name": "finsight_combined"
        # }

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
    
    # Initialize trainer with optimized settings and pause-resume callback
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.05
            ),
            PauseResumeCallback(pause_minutes=training_args.pause_minutes)
        ],
        data_collator=None,
    )

    # Train
    logger.info("Starting training")
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