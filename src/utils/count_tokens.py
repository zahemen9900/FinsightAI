from transformers import AutoTokenizer
import json
from pathlib import Path
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from typing import List, Dict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")
console = Console()

def extract_text_from_conversation(data: Dict) -> str:
    """Extract all text from a conversation format"""
    try:
        messages = data.get('messages', [])
        texts = []
        
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                content = msg['content']
                if isinstance(content, str) and content.strip():
                    texts.append(content.strip())
        
        return "\n".join(texts)
    except Exception as e:
        logger.warning(f"Error extracting text from conversation: {e}")
        return ""

def count_tokens_in_file(file_path: str, tokenizer) -> Dict:
    """Count tokens in a JSONL file containing conversations"""
    total_tokens = 0
    line_count = 0
    token_counts = []
    total_conversations = 0
    skipped = 0
    
    try:
        logger.info(f"Processing all conversations in {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = extract_text_from_conversation(data)
                    
                    if text:
                        tokens = len(tokenizer.encode(text))
                        total_tokens += tokens
                        token_counts.append(tokens)
                        line_count += 1
                        total_conversations += 1
                        
                        # Log progress every 1000 conversations
                        # if total_conversations % 1000 == 0:
                        #     logger.info(f"Processed {total_conversations} conversations...")
                    else:
                        skipped += 1
                        
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {file_path}")
                    skipped += 1
                    continue
                    
        # Log final processing details
        logger.info(f"Finished processing {total_conversations} conversations")
        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid entries")
            
        return {
            "total_tokens": total_tokens,
            "avg_tokens": total_tokens / line_count if line_count > 0 else 0,
            "line_count": line_count,
            "token_distribution": {
                "min": min(token_counts) if token_counts else 0,
                "max": max(token_counts) if token_counts else 0,
                "median": np.median(token_counts) if token_counts else 0
            },
            "conversations": total_conversations,
            "skipped": skipped
        }
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {
            "total_tokens": 0,
            "avg_tokens": 0,
            "line_count": 0,
            "token_distribution": {"min": 0, "max": 0, "median": 0},
            "conversations": 0,
            "skipped": 0
        }

def main():
    # Your dataset paths
    dataset_paths = [
        {
            "path": "/home/zahemen/datasets/reddit-finance-250k/sft_cleaned_data.jsonl",
            "name": "reddit_finance",
        },
        {
            "path": "/home/zahemen/datasets/finance_qa_conversations.jsonl",
            "name": "finance_qa",
        },
        {
            "path": "/home/zahemen/datasets/intro_conversations.jsonl",
            "name": "intro_conversations",
        }
    ]
    
    # Load the same tokenizer used in training
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # Update this to your model
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Enhanced table with more columns
    table = Table(title="Dataset Token Statistics")
    table.add_column("Dataset", style="cyan")
    table.add_column("Total Tokens", style="green")
    table.add_column("Conversations", style="blue")
    table.add_column("Valid %", style="yellow")
    table.add_column("Avg Tokens/Conv", style="yellow")
    table.add_column("Min Tokens", style="magenta")
    table.add_column("Max Tokens", style="magenta")
    table.add_column("Median", style="magenta")
    
    grand_total = 0
    total_conversations = 0
    
    for dataset in dataset_paths:
        name = dataset["name"]
        path = dataset["path"]
        
        if not Path(path).exists():
            logger.error(f"Dataset not found: {path}")
            continue
            
        logger.info(f"\nAnalyzing {name}...")
        stats = count_tokens_in_file(path, tokenizer)  # Removed sample_size parameter
        
        grand_total += stats["total_tokens"]
        total_conversations += stats["conversations"]
        
        valid_percent = (stats["conversations"] / (stats["conversations"] + stats["skipped"]) * 100 
                        if stats["conversations"] + stats["skipped"] > 0 else 0)
        
        table.add_row(
            name,
            f"{stats['total_tokens']:,}",
            f"{stats['conversations']:,}",
            f"{valid_percent:.1f}%",
            f"{stats['avg_tokens']:.1f}",
            f"{stats['token_distribution']['min']:,}",
            f"{stats['token_distribution']['max']:,}",
            f"{stats['token_distribution']['median']:,}"
        )
    
    console.print(table)
    
    # Enhanced statistics
    logger.info(f"\nTotal tokens across all datasets: {grand_total:,}")
    logger.info(f"Total conversations: {total_conversations:,}")
    logger.info(f"Average tokens per conversation: {(grand_total/total_conversations if total_conversations > 0 else 0):.1f}")
    
    # Training cost estimates
    tokens_per_second = 1000  # Approximate throughput for A100
    training_hours = (grand_total/tokens_per_second/3600)
    logger.info(f"Approximate training time on A100 GPU (at {tokens_per_second:,} tokens/sec):")
    logger.info(f"  Hours: {training_hours:.2f}")
    logger.info(f"  Days: {training_hours/24:.2f}")

if __name__ == "__main__":
    main()