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

def extract_text_from_conversation(data: Dict) -> List[Dict[str, str]]:
    """Extract messages from a conversation format"""
    try:
        messages = data.get('messages', [])
        if isinstance(messages, list) and all(
            isinstance(msg, dict) and 
            isinstance(msg.get('content', ''), str) and 
            isinstance(msg.get('role', ''), str)
            for msg in messages
        ):
            return messages
        return []
    except Exception as e:
        logger.warning(f"Error extracting messages from conversation: {e}")
        return []

def count_tokens_in_file(file_path: str, tokenizer) -> Dict:
    """Count tokens in a JSONL file using the model's tokenizer"""
    total_tokens = 0
    line_count = 0
    token_counts = []
    total_conversations = 0
    skipped = 0
    role_token_counts = {
        'system': [],
        'user': [],
        'assistant': []
    }
    
    try:
        logger.info(f"Processing conversations in {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    messages = extract_text_from_conversation(data)
                    
                    if messages:
                        # Count tokens in each message separately
                        conversation_tokens = 0
                        for msg in messages:
                            role = msg.get('role', '')
                            content = msg.get('content', '').strip()
                            
                            if content:
                                # Apply chat template to get accurate token count
                                formatted_msg = tokenizer.apply_chat_template(
                                    [msg],
                                    tokenize=False,
                                    add_generation_prompt=False
                                )
                                tokens = len(tokenizer.encode(formatted_msg))
                                
                                conversation_tokens += tokens
                                if role in role_token_counts:
                                    role_token_counts[role].append(tokens)
                        
                        total_tokens += conversation_tokens
                        token_counts.append(conversation_tokens)
                        line_count += 1
                        total_conversations += 1
                        
                    else:
                        skipped += 1
                        
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {file_path}")
                    skipped += 1
                    continue
                    
        # Calculate statistics for each role
        role_stats = {}
        for role, counts in role_token_counts.items():
            if counts:
                role_stats[role] = {
                    'avg': np.mean(counts),
                    'min': min(counts),
                    'max': max(counts),
                    'median': np.median(counts),
                    'total': sum(counts)
                }
            
        return {
            "total_tokens": total_tokens,
            "avg_tokens_per_conv": total_tokens / line_count if line_count > 0 else 0,
            "line_count": line_count,
            "token_distribution": {
                "min": min(token_counts) if token_counts else 0,
                "max": max(token_counts) if token_counts else 0,
                "median": np.median(token_counts) if token_counts else 0,
                "p95": np.percentile(token_counts, 95) if token_counts else 0
            },
            "role_statistics": role_stats,
            "conversations": total_conversations,
            "skipped": skipped
        }
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {}

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
            "path": "/home/zahemen/datasets/financial_defs_large_sft.jsonl",
            "name": "financial_definitions",
        }
    ]
    
    # Load the same tokenizer used in training
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # Update this to your model
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Enhanced table with token breakdown
    table = Table(title="Dataset Token Statistics")
    table.add_column("Dataset", style="cyan")
    table.add_column("Total Tokens", style="green")
    table.add_column("Conversations", style="blue")
    table.add_column("Avg Tokens/Conv", style="yellow")
    table.add_column("System Avg", style="magenta")
    table.add_column("User Avg", style="magenta")
    table.add_column("Assistant Avg", style="magenta")
    table.add_column("95th %ile", style="red")
    
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
        
        table.add_row(
            name,
            f"{stats['total_tokens']:,}",
            f"{stats['conversations']:,}",
            f"{stats['avg_tokens_per_conv']:.1f}",
            f"{stats['role_statistics'].get('system', {}).get('avg', 0):.1f}",
            f"{stats['role_statistics'].get('user', {}).get('avg', 0):.1f}",
            f"{stats['role_statistics'].get('assistant', {}).get('avg', 0):.1f}",
            f"{stats['token_distribution']['p95']:,.0f}"
        )
    
    console.print(table)
    
    # Add detailed token analysis
    logger.info("\nDetailed Token Analysis:")
    for dataset in dataset_paths:
        name = dataset["name"]
        stats = count_tokens_in_file(dataset["path"], tokenizer)
        
        console.print(f"\n[cyan]{name}[/cyan] Role Breakdown:")
        for role, role_stats in stats['role_statistics'].items():
            console.print(f"  [yellow]{role}[/yellow]:")
            console.print(f"    Average: {role_stats['avg']:.1f}")
            console.print(f"    Range: {role_stats['min']}-{role_stats['max']}")
            console.print(f"    Total: {role_stats['total']:,}")
    
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