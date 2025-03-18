#!/usr/bin/env python3
"""
Script to automate the processing pipeline for finance QA files:
1. Fix formatting issues in the QA files
2. Extract conversations from the fixed files
"""

import os
import logging
import argparse
import subprocess
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")
console = Console()

def process_qa_files(
    input_file: str,
    output_dir: str = "/home/zahemen/datasets/sft_datasets",
    min_turns: int = 1,
    max_turns: int = 8,
    num_conversations: int = 4000  # Updated default to 4000 as requested
) -> None:
    """
    Process QA files through the entire pipeline.
    
    Args:
        input_file: Path to the input file with questions and answers
        output_dir: Directory to save output files
        min_turns: Minimum number of turns in generated conversations
        max_turns: Maximum number of turns in generated conversations
        num_conversations: Target number of conversations to generate
    """
    try:
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Fix the format of the input file
        logger.info("Step 1: Fixing question format...")
        fixed_file = input_path.with_suffix('.fixed.txt')
        
        fix_cmd = [
            "python", 
            os.path.join(Path(__file__).parent, "fix_question_format.py"),
            str(input_path),
            "--output", str(fixed_file)
        ]
        
        result = subprocess.run(fix_cmd, check=True, capture_output=True, text=True)
        logger.info("Format fixing completed")
        
        # Step 2: Extract conversations
        logger.info("Step 2: Extracting conversations...")
        output_jsonl = output_path / f"{input_path.stem}_conversations.jsonl"
        
        extract_cmd = [
            "python",
            os.path.join(Path(__file__).parent, "extract_advanced_finance_conversations.py"),
            "--input", str(fixed_file),
            "--output", str(output_jsonl),
            "--min-turns", str(min_turns),
            "--max-turns", str(max_turns),
            "--num-conversations", str(num_conversations)
        ]
        
        result = subprocess.run(extract_cmd, check=True, capture_output=True, text=True)
        logger.info(f"Conversation extraction completed. Output: {output_jsonl}")
        
        # Show success message
        console.print(Panel(
            f"[green]Processing completed successfully![/green]\n\n"
            f"Fixed file: [cyan]{fixed_file}[/cyan]\n"
            f"Generated conversations: [cyan]{output_jsonl}[/cyan]",
            title="QA Processing Pipeline",
            border_style="green"
        ))
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Process failed with error code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Process QA files through fixing and conversation extraction"
    )
    
    parser.add_argument(
        "input_file", 
        type=str,
        help="Path to the input QA file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="/home/zahemen/datasets/sft_datasets",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--min-turns",
        type=int,
        default=1,  # Set to 1 to handle files with few QA pairs
        help="Minimum number of turns in generated conversations"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum number of turns in generated conversations"
    )
    
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=1000,
        help="Target number of conversations to generate"
    )
    
    args = parser.parse_args()
    
    process_qa_files(
        args.input_file,
        args.output_dir,
        args.min_turns,
        args.max_turns,
        args.num_conversations
    )

if __name__ == "__main__":
    main()
