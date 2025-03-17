#!/usr/bin/env python3
"""
Utility script to check and fix the format of advanced finance questions file
to ensure it can be properly processed by the extraction script.
"""

import re
import argparse
from pathlib import Path
import logging
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import track
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger('rich')
console = Console()

def fix_question_format(input_file: str, output_file: str = None, dryrun: bool = False) -> None:
    """
    Check and fix the format of questions file to ensure proper extraction.
    
    Args:
        input_file: Path to the input questions file
        output_file: Path to save the fixed file (defaults to input_file + ".fixed")
        dryrun: If True, only analyze without saving changes
    """
    input_path = Path(input_file)
    if not output_file:
        output_file = str(input_path.with_suffix('.fixed.txt'))
    output_path = Path(output_file)
    
    logger.info(f"Analyzing file: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple pattern to extract pairs of questions and answers
        qa_pairs = re.findall(r'Question\s+\d+:(.*?)Answer:(.*?)(?=Question \d+:|$)', content, re.DOTALL)
        
        logger.info(f"Found {len(qa_pairs)} question-answer pairs")
        
        # Show a sample of content
        if qa_pairs:
            logger.info("Sample of first question-answer pair:")
            sample = f"Question 1:{qa_pairs[0][0]}\nAnswer:{qa_pairs[0][1]}"
            console.print(sample[:500] + "..." if len(sample) > 500 else sample)
        
        # Rebuild the content with proper formatting
        fixed_content = ""
        for i, (question, answer) in enumerate(qa_pairs, 1):
            fixed_content += f"Question {i}:{question.strip()}\n\nAnswer:{answer.strip()}\n\n"
            fixed_content += "-" * 80 + "\n\n"
        
        # Save the fixed content
        if not dryrun:
            logger.info(f"Saving fixed file to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            logger.info("File successfully fixed and saved!")
        else:
            logger.info("Dry run - no changes saved")
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Check and fix the format of advanced finance questions file"
    )
    
    parser.add_argument(
        "input_file", 
        type=str,
        help="Path to the input questions file"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        help="Path to save the fixed file (defaults to input_file + '.fixed.txt')"
    )
    
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Only analyze without saving changes"
    )
    
    args = parser.parse_args()
    fix_question_format(args.input_file, args.output, args.dryrun)

if __name__ == "__main__":
    main()
