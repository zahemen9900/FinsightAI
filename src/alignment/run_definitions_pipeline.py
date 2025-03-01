import logging
import argparse
from pathlib import Path
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

# Import our modules
from extract_financial_definitions import FinancialDefinitionsExtractor
from prepare_definition_dataset import DefinitionsDatasetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")

def main():
    parser = argparse.ArgumentParser(description="Generate financial definitions dataset for fine-tuning")
    
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="/home/zahemen/datasets/all_finance_definitions_in_txt",
        help="Directory containing text files with financial definitions"
    )
    
    parser.add_argument(
        "--defs-output", 
        type=str, 
        default="/home/zahemen/datasets/financial_definitions.jsonl",
        help="Output file for extracted definitions"
    )
    
    parser.add_argument(
        "--dataset-output", 
        type=str, 
        default="/home/zahemen/datasets/financial_definitions_dataset.jsonl",
        help="Output file for the generated dataset"
    )
    
    parser.add_argument(
        "--samples", 
        type=int, 
        default=3000,
        help="Number of samples to generate for the dataset"
    )
    
    args = parser.parse_args()
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    ) as progress:
        # Step 1: Extract definitions
        extract_task = progress.add_task("[green]Extracting definitions...", total=1)
        
        extractor = FinancialDefinitionsExtractor(
            input_dir=args.input_dir,
            output_file=args.defs_output
        )
        extractor.run()
        progress.update(extract_task, completed=1)
        
        # Step 2: Generate dataset
        dataset_task = progress.add_task("[blue]Generating dataset...", total=1)
        
        generator = DefinitionsDatasetGenerator(
            definitions_file=args.defs_output,
            output_file=args.dataset_output,
            num_samples=args.samples
        )
        generator.run()
        progress.update(dataset_task, completed=1)
        
    logger.info("✅ Financial definitions pipeline completed successfully!")
    logger.info(f"📝 Extracted definitions: {args.defs_output}")
    logger.info(f"📚 Generated dataset: {args.dataset_output}")
    logger.info(f"🔢 Total samples: {args.samples}")

if __name__ == "__main__":
    main()
