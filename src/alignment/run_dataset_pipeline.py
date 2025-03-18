#!/usr/bin/env python3
import logging
import argparse
from pathlib import Path
import time
import sys
import os
from typing import List, Dict, Any
import json
import traceback
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add parent directory to path to ensure imports work
sys.path.append(str(Path(__file__).parent))

# Import dataset generators
from extract_financial_definitions import FinancialDefinitionsExtractor
from prepare_definition_dataset import DefinitionsDatasetGenerator
from extract_finance_conversations import FinanceConversationExtractor
from create_intro_dataset import IntroDatasetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")
console = Console()

class DatasetPipeline:
    """Pipeline for generating and processing financial conversation datasets."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the pipeline with command line arguments."""
        self.args = args
        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.stats = {}
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results summary dictionary
        self.results = {
            "statistics": {},
            "files_generated": [],
            "errors": []
        }
        
        # Verify critical input paths
        self._verify_critical_paths()
        
    def _verify_critical_paths(self):
        """Verify critical input paths exist and report any issues."""
        paths_to_check = {
            "Enhanced Q&A directory": self.data_dir / "enhanced_q_and_a",
            "Finance questions file": self.data_dir / "enhanced_q_and_a/finance_questions.txt",
        }
        
        # Add optional paths if those components were requested
        if self.args.company_qa:
            paths_to_check["Financial-QA dataset"] = self.data_dir / "Financial-QA-10k.csv"
        
        if self.args.reddit:
            paths_to_check["Reddit dataset"] = self.data_dir / "reddit-finance-250k/Data.jsonl"
        
        if self.args.advanced_finance:
            paths_to_check["Advanced finance questions"] = self.data_dir / "advanced_finance_questions.txt"

        # Check paths
        missing_paths = []
        for name, path in paths_to_check.items():
            if not path.exists():
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            console.print("[bold yellow]Warning: Some required input paths are missing:[/]")
            for missing in missing_paths:
                console.print(f"[yellow]  • {missing}[/]")
            console.print("")
        
    def run_intro_dataset(self) -> Path:
        """Generate introduction conversation dataset."""
        console.rule("[bold cyan]Generating Introduction Conversations Dataset")
        
        try:
            output_file = self.output_dir / "intro_conversations.jsonl"
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("[cyan]Generating intro conversations...", total=1)
                
                generator = IntroDatasetGenerator(str(output_file))
                generator.create_dataset(num_conversations=self.args.num_intro)
                
                progress.update(task, completed=1)
                
            # Count number of records generated
            count = self.count_jsonl_records(output_file)
            self.stats["intro_conversations"] = count
            self.results["files_generated"].append({"file": str(output_file), "records": count})
            
            logger.info(f"✅ Introduction conversations dataset generated: {output_file} ({count:,} records)")
            return output_file
            
        except Exception as e:
            error_msg = f"Error generating introduction conversations dataset: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())  # Add detailed traceback
            self.results["errors"].append(error_msg)
            return None
    
    def run_financial_definitions(self) -> Path:
        """Extract financial definitions and generate training dataset."""
        console.rule("[bold cyan]Processing Financial Definitions")
        
        definitions_file = self.output_dir / "financial_definitions.jsonl"
        qa_dataset_file = self.output_dir / "financial_definitions_dataset.jsonl"
        
        try:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn()
            ) as progress:
                # Step 1: Extract definitions
                extract_task = progress.add_task("[green]Extracting definitions...", total=1)
                
                extractor = FinancialDefinitionsExtractor(
                    input_dir=str(self.data_dir / "enhanced_q_and_a"),
                    output_file=str(definitions_file),
                    exclude_files=["finance_questions.txt"]
                )
                extractor.run()
                progress.update(extract_task, completed=1)
                
                # Count extracted definitions
                def_count = self.count_jsonl_records(definitions_file)
                self.stats["financial_definitions"] = def_count
                self.results["files_generated"].append({"file": str(definitions_file), "records": def_count})
                
                # Step 2: Generate QA dataset
                dataset_task = progress.add_task("[blue]Generating definitions dataset...", total=1)
                
                generator = DefinitionsDatasetGenerator(
                    definitions_dir=str(self.data_dir / "enhanced_q_and_a"),
                    output_file=str(qa_dataset_file),
                    num_samples=self.args.num_definitions
                )
                generator.run()
                progress.update(dataset_task, completed=1)
                
                # Count generated samples
                dataset_count = self.count_jsonl_records(qa_dataset_file)
                self.stats["financial_definitions_qa"] = dataset_count
                self.results["files_generated"].append({"file": str(qa_dataset_file), "records": dataset_count})
                
            logger.info(f"✅ Financial definitions processed: {def_count:,} definitions extracted, {dataset_count:,} QA samples generated")
            return qa_dataset_file
            
        except Exception as e:
            error_msg = f"Error processing financial definitions: {e}"
            logger.error(error_msg)
            self.results["errors"].append(error_msg)
            return None
    
    def run_finance_conversations(self) -> Path:
        """Extract and process finance conversations."""
        console.rule("[bold cyan]Processing Finance Conversations")
        
        output_file = self.output_dir / "finance_conversations.jsonl"
        
        try:
            # First verify the input file exists
            input_file = self.data_dir / "enhanced_q_and_a/finance_questions.txt"
            if not input_file.exists():
                raise FileNotFoundError(f"Finance questions file not found: {input_file}")
            
            # Create the extractor with explicit paths and set use_progress=False to avoid nested progress bars
            extractor = FinanceConversationExtractor(
                input_file=str(input_file),
                output_file=str(output_file),
                min_turns=self.args.min_turns,
                max_turns=self.args.max_turns,
                max_reuses=25,  # Allow more reuse to generate sufficient samples
                target_conversations=self.args.num_finance_conversations,
                use_progress=False  # Disable internal progress bars
            )
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("[green]Generating finance conversations...", total=1)
                
                # Run with explicit argument to ensure we're using the correct target
                extractor.run(self.args.num_finance_conversations)
                
                progress.update(task, completed=1)
                
            # Verify the output was actually created
            if not output_file.exists():
                raise FileNotFoundError(f"Failed to generate finance conversations output file: {output_file}")
                
            # Count number of records generated
            count = self.count_jsonl_records(output_file)
            if count == 0:
                raise ValueError("Finance conversations file was created but contains zero records")
                
            self.stats["finance_conversations"] = count
            self.results["files_generated"].append({"file": str(output_file), "records": count})
            
            logger.info(f"✅ Finance conversations dataset generated: {output_file} ({count:,} records)")
            return output_file
            
        except Exception as e:
            error_msg = f"Error generating finance conversations: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())  # Add detailed traceback
            self.results["errors"].append(error_msg)
            return None
    
    def run_company_qa(self) -> Path:
        """Process company Q&A dataset."""
        console.rule("[bold cyan]Processing Company Q&A Dataset")
        
        output_file = self.output_dir / "company_conversations.jsonl"
        
        try:
            # Dynamically import prepare_company_qa to avoid circular imports
            from prepare_company_qa import FinanceQAProcessor
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("[cyan]Processing company Q&A...", total=1)
                
                processor = FinanceQAProcessor(
                    dataset_path=self.data_dir / "Financial-QA-10k.csv", 
                    num_samples=self.args.num_company_qa
                )
                processor.process_dataset(output_file)
                
                progress.update(task, completed=1)
                
            # Count number of records generated
            count = self.count_jsonl_records(output_file)
            self.stats["company_qa"] = count
            self.results["files_generated"].append({"file": str(output_file), "records": count})
            
            logger.info(f"✅ Company Q&A dataset generated: {output_file} ({count:,} records)")
            return output_file
            
        except Exception as e:
            error_msg = f"Error processing company Q&A: {e}"
            logger.error(error_msg)
            self.results["errors"].append(error_msg)
            return None
    
    def run_reddit_processing(self) -> Path:
        """Process Reddit data."""
        console.rule("[bold cyan]Processing Reddit Financial Data")
        
        output_file = self.output_dir / "reddit_finance_conversations.jsonl"
        
        try:
            # Check if input file exists
            reddit_input = self.data_dir / "reddit-finance-250k/Data.jsonl"
            reddit_starters = self.data_dir / "reddit-finance-250k/conv_starter_pairs.txt"
            
            if not reddit_input.exists():
                logger.warning(f"Reddit data file not found: {reddit_input}")
                logger.warning("Skipping Reddit data processing.")
                return None
                
            # Import datasetcleaner to avoid circular imports
            from prepare_reddit_data import DatasetCleaner
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("[magenta]Processing Reddit data...", total=1)
                
                cleaner = DatasetCleaner(
                    input_file=str(reddit_input),
                    output_file=str(output_file),
                    conv_starters_file=str(reddit_starters),
                    silent_warnings=True
                )
                cleaner.process_dataset()
                
                progress.update(task, completed=1)
                
            # Count number of records generated
            count = self.count_jsonl_records(output_file)
            self.stats["reddit_data"] = count
            self.results["files_generated"].append({"file": str(output_file), "records": count})
            
            logger.info(f"✅ Reddit dataset generated: {output_file} ({count:,} records)")
            return output_file
            
        except Exception as e:
            error_msg = f"Error processing Reddit data: {e}"
            logger.error(error_msg)
            self.results["errors"].append(error_msg)
            return None
    
    def run_advanced_finance_conversations(self) -> Path:
        """Generate advanced finance conversation dataset."""
        console.rule("[bold cyan]Generating Advanced Finance Conversations Dataset")
        
        output_file = self.output_dir / "advanced_finance_conversations.jsonl"
        
        try:
            # First verify the input file exists
            input_file = self.data_dir / "advanced_finance_questions.txt"
            if not input_file.exists():
                raise FileNotFoundError(f"Advanced finance questions file not found: {input_file}")
            
            # Import the process_qa_files function from process_advanced_qa_files
            from process_advanced_qa_files import process_qa_files
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("[green]Generating advanced finance conversations...", total=1)
                
                # Run the processing function
                process_qa_files(
                    input_file=str(input_file),
                    output_dir=str(self.output_dir),
                    min_turns=self.args.min_turns,
                    max_turns=self.args.max_turns,
                    num_conversations=self.args.num_advanced_finance_conversations
                )
                
                progress.update(task, completed=1)
            
            # Verify the output was created (using the expected naming convention)
            expected_output = self.output_dir / f"advanced_finance_conversations.jsonl"
            if not expected_output.exists():
                raise FileNotFoundError(f"Failed to generate advanced finance conversations output file: {expected_output}")
            
            # If file exists but has different name than our output_file, copy/rename it
            if expected_output != output_file and expected_output.exists():
                import shutil
                shutil.copy2(expected_output, output_file)
            
            # Count number of records generated
            count = self.count_jsonl_records(output_file)
            if count == 0:
                raise ValueError("Advanced finance conversations file was created but contains zero records")
            
            self.stats["advanced_finance_conversations"] = count
            self.results["files_generated"].append({"file": str(output_file), "records": count})
            
            logger.info(f"✅ Advanced finance conversations dataset generated: {output_file} ({count:,} records)")
            return output_file
            
        except Exception as e:
            error_msg = f"Error generating advanced finance conversations: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())  # Add detailed traceback
            self.results["errors"].append(error_msg)
            return None
    
    def merge_datasets(self, input_files: List[Path]) -> Path:
        """Merge all generated datasets into one consolidated file."""
        console.rule("[bold cyan]Merging Datasets")
        
        # Filter out None values
        input_files = [f for f in input_files if f is not None and f.exists()]
        if not input_files:
            logger.warning("No datasets available to merge.")
            return None
            
        output_file = self.output_dir / f"finsight_combined_dataset_{self.timestamp}.jsonl"
        
        try:
            total_records = 0
            written_records = 0
            
            # Count total records first
            for file_path in input_files:
                count = self.count_jsonl_records(file_path)
                logger.info(f"Adding {count:,} records from {file_path.name}")
                total_records += count
            
            with open(output_file, 'w', encoding='utf-8') as outfile:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn()
                ) as progress:
                    merge_task = progress.add_task("[yellow]Merging datasets...", total=total_records)
                    
                    for file_path in input_files:
                        file_name = file_path.name
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                for line_num, line in enumerate(infile, 1):
                                    try:
                                        # Parse the JSON to ensure it's valid
                                        entry = json.loads(line)
                                        
                                        # Add dataset source metadata if not present
                                        if "metadata" not in entry:
                                            entry["metadata"] = {}
                                        
                                        # Add source file information
                                        entry["metadata"]["source_file"] = file_name
                                        
                                        # Ensure "conversation_id" is present in metadata
                                        if "id" in entry and "conversation_id" not in entry["metadata"]:
                                            entry["metadata"]["conversation_id"] = entry["id"]
                                            # Remove the top-level id field to maintain consistent schema
                                            if "id" in entry:
                                                del entry["id"]
                                        
                                        # Convert metadata arrays to strings
                                        for key, value in entry["metadata"].items():
                                            if isinstance(value, list):
                                                # Join array elements into a comma-separated string
                                                entry["metadata"][key] = ", ".join(str(item) for item in value)
                                            elif value is None or value == []:
                                                # Replace None values with empty strings
                                                entry["metadata"][key] = ""
                                        
                                        # Write back
                                        outfile.write(json.dumps(entry) + '\n')
                                        written_records += 1
                                        progress.update(merge_task, advance=1)
                                        
                                    except json.JSONDecodeError:
                                        logger.warning(f"Invalid JSON in file {file_name} at line {line_num}")
                                    except Exception as e:
                                        logger.warning(f"Error processing line {line_num} in {file_name}: {e}")
                                        
                        except Exception as e:
                            logger.error(f"Error processing file {file_name}: {e}")
            
            self.stats["merged_dataset"] = written_records
            self.results["files_generated"].append({"file": str(output_file), "records": written_records})
            
            logger.info(f"✅ Merged dataset created: {output_file} ({written_records:,} total records)")
            return output_file
            
        except Exception as e:
            error_msg = f"Error merging datasets: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())  # Add detailed traceback
            self.results["errors"].append(error_msg)
            return None
    
    def count_jsonl_records(self, file_path: Path) -> int:
        """Count the number of records in a JSONL file."""
        try:
            if not file_path or not file_path.exists():
                return 0
                
            count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    count += 1
            return count
        except Exception as e:
            logger.error(f"Error counting records in {file_path}: {e}")
            return 0
    
    def display_summary(self):
        """Display a summary of the dataset generation results."""
        console.rule("[bold green]Dataset Generation Complete")
        
        # Create statistics table
        stats_table = Table(title="Dataset Statistics", box=box.ROUNDED)
        stats_table.add_column("Dataset Type", style="cyan")
        stats_table.add_column("Records", style="magenta")
        
        total_records = 0
        
        for dataset_name, count in self.stats.items():
            stats_table.add_row(dataset_name.replace("_", " ").title(), f"{count:,}")
            total_records += count
        
        stats_table.add_row("Total", f"{total_records:,}")
        
        # Create files table
        files_table = Table(title="Generated Files", box=box.ROUNDED)
        files_table.add_column("File Path", style="green")
        files_table.add_column("Records", style="blue")
        
        for file_info in self.results["files_generated"]:
            files_table.add_row(file_info["file"], f"{file_info['records']:,}")
        
        # Display tables
        console.print("\n")
        console.print(stats_table)
        console.print("\n")
        console.print(files_table)
        
        # Display errors if any
        if self.results["errors"]:
            console.print("\n")
            error_panel = Panel(
                "\n".join([f"• {error}" for error in self.results["errors"]]),
                title="[bold red]Errors",
                border_style="red"
            )
            console.print(error_panel)
        
        # Save summary to file
        summary_file = self.output_dir / f"generation_summary_{self.timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n[bold green]Summary saved to:[/] {summary_file}")
    
    def run_pipeline(self) -> None:
        """Run the complete dataset generation pipeline."""
        start_time = time.time()
        console.rule("[bold green]Starting Financial Dataset Generation Pipeline")
        
        # Print configuration info
        console.print(f"[cyan]Input directory:[/] {self.data_dir}")
        console.print(f"[cyan]Output directory:[/] {self.output_dir}")
        
        generated_files = []
        
        # Track success/failure of each component
        component_status = {}
        
        # Run each component based on command-line flags
        if self.args.intro:
            result = self.run_intro_dataset()
            generated_files.append(result)
            component_status["intro"] = result is not None
        
        if self.args.definitions:
            result = self.run_financial_definitions()
            generated_files.append(result)
            component_status["definitions"] = result is not None
        
        if self.args.finance_convos:
            result = self.run_finance_conversations()
            generated_files.append(result)
            component_status["finance_convos"] = result is not None
        
        if self.args.advanced_finance:
            result = self.run_advanced_finance_conversations()
            generated_files.append(result)
            component_status["advanced_finance"] = result is not None
        
        if self.args.company_qa:
            result = self.run_company_qa()
            generated_files.append(result)
            component_status["company_qa"] = result is not None
        
        if self.args.reddit:
            result = self.run_reddit_processing()
            generated_files.append(result)
            component_status["reddit"] = result is not None
        
        # Merge datasets if requested 
        merged_result = None
        if self.args.merge and any(f is not None for f in generated_files):
            merged_result = self.merge_datasets(generated_files)
            component_status["merge"] = merged_result is not None
        
        # Display summary statistics
        self.display_summary()
        
        # Show completion time
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        console.print(f"\n[bold]Total processing time:[/] {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Show success/failure overview
        status_table = Table(title="Component Status", box=box.ROUNDED)
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="bold")
        
        for component, success in component_status.items():
            status = "[green]Success" if success else "[red]Failed"
            status_table.add_row(component.replace("_", " ").title(), status)
            
        console.print("\n")
        console.print(status_table)
        console.print("\n[bold green]Pipeline execution complete![/]\n")

def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Financial Dataset Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main directories
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="/home/zahemen/datasets",
        help="Root directory containing input datasets"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/home/zahemen/datasets/sft_datasets",
        help="Directory to save generated datasets"
    )
    
    # Component flags
    parser.add_argument(
        "--intro", 
        action="store_true",
        help="Generate introduction conversations dataset"
    )
    
    parser.add_argument(
        "--definitions", 
        action="store_true",
        help="Process financial definitions dataset"
    )
    
    parser.add_argument(
        "--finance-convos", 
        action="store_true",
        help="Generate finance conversations dataset"
    )
    
    parser.add_argument(
        "--advanced-finance", 
        action="store_true",
        help="Generate advanced finance conversations dataset"
    )
    
    parser.add_argument(
        "--company-qa", 
        action="store_true",
        help="Process company Q&A dataset"
    )
    
    parser.add_argument(
        "--reddit", 
        action="store_true",
        help="Process Reddit financial data"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all dataset generation components"
    )
    
    parser.add_argument(
        "--merge", 
        action="store_true",
        help="Merge all generated datasets into one file"
    )
    
    # Dataset size parameters
    parser.add_argument(
        "--num-intro", 
        type=int, 
        default=1000,
        help="Number of introduction conversations to generate"
    )
    
    parser.add_argument(
        "--num-definitions", 
        type=int, 
        default=2000,
        help="Number of definition QA pairs to generate"
    )
    
    parser.add_argument(
        "--num-finance-conversations", 
        type=int, 
        default=2000,
        help="Number of finance conversations to generate"
    )
    
    parser.add_argument(
        "--num-advanced-finance-conversations", 
        type=int, 
        default=4000,
        help="Number of advanced finance conversations to generate"
    )
    
    parser.add_argument(
        "--num-company-qa", 
        type=int, 
        default=2000,
        help="Number of company Q&A samples to generate"
    )
    
    parser.add_argument(
        "--min-turns", 
        type=int, 
        default=3,
        help="Minimum number of turns in conversations"
    )
    
    parser.add_argument(
        "--max-turns", 
        type=int, 
        default=10,
        help="Maximum number of turns in conversations"
    )
    
    args = parser.parse_args()
    
    # If --all is specified, set all component flags to True
    if args.all:
        args.intro = True
        args.definitions = True
        args.finance_convos = True
        args.advanced_finance = True
        args.company_qa = True
        args.reddit = True
        args.merge = True
    
    # If no components specified, show help and exit
    if not (args.intro or args.definitions or args.finance_convos or 
            args.advanced_finance or args.company_qa or args.reddit):
        parser.print_help()
        console.print("\n[yellow]No dataset components selected. Use --all to run everything.[/]")
        sys.exit(1)
    
    # Run the pipeline
    pipeline = DatasetPipeline(args)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
