#!/usr/bin/env python3
"""
Helper script to list all valid training arguments for Unsloth training
"""

import sys
import logging
from dataclasses import fields
from rich.console import Console
from rich.table import Table

# Import your argument classes
from train import ModelArguments
from main.unsloth_src.train_unsloth import UnslothTrainingArguments

console = Console()

def print_dataclass_fields(cls, title):
    """Print all fields in a dataclass as a table"""
    table = Table(title=title)
    table.add_column("Argument Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Default Value", style="yellow")
    table.add_column("Required", style="red")
    
    for field in fields(cls):
        # Determine if field is required
        required = "Yes" if field.default == field.default_factory == ... else "No"
        # Get default value as string
        default = str(field.default) if field.default is not field.default_factory else ""
        if default == "...":
            default = "Required"
        # Add row
        table.add_row(field.name, str(field.type), default, required)
    
    console.print(table)

def main():
    """Print all valid argument fields for the training scripts"""
    console.print("[bold]Valid Arguments for Unsloth Training[/bold]")
    
    # Print fields for ModelArguments
    print_dataclass_fields(ModelArguments, "Model Arguments")
    
    # Print fields for UnslothTrainingArguments
    print_dataclass_fields(UnslothTrainingArguments, "Training Arguments")
    
    # Additional guidance
    console.print("\n[bold]JSON Configuration Example:[/bold]")
    console.print("""
{
    "model_name_or_path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "trust_remote_code": true,
    "output_dir": "unsloth_output",
    "per_device_train_batch_size": 2,
    "eval_strategy": "steps"
}
""")
    
    console.print("\n[italic]Note: All arguments should be at the top level, not nested under 'model_args' or other keys.[/italic]")

if __name__ == "__main__":
    main()
