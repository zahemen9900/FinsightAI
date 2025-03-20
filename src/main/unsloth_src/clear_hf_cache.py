#!/usr/bin/env python3
"""
Helper script to clear the Huggingface cache for specific models
to ensure they're re-downloaded fresh
"""

import os
import shutil
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import track

console = Console()

def clear_model_cache(model_name):
    """Clear the cache for a specific model"""
    # Convert model name to cache directory format
    cache_name = model_name.replace("/", "--")
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_dirs = list(Path(cache_dir).glob(f"models--{cache_name}*"))
    
    # Also look for other potential matches
    if "SmolLM2" in model_name:
        llama_dirs = list(Path(cache_dir).glob("models--unsloth--llama*"))
        model_dirs.extend(llama_dirs)
    
    if not model_dirs:
        console.print(f"[yellow]No cache found for {model_name}[/yellow]")
        return
    
    for model_dir in model_dirs:
        try:
            console.print(f"[yellow]Removing cache directory:[/yellow] {model_dir}")
            shutil.rmtree(model_dir)
            console.print(f"[green]Successfully removed cache for {model_dir.name}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to clear cache for {model_dir}: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description="Clear Huggingface model cache")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                        help="Model name to clear from cache")
    parser.add_argument("--clear_all_unsloth", action="store_true",
                        help="Clear all Unsloth model caches")
    
    args = parser.parse_args()
    
    console.print("[bold]Huggingface Cache Cleaner[/bold]")
    
    if args.clear_all_unsloth:
        console.print("[yellow]Clearing all Unsloth model caches...[/yellow]")
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        unsloth_dirs = list(Path(cache_dir).glob("models--unsloth--*"))
        
        for unsloth_dir in track(unsloth_dirs, description="Clearing Unsloth caches"):
            try:
                shutil.rmtree(unsloth_dir)
                console.print(f"[green]Removed {unsloth_dir.name}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to remove {unsloth_dir.name}: {e}[/red]")
    else:
        clear_model_cache(args.model)
    
    console.print("[bold green]Cache cleaning completed![/bold green]")
    console.print("You can now re-run your training script to download a fresh copy of the model.")

if __name__ == "__main__":
    main()
