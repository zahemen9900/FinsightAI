#!/usr/bin/env python3
"""
Helper script to resume Unsloth training from checkpoints
"""

import os
import json
import argparse
import subprocess
import logging
from pathlib import Path
import gc
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import track
import sys

# Configure rich logging
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger("resume_unsloth")

def find_checkpoints(output_dir):
    """Find all checkpoints in the output directory with their metadata"""
    checkpoints = []
    output_path = Path(output_dir)
    
    console.print(f"[bold]Looking for checkpoints in:[/bold] {output_path}")
    
    # Look for checkpoint directories
    checkpoint_dirs = list(output_path.glob("checkpoint-*"))
    
    if not checkpoint_dirs:
        console.print("[yellow]No checkpoint directories found[/yellow]")
        return []
    
    for checkpoint_dir in track(checkpoint_dirs, description="Scanning checkpoints"):
        try:
            checkpoint = {"path": str(checkpoint_dir)}
            checkpoint["step"] = int(checkpoint_dir.name.split("-")[-1])
            
            # Check for required files
            trainer_state_path = checkpoint_dir / "trainer_state.json"
            if trainer_state_path.exists():
                with open(trainer_state_path, "r") as f:
                    state = json.load(f)
                    checkpoint["global_step"] = state.get("global_step")
                    checkpoint["epoch"] = state.get("epoch")
                    
                    # Get last loss value
                    log_history = state.get("log_history", [])
                    if log_history:
                        last_logs = log_history[-1]
                        loss_keys = ["loss", "eval_loss", "train_loss"]
                        for key in loss_keys:
                            if key in last_logs:
                                checkpoint["loss"] = last_logs[key]
                                break
            else:
                checkpoint["status"] = "incomplete"
                
            # Check for adapter model files which Unsloth LoRA uses
            adapter_exists = (checkpoint_dir / "adapter_model.safetensors").exists() or \
                            (checkpoint_dir / "adapter_model.bin").exists()
            
            # Check integrity
            if adapter_exists and trainer_state_path.exists():
                checkpoint["status"] = "complete"
            else:
                checkpoint["status"] = "incomplete"
                
            checkpoints.append(checkpoint)
        except Exception as e:
            console.print(f"[red]Error scanning checkpoint {checkpoint_dir}: {e}[/red]")
    
    # Sort by step
    checkpoints.sort(key=lambda x: x.get("step", 0))
    return checkpoints

def display_checkpoints(checkpoints):
    """Display checkpoints in a formatted table"""
    if not checkpoints:
        console.print("[yellow]No checkpoints found[/yellow]")
        return
        
    table = Table(title="Available Checkpoints")
    table.add_column("Step", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Epoch", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Path", justify="left")
    
    for ckpt in checkpoints:
        step = str(ckpt.get("step", "N/A"))
        status = ckpt.get("status", "unknown")
        status_color = "green" if status == "complete" else "red"
        epoch = f"{ckpt.get('epoch', 'N/A'):.2f}" if isinstance(ckpt.get('epoch'), float) else str(ckpt.get('epoch', 'N/A'))
        loss = f"{ckpt.get('loss', 'N/A'):.4f}" if isinstance(ckpt.get('loss'), (int, float)) else str(ckpt.get('loss', 'N/A'))
        
        table.add_row(
            step, 
            f"[{status_color}]{status}[/{status_color}]", 
            epoch,
            loss,
            ckpt["path"]
        )
    
    console.print(table)

def resume_training(checkpoint_path, batch_size=None, max_seq_length=None, extra_args=None):
    """Resume training from the specified checkpoint with proper JSON config"""
    # Create a temporary config file for resuming
    resume_config = {
        "model_name_or_path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # Will be loaded from checkpoint anyway
        "trust_remote_code": True,
        "force_download": True,  # Added to ensure it downloads the right model 
        "resume_from_checkpoint": checkpoint_path,
        "eval_strategy": "steps"  # Important: use eval_strategy, not evaluation_strategy
    }
    
    # Update with custom parameters if provided
    if batch_size:
        resume_config["per_device_train_batch_size"] = batch_size
        resume_config["per_device_eval_batch_size"] = batch_size
    
    if max_seq_length:
        resume_config["max_seq_length"] = max_seq_length
    
    # Write config to file
    config_path = "resume_config.json"
    with open(config_path, "w") as f:
        json.dump(resume_config, f, indent=2)
    
    console.print(f"\n[bold green]Resume configuration saved to {config_path}:[/bold green]")
    with open(config_path, "r") as f:
        console.print(f.read())
    
    # Build command
    cmd = [
        sys.executable,
        "src/main/train_unsloth.py",
        config_path
    ]
    
    # Add any additional arguments
    if extra_args:
        cmd.extend(extra_args)
    
    console.print(f"\n[bold green]Resuming training with command:[/bold green]")
    console.print(" ".join(cmd))
    
    confirm = input("\nConfirm resume (yes/no): ").lower()
    if confirm in ["yes", "y"]:
        # Clean up GPU memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        console.print("\n[bold]Starting training...[/bold]")
        subprocess.run(cmd)
    else:
        console.print("[yellow]Resume cancelled[/yellow]")

def check_gpu_memory():
    """Check and display available GPU memory"""
    if not torch.cuda.is_available():
        console.print("[yellow]CUDA not available - will run on CPU (very slow)[/yellow]")
        return
    
    try:
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        prop = torch.cuda.get_device_properties(device)
        total_memory = prop.total_memory / (1024**3)  # Convert to GB
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
        free_memory = total_memory - allocated
        
        console.print(f"[bold]GPU:[/bold] {gpu_name}")
        console.print(f"[bold]Total memory:[/bold] {total_memory:.2f} GB")
        console.print(f"[bold]Used memory:[/bold] {allocated:.2f} GB")
        console.print(f"[bold]Free memory:[/bold] {free_memory:.2f} GB")
        
        # Recommend batch size based on available memory
        if free_memory < 2:
            console.print("[bold red]Warning: Very limited GPU memory![/bold red]")
            console.print("Recommended batch size: 1, max_seq_length: 1024")
        elif free_memory < 4:
            console.print("[yellow]Limited GPU memory available[/yellow]")
            console.print("Recommended batch size: 2, max_seq_length: 2048")
        elif free_memory < 8:
            console.print("[green]Adequate GPU memory available[/green]")
            console.print("Recommended batch size: 4, max_seq_length: 2048")
        else:
            console.print("[bold green]Ample GPU memory available[/bold green]")
            console.print("Recommended batch size: 8, max_seq_length: 4096")
            
    except Exception as e:
        console.print(f"[yellow]Error checking GPU memory: {e}[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Resume Unsloth training from checkpoint")
    parser.add_argument("--output_dir", type=str, default="unsloth_output",
                        help="Directory containing the checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint")
    parser.add_argument("--resume_from", type=str,
                        help="Resume from a specific checkpoint number or path")
    parser.add_argument("--list", action="store_true",
                        help="Just list available checkpoints without resuming")
    parser.add_argument("--batch_size", type=int,
                        help="Override batch size for resuming")
    parser.add_argument("--max_seq_length", type=int,
                        help="Override sequence length for resuming")
    parser.add_argument("--check_gpu", action="store_true",
                        help="Check GPU memory and recommend settings")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER,
                        help="Extra arguments to pass to train_unsloth.py")
    parser.add_argument("--use_config", action="store_true",
                      help="Use JSON config approach for resumption (recommended)")
    
    args = parser.parse_args()
    
    # Check GPU memory if requested
    if args.check_gpu:
        check_gpu_memory()
    
    # Find checkpoints
    checkpoints = find_checkpoints(args.output_dir)
    
    # Display checkpoints
    if checkpoints:
        display_checkpoints(checkpoints)
    
    # If just listing checkpoints, exit here
    if args.list:
        return
    
    # Resume training if requested
    if args.resume or args.resume_from:
        checkpoint_to_resume = None
        
        if args.resume_from:
            # Check if it's a step number or path
            if args.resume_from.isdigit():
                step = int(args.resume_from)
                matching = [c for c in checkpoints if c.get("step") == step]
                if matching:
                    checkpoint_to_resume = matching[0]["path"]
                else:
                    console.print(f"[red]No checkpoint found for step {step}[/red]")
            else:
                # It's a path
                checkpoint_to_resume = args.resume_from
        elif args.resume:
            # Find the most recent complete checkpoint
            complete_checkpoints = [c for c in checkpoints if c.get("status") == "complete"]
            if complete_checkpoints:
                checkpoint_to_resume = max(complete_checkpoints, key=lambda x: x.get("step", 0))["path"]
            else:
                console.print("[red]No complete checkpoints found to resume from[/red]")
        
        if checkpoint_to_resume:
            # Always use the config approach (more reliable)
            resume_training(
                checkpoint_to_resume, 
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                extra_args=args.extra_args
            )

if __name__ == "__main__":
    main()
