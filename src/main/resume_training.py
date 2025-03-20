#!/usr/bin/env python3
"""
Resume training helper script that finds the best checkpoint to resume from
and restarts training with appropriate flags.
"""

import os
import json
import shutil
import argparse
import subprocess
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import sys
from typing import List, Dict, Optional, Tuple

# Configure logging
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger("resume_training")

def find_checkpoints(output_dir: str) -> List[Dict]:
    """Find all checkpoints in the output directory with their metadata."""
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
                    checkpoint["best_metric"] = state.get("best_metric")
                    checkpoint["loss"] = state.get("log_history", [{}])[-1].get("loss")
            else:
                checkpoint["status"] = "incomplete"
                
            # Check for optimizer and scheduler
            checkpoint["has_optimizer"] = (checkpoint_dir / "optimizer.pt").exists()
            checkpoint["has_scheduler"] = (checkpoint_dir / "scheduler.pt").exists() 
            
            # Check integrity
            if checkpoint["has_optimizer"] and checkpoint["has_scheduler"] and "global_step" in checkpoint:
                checkpoint["status"] = "complete"
            else:
                checkpoint["status"] = "incomplete"
                
            checkpoints.append(checkpoint)
        except Exception as e:
            console.print(f"[red]Error scanning checkpoint {checkpoint_dir}: {e}[/red]")
    
    # Sort by step
    checkpoints.sort(key=lambda x: x.get("step", 0))
    return checkpoints

def display_checkpoints(checkpoints: List[Dict]) -> None:
    """Display checkpoints in a formatted table."""
    if not checkpoints:
        console.print(Panel("[yellow]No checkpoints found[/yellow]", title="Checkpoints"))
        return
        
    table = Table(title="Available Checkpoints")
    table.add_column("Step", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Epoch", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Optimizer", justify="center")
    table.add_column("Scheduler", justify="center") 
    table.add_column("Path", justify="left")
    
    for ckpt in checkpoints:
        step = str(ckpt.get("step", "N/A"))
        status = ckpt.get("status", "unknown")
        status_color = "green" if status == "complete" else "red"
        epoch = f"{ckpt.get('epoch', 'N/A'):.2f}" if isinstance(ckpt.get('epoch'), float) else str(ckpt.get('epoch', 'N/A'))
        loss = f"{ckpt.get('loss', 'N/A'):.4f}" if isinstance(ckpt.get('loss'), float) else str(ckpt.get('loss', 'N/A'))
        
        optimizer = "✅" if ckpt.get("has_optimizer") else "❌"
        scheduler = "✅" if ckpt.get("has_scheduler") else "❌"
        
        table.add_row(
            step, 
            f"[{status_color}]{status}[/{status_color}]", 
            epoch,
            loss,
            optimizer,
            scheduler,
            ckpt["path"]
        )
    
    console.print(table)

def cleanup_checkpoints(checkpoints: List[Dict], keep_best: int = 3, keep_last: int = 1) -> None:
    """Clean up old or incomplete checkpoints."""
    if not checkpoints:
        return
        
    # Identify complete checkpoints
    complete_checkpoints = [c for c in checkpoints if c.get("status") == "complete"]
    incomplete_checkpoints = [c for c in checkpoints if c.get("status") != "complete"]
    
    to_delete = []
    
    # Mark incomplete checkpoints for deletion
    if incomplete_checkpoints:
        console.print(f"\n[yellow]Found {len(incomplete_checkpoints)} incomplete checkpoints[/yellow]")
        for ckpt in incomplete_checkpoints:
            to_delete.append(ckpt)
    
    # Keep only the best and most recent complete checkpoints
    if len(complete_checkpoints) > keep_best + keep_last:
        # Sort by loss (if available)
        if all("loss" in c for c in complete_checkpoints):
            best_checkpoints = sorted([c for c in complete_checkpoints if isinstance(c.get("loss"), (int, float))], 
                                      key=lambda x: x["loss"])[:keep_best]
        else:
            best_checkpoints = []
            
        # Sort by step for last checkpoints
        last_checkpoints = sorted(complete_checkpoints, key=lambda x: x.get("step", 0))[-keep_last:]
        
        # All checkpoints to keep
        keep_checkpoints = set([c["path"] for c in best_checkpoints + last_checkpoints])
        
        # Mark others for deletion
        for ckpt in complete_checkpoints:
            if ckpt["path"] not in keep_checkpoints:
                to_delete.append(ckpt)
    
    # Delete marked checkpoints
    if to_delete:
        console.print(f"\n[bold red]Will delete {len(to_delete)} checkpoints:[/bold red]")
        for ckpt in to_delete:
            console.print(f"  - {ckpt['path']}")
            
        confirm = input("\nConfirm deletion (yes/no): ").lower()
        if confirm in ["yes", "y"]:
            for ckpt in to_delete:
                try:
                    shutil.rmtree(ckpt["path"])
                    console.print(f"[green]Deleted:[/green] {ckpt['path']}")
                except Exception as e:
                    console.print(f"[red]Failed to delete {ckpt['path']}: {e}[/red]")
        else:
            console.print("[yellow]Deletion cancelled[/yellow]")

def resume_training(checkpoint_path: str, additional_args: List[str] = None, disable_deepspeed: bool = False) -> None:
    """Resume training from the specified checkpoint."""
    # Assemble command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        "src/main/train_qlora.py",  # Training script
        f"--resume_from_checkpoint={checkpoint_path}",  # Checkpoint to resume from
        "--checkpoint_cleanup"  # Add flag to clean up temporary checkpoints
    ]
    
    # Add disable_deepspeed flag for better resume stability
    if disable_deepspeed:
        cmd.append("--disable_deepspeed")
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    console.print(f"\n[bold green]Resuming training with command:[/bold green]")
    console.print(" ".join(cmd))
    
    confirm = input("\nConfirm resume (yes/no): ").lower()
    if confirm in ["yes", "y"]:
        console.print("\n[bold]Starting training...[/bold]")
        # Execute the command
        subprocess.run(cmd)
    else:
        console.print("[yellow]Resume cancelled[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Resume training helper")
    parser.add_argument("--output_dir", type=str, default="qlora_output", 
                        help="Training output directory to scan for checkpoints")
    parser.add_argument("--cleanup", action="store_true",
                        help="Cleanup old or incomplete checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the best checkpoint")
    parser.add_argument("--resume_from", type=str, 
                        help="Resume from specific checkpoint (step number or path)")
    parser.add_argument("--keep_best", type=int, default=3,
                        help="Number of best checkpoints to keep when cleaning up")
    parser.add_argument("--keep_last", type=int, default=1,
                        help="Number of most recent checkpoints to keep when cleaning up")
    parser.add_argument("--disable_deepspeed", action="store_true",
                        help="Disable DeepSpeed when resuming for better stability")
    # Add any additional arguments to pass to train_qlora.py
    parser.add_argument("--extra_args", type=str, nargs=argparse.REMAINDER,
                        help="Additional arguments to pass to train_qlora.py")
    
    args = parser.parse_args()
    
    # Find and display checkpoints
    checkpoints = find_checkpoints(args.output_dir)
    if checkpoints:
        display_checkpoints(checkpoints)
    
    # Clean up checkpoints if requested
    if args.cleanup:
        cleanup_checkpoints(checkpoints, args.keep_best, args.keep_last)
        # Refresh checkpoint list after cleanup
        checkpoints = find_checkpoints(args.output_dir)
    
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
            resume_training(checkpoint_to_resume, args.extra_args, args.disable_deepspeed)

if __name__ == "__main__":
    main()
