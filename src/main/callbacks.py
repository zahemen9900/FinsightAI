

from transformers.trainer_callback import TrainerCallback
from rich.console import Console
from rich.progress import Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import time
import datetime
from pathlib import Path
# Create a custom callback for pausing training with improved logging
class PauseResumeCallback(TrainerCallback):
    """Callback to pause training halfway through and resume after a specified delay."""
    
    def __init__(self, pause_minutes=30):
        self.pause_minutes = pause_minutes
        self.pause_steps = None
        self.has_paused = False
        self.console = Console()
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Set the step at which to pause training."""
        if state.max_steps > 0:
            self.pause_steps = state.max_steps // 2
            self.console.print(f"[bold yellow]Training will pause for {self.pause_minutes} minutes at step {self.pause_steps}[/bold yellow]")
        elif args.num_train_epochs:
            # We can't know exact steps with epochs, so we'll use a percentage
            self.pause_steps = int(args.num_train_epochs * 0.5 * args.num_update_steps_per_epoch)
            self.console.print(f"[bold yellow]Training will pause for {self.pause_minutes} minutes around step {self.pause_steps} (mid-training)[/bold yellow]")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Check if we've reached the pause step."""
        if not self.has_paused and self.pause_steps and state.global_step >= self.pause_steps:
            self.console.print("")
            self.console.print("[bold cyan]╔════════════════════════════════════════╗[/bold cyan]")
            self.console.print("[bold cyan]║           TRAINING PAUSED              ║[/bold cyan]")
            self.console.print("[bold cyan]╚════════════════════════════════════════╝[/bold cyan]")
            self.console.print(f"[yellow]Reached step {state.global_step}. Training will resume after {self.pause_minutes} minutes.[/yellow]")
            
            # Save checkpoint before pausing
            self.console.print("[green]Saving checkpoint before pause...[/green]")
            control.should_save = True
            pause_checkpoint_dir = Path(args.output_dir) / "pause_checkpoint"
            pause_checkpoint_dir.mkdir(exist_ok=True, parents=True)
            
            # Calculate total seconds for pause
            total_seconds = self.pause_minutes * 60
            pause_start_time = time.time()
            pause_end_time = pause_start_time + total_seconds
            
            # Format for nicer display
            pause_start_formatted = datetime.datetime.fromtimestamp(pause_start_time).strftime("%H:%M:%S")
            pause_end_formatted = datetime.datetime.fromtimestamp(pause_end_time).strftime("%H:%M:%S")
            
            self.console.print(f"[yellow]Pause started at: {pause_start_formatted}[/yellow]")
            self.console.print(f"[green]Training will resume at: {pause_end_formatted}[/green]")
            
            # Use rich.progress for a nice countdown
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("[cyan]Paused...", total=total_seconds)
                
                elapsed = 0
                while elapsed < total_seconds:
                    time.sleep(1)  # Update every second
                    elapsed = time.time() - pause_start_time
                    remaining = total_seconds - elapsed
                    
                    # Update the progress bar
                    progress.update(task, completed=elapsed)
                    
                    # Every 10 mins, also log a message
                    if int(remaining) % 60 == 0 and int(remaining) > 0 and int(elapsed) > 0:
                        mins_remaining = int(remaining // 60)
                        if mins_remaining > 0 and mins_remaining % 10 == 0:  # Avoid duplicate 0 minute messages
                            self.console.print(f"[yellow]Still paused. {mins_remaining} minutes remaining until training resumes.[/yellow]")
            
            self.console.print("")
            self.console.print("[bold green]╔════════════════════════════════════════╗[/bold green]")
            self.console.print("[bold green]║           RESUMING TRAINING            ║[/bold green]")
            self.console.print("[bold green]╚════════════════════════════════════════╝[/bold green]")
            self.has_paused = True
        
        return control
