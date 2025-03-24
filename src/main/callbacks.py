from transformers.trainer_callback import TrainerCallback
from rich.console import Console
from rich.progress import Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import time
import datetime
from pathlib import Path
import torch

# Create a custom callback for pausing training with improved logging
class PauseResumeCallback(TrainerCallback):
    """Callback to pause training at multiple intervals and resume after specified delays."""
    
    def __init__(self, pause_intervals=[0.5], pause_durations=[30]):
        """
        Initialize the callback with customizable pause points and durations.
        
        Args:
            pause_intervals (list): List of training completion percentages to pause at (0.0-1.0)
                                   e.g., [0.25, 0.5, 0.75] would pause at 25%, 50%, and 75% of training
            pause_durations (list): List of pause durations in minutes, corresponding to each interval
                                   e.g., [60, 240, 60] would pause for 1hr, 4hrs, and 1hr at the respective intervals
        """
        self.pause_intervals = pause_intervals
        self.pause_durations = pause_durations
        
        # Ensure pause_durations has same length as pause_intervals
        if len(pause_durations) < len(pause_intervals):
            # Extend with the last value if needed
            self.pause_durations.extend([pause_durations[-1]] * (len(pause_intervals) - len(pause_durations)))
        
        self.pause_steps = []
        self.completed_pauses = set()
        self.console = Console()
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Set the steps at which to pause training."""
        if state.max_steps > 0:
            # Calculate pause steps based on percentages of max_steps
            self.pause_steps = [int(interval * state.max_steps) for interval in self.pause_intervals]
            
            # Print the pause schedule
            self.console.print("[bold yellow]===== TRAINING PAUSE SCHEDULE =====[/bold yellow]")
            for idx, (step, duration) in enumerate(zip(self.pause_steps, self.pause_durations)):
                percent = self.pause_intervals[idx] * 100
                self.console.print(f"[yellow]Pause {idx+1}: At step {step} ({percent:.1f}% complete) - Will pause for {duration} minutes[/yellow]")
            self.console.print("[bold yellow]=================================[/bold yellow]")
            
        elif args.num_train_epochs:
            # If using epochs, calculate approximate steps
            steps_per_epoch = args.num_update_steps_per_epoch
            total_steps = args.num_train_epochs * steps_per_epoch
            
            # Calculate pause steps based on percentages of estimated total steps
            self.pause_steps = [int(interval * total_steps) for interval in self.pause_intervals]
            
            # Print the pause schedule
            self.console.print("[bold yellow]===== TRAINING PAUSE SCHEDULE (EPOCH-BASED) =====[/bold yellow]")
            for idx, (step, duration) in enumerate(zip(self.pause_steps, self.pause_durations)):
                percent = self.pause_intervals[idx] * 100
                epoch_estimate = step / steps_per_epoch
                self.console.print(f"[yellow]Pause {idx+1}: Around step {step} (Epoch ~{epoch_estimate:.1f}, {percent:.1f}% complete) - Will pause for {duration} minutes[/yellow]")
            self.console.print("[bold yellow]================================================[/bold yellow]")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Check if we've reached a pause step."""
        current_step = state.global_step
        
        # Check if we need to pause at this step
        for i, pause_step in enumerate(self.pause_steps):
            # If we've reached a pause step and haven't paused at this step yet
            if current_step >= pause_step and pause_step not in self.completed_pauses:
                # Get the pause duration for this interval
                pause_minutes = self.pause_durations[i]
                percent_complete = self.pause_intervals[i] * 100
                
                self.console.print("")
                self.console.print("[bold cyan] ╔═════════════════════════════════════════════════════════════╗[/bold cyan]")
                self.console.print(f"[bold cyan]║   TRAINING PAUSED AT {percent_complete:.1f}% COMPLETION     ║[/bold cyan]")
                self.console.print("[bold cyan] ╚═════════════════════════════════════════════════════════════╝[/bold cyan]")
                self.console.print(f"[yellow]Reached step {current_step}. Training will resume after {pause_minutes} minutes.[/yellow]")
                
                # Set a flag to save checkpoint in train_qlora.py
                control.should_save = True
                
                # Create a special pause checkpoint folder path
                pause_checkpoint_dir = Path(args.output_dir) / f"pause_checkpoint_{i+1}"
                
                # Save the checkpoint dir and pause index in the kwargs
                kwargs["pause_checkpoint_dir"] = str(pause_checkpoint_dir)
                kwargs["pause_index"] = i + 1
                kwargs["pause_minutes"] = pause_minutes
                kwargs["pause_percent"] = percent_complete
                
                # Calculate total seconds for pause
                total_seconds = pause_minutes * 60
                pause_start_time = time.time()
                pause_end_time = pause_start_time + total_seconds
                
                # Format for nicer display
                pause_start_formatted = datetime.datetime.fromtimestamp(pause_start_time).strftime("%H:%M:%S")
                pause_end_formatted = datetime.datetime.fromtimestamp(pause_end_time).strftime("%H:%M:%S")
                
                self.console.print(f"[yellow]Pause started at: {pause_start_formatted}[/yellow]")
                self.console.print(f"[green]Training will resume at: {pause_end_formatted}[/green]")
                
                # Free up GPU memory during pause
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Use rich.progress for a nice countdown
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task(f"[cyan]Pause {i+1} in progress...", total=total_seconds)
                    
                    elapsed = 0
                    while elapsed < total_seconds:
                        # Sleep in small increments to be responsive
                        time.sleep(1)
                        elapsed = time.time() - pause_start_time
                        remaining = total_seconds - elapsed
                        
                        # Update the progress bar
                        progress.update(task, completed=elapsed)
                        
                        # Log status updates periodically
                        if int(remaining) % 60 == 0 and int(remaining) > 0 and int(elapsed) > 0:
                            mins_remaining = int(remaining // 60)
                            hrs_remaining = mins_remaining // 60
                            mins_left = mins_remaining % 60
                            
                            if mins_remaining > 0 and mins_remaining % 10 == 0:  # Every 10 minutes
                                if hrs_remaining > 0:
                                    self.console.print(f"[yellow] {hrs_remaining}h {mins_left}m remaining until training resumes.[/yellow]")
                                else:
                                    self.console.print(f"[yellow] {mins_remaining} minutes remaining until training resumes.[/yellow]")
                
                self.console.print("")
                self.console.print("[bold green] ╔════════════════════════════════════════════════╗[/bold green]")
                self.console.print(f"[bold green]║      RESUMING TRAINING AFTER PAUSE {i+1}       ║[/bold green]")
                self.console.print("[bold green] ╚════════════════════════════════════════════════╝[/bold green]")
                
                # Mark this pause as completed
                self.completed_pauses.add(pause_step)
                
                # Only pause once per step (in case multiple pause points match)
                return control
        
        return control