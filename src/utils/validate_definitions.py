import json
import re
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print

console = Console()

class DefinitionValidator:
    """Validate generated financial definitions for common issues."""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.issues = {
            "redundant_phrases": [],
            "awkward_starts": [],
            "formal_language": [],
            "poor_transitions": [],
            "overlapping_text": []
        }
        self.samples = []
        
        # Expanded patterns to check for issues
        self.redundant_patterns = [
            r"refers to\s+it\s+refers\s+to",
            r"is\s+this\s+is",
            r"means\s+this\s+means",
            r"represents\s+this\s+represents",
            r"describes\s+this\s+describes",
            r"involves\s+this\s+involves",
            r"encompasses\s+this\s+encompasses",
        ]
        
        self.awkward_starts = [
            r"^This refers to",
            r"^It refers to",
            r"^This is a term that",
            r"^This is a concept that",
        ]
        
        self.formal_language = [
            r"is defined as",
            r"may be defined as",
            r"can be defined as",
            r"is described as",
            r"is characterized by",
            r"is utilized to",
            r"is employed to",
            r"in nature$",
        ]
        
        self.poor_transitions = [
            r"On the other hand, .+ refers to",
            r"In contrast, .+ refers to", 
            r"Meanwhile, .+ refers to",
        ]
        
        self.overlapping_patterns = [
            (r"In simple terms, .+ is this is", "overlapping intro with 'this is'"),
            (r"Simply put, .+ is this is", "overlapping intro with 'this is'"),
            (r"At its core, .+ is this is", "overlapping intro with 'this is'"),
            (r"Essentially, .+ is this is", "overlapping intro with 'this is'"),
            (r".+, in financial terms, is this is", "overlapping intro with 'this is'"),
            (r"In the financial world, .+ is this is", "overlapping intro with 'this is'"),
            (r".+ basically means this means", "overlapping intro with 'this means'"),
            (r".+ represents this represents", "overlapping intro with 'this represents'"),
            (r"is essentially The", "missing space or punctuation"),
            (r"is basically The", "missing space or punctuation"),
        ]
    
    def load_dataset(self):
        """Load the dataset from a JSONL file."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.samples = [json.loads(line) for line in f]
            print(f"Loaded {len(self.samples)} samples from {self.dataset_path}")
        except Exception as e:
            print(f"[bold red]Error loading dataset:[/bold red] {e}")
            return False
        return True
    
    def check_sample(self, sample):
        """Check a single sample for issues."""
        found_issues = {}
        
        # Get assistant messages
        assistant_messages = [msg["content"] for msg in sample["messages"] 
                             if msg["role"] == "assistant"]
        
        # Check each message
        for i, message in enumerate(assistant_messages):
            message_issues = []
            
            # Check for redundant phrases
            for pattern in self.redundant_patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    message_issues.append(f"redundant_phrases: '{pattern}'")
                    self.issues["redundant_phrases"].append(sample)
            
            # Check for awkward starts
            for pattern in self.awkward_starts:
                if re.search(pattern, message, re.IGNORECASE):
                    message_issues.append(f"awkward_starts: '{pattern}'")
                    self.issues["awkward_starts"].append(sample)
            
            # Check for formal language
            for pattern in self.formal_language:
                if re.search(pattern, message, re.IGNORECASE):
                    message_issues.append(f"formal_language: '{pattern}'")
                    self.issues["formal_language"].append(sample)
            
            # Check for poor transitions
            for pattern in self.poor_transitions:
                if re.search(pattern, message, re.IGNORECASE):
                    message_issues.append(f"poor_transitions: '{pattern}'")
                    self.issues["poor_transitions"].append(sample)
            
            # Check for overlapping text
            for pattern, issue_type in self.overlapping_patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    message_issues.append(f"overlapping_text: '{issue_type}'")
                    self.issues["overlapping_text"].append(sample)
            
            if message_issues:
                found_issues[i] = message_issues
        
        return found_issues
    
    def validate_dataset(self):
        """Validate all samples in the dataset."""
        all_issues = {}
        
        for i, sample in enumerate(self.samples):
            sample_issues = self.check_sample(sample)
            if sample_issues:
                all_issues[i] = {
                    "sample": sample,
                    "issues": sample_issues
                }
        
        return all_issues
    
    def fix_definition(self, definition):
        """Fix common definition issues for demonstration purposes."""
        # Remove redundant phrases
        fixed = definition
        fixed = re.sub(r'is\s+this\s+is', 'is', fixed, flags=re.IGNORECASE)
        fixed = re.sub(r'means\s+this\s+means', 'means', fixed, flags=re.IGNORECASE)
        fixed = re.sub(r'represents\s+this\s+represents', 'represents', fixed, flags=re.IGNORECASE)
        
        # Fix capitalization after certain phrases
        fixed = re.sub(r'(is essentially|is basically) ([A-Z])', r'\1 \2', fixed)
        
        return fixed
    
    def print_summary(self, all_issues):
        """Print a summary of validation results."""
        # Create a table for the summary
        table = Table(title="Definition Validation Results")
        table.add_column("Issue Type", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("% of Samples", style="green")
        
        total_samples = len(self.samples)
        total_issues = sum(len(v) for v in self.issues.values())
        unique_samples_with_issues = len(all_issues)
        
        # Add rows for each issue type
        for issue_type, samples in self.issues.items():
            count = len(samples)
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            table.add_row(
                issue_type.replace("_", " ").title(),
                str(count),
                f"{percentage:.1f}%"
            )
        
        # Add summary row
        table.add_row(
            "Total Unique Samples with Issues", 
            str(unique_samples_with_issues),
            f"{(unique_samples_with_issues / total_samples) * 100:.1f}%" if total_samples > 0 else "0%",
            style="bold"
        )
        
        console.print(table)
        
        # Print example issues with fixes
        if all_issues:
            print("\n[bold]Examples of issues found (with suggested fixes):[/bold]")
            shown = 0
            for sample_idx, data in all_issues.items():
                if shown >= 5:  # Limit to 5 examples
                    break
                
                sample = data["sample"]
                term = sample["metadata"].get("term", sample["metadata"].get("terms", "Unknown term"))
                
                print(f"\n[bold cyan]Sample {sample_idx}[/bold cyan] [yellow]({term})[/yellow]:")
                for msg_idx, msg_issues in data["issues"].items():
                    # Get the problematic message
                    message = sample["messages"][msg_idx*2+2]["content"]  # Assistant messages
                    
                    # Show truncated original
                    print(f"  Original: \"{message[:100]}...\"" if len(message) > 100 else f"  Original: \"{message}\"")
                    print(f"  Issues: {', '.join(msg_issues)}")
                    
                    # Show suggested fix
                    fixed_message = self.fix_definition(message)
                    if fixed_message != message:
                        print(f"  Possible fix: \"{fixed_message[:100]}...\"" if len(fixed_message) > 100 
                              else f"  Possible fix: \"{fixed_message}\"")
                
                shown += 1
            
            if len(all_issues) > 5:
                print(f"\n...and {len(all_issues) - 5} more samples with issues")
        
        print("\n[bold green]Validation complete![/bold green]")
        
        print("\n[bold]Recommendation:[/bold] Run the dataset generation again with the improved cleanup patterns.")
    
    def run(self):
        """Run the validation process."""
        if not self.load_dataset():
            return
        
        print("[bold]Validating definitions...[/bold]")
        all_issues = self.validate_dataset()
        self.print_summary(all_issues)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate financial definition dataset")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="/home/zahemen/datasets/financial_definitions_dataset.jsonl", 
        help="Path to the dataset JSONL file"
    )
    args = parser.parse_args()
    
    validator = DefinitionValidator(args.dataset)
    validator.run()
