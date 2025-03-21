import re
import argparse
from pathlib import Path
from typing import List, Set, Dict, Tuple
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
import logging

console = Console()

class FinancialFigureRounder:
    """Round overly specific financial figures to more natural values in text files."""

    def __init__(
        self,
        input_file: str = "/home/zahemen/datasets/advanced_finance_questions.txt",
        output_file: str = "/home/zahemen/datasets/advanced_finance_questions_rounded.txt",
        exclude_company_questions: bool = True,
        rounding_precision: Dict[str, int] = None
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.exclude_company_questions = exclude_company_questions
        
        # Default rounding precision by magnitude:
        # - Numbers under 1,000: round to nearest 10
        # - Numbers 1,000-9,999: round to nearest 100
        # - Numbers 10,000-99,999: round to nearest 100
        # - Numbers 100,000-999,999: round to nearest 1,000
        # - Numbers 1M+: round to nearest 10,000
        self.rounding_precision = rounding_precision or {
            "1": 10,     # $1-999 -> round to nearest $10
            "1000": 100, # $1,000-9,999 -> round to nearest $100
            "10000": 100, # $10,000-99,999 -> round to nearest $100
            "100000": 1000, # $100,000-999,999 -> round to nearest $1,000
            "1000000": 10000 # $1M+ -> round to nearest $10,000
        }
        
        # Load company names from the extract_advanced_finance_conversations.py file
        self.company_names = self._get_company_names()
        
        # Stats for reporting
        self.stats = {
            "questions_processed": 0,
            "questions_excluded": 0,
            "figures_rounded": 0,
            "excluded_questions": []
        }

    def _get_company_names(self) -> Set[str]:
        """Extract company names from the list in the extraction script."""
        # Hard-coded company names from the file
        company_names = {
            # Tech
            "Apple", "Microsoft", "Amazon", "Google", "Meta", "Facebook", "Tesla", "Nvidia",
            "Intel", "AMD", "IBM", "Oracle", "Salesforce", "Adobe", "Dell", "PayPal", "Qualcomm",
            # Financial
            "JPMorgan", "Goldman Sachs", "Visa", "Bank of America", "Citigroup", "Wells Fargo", 
            "American Express", "Morgan Stanley", "Mastercard",
            # Healthcare
            "Johnson & Johnson", "UnitedHealth", "Pfizer", "Merck", "Novartis", "Eli Lilly", 
            "AstraZeneca", "Moderna", "CVS Health",
            # Retail
            "Walmart", "Target", "Home Depot", "Costco", "Lowe's", "TJX", "Dollar General", 
            "Best Buy", "Kroger",
            # Energy
            "ExxonMobil", "Chevron", "Shell", "BP", "ConocoPhillips", "Occidental", 
            "Duke Energy", "NextEra Energy",
            # Industrial
            "Boeing", "Caterpillar", "3M", "General Electric", "Honeywell", "Lockheed Martin", 
            "Airbus", "Union Pacific", "Deere & Company", "Ford", "GM",
            # Consumer Goods
            "Procter & Gamble", "Coca-Cola", "PepsiCo", "Nike", "Unilever", "Colgate-Palmolive", 
            "Kraft Heinz", "Adidas", "McDonald's", "Starbucks",
            # Entertainment/Media
            "Disney", "Netflix", "Warner Bros", "Spotify", "Sony", "AT&T", "Verizon", "T-Mobile",
            # International
            "Toyota", "Samsung", "Alibaba", "HSBC", "Volkswagen", "Tencent", "Nestlé", "Honda", 
            "Siemens", "BASF", "JD.com", "Reliance Industries", "Roche", "SAP"
        }
        
        # Add variants for case-insensitive matching
        variants = set()
        for name in company_names:
            # Add possessive form 
            variants.add(f"{name}'s")
            # Add lowercase version
            variants.add(name.lower())
            variants.add(f"{name.lower()}'s")
        
        return company_names.union(variants)

    def _contains_company_reference(self, text: str) -> bool:
        """Check if text contains any company name references."""
        for company in self.company_names:
            if company in text:
                return True
        
        # Also check for common company-related terms
        company_terms = [
            r"\bP/E ratio\b", r"\bdividend yield\b", r"\bmarket cap\b", 
            r"\bquarterly report\b", r"\bearnings\b", r"\bstock price\b",
            r"\bshareholders\b", r"\bmarket share\b", r"\bacquired\b",
            r"\bmerger\b", r"\bacquisition\b"
        ]
        
        for term in company_terms:
            if re.search(term, text, re.IGNORECASE):
                return True
                
        return False

    def _round_money_figure(self, match: re.Match) -> str:
        """Round a money figure to a more natural value based on its magnitude."""
        # Extract the number without $ and commas
        number_str = match.group(2).replace(',', '')
        
        # Handle decimal values if present
        if '.' in number_str:
            parts = number_str.split('.')
            integer_part = parts[0]
            decimal_part = parts[1]
        else:
            integer_part = number_str
            decimal_part = ""
        
        # Convert to integer for rounding
        number = int(integer_part)
        
        # Determine rounding precision based on magnitude
        precision = 10  # Default
        for magnitude_str, prec in sorted(self.rounding_precision.items(), key=lambda x: int(x[0])):
            magnitude = int(magnitude_str)
            if number >= magnitude:
                precision = prec
        
        # Round the number
        rounded = round(number / precision) * precision
        
        # Format result with commas
        result = f"{rounded:,}"
        
        # Add decimal part back if needed (uncommon for dollar amounts)
        if decimal_part and int(decimal_part) > 0:
            # Round to nearest 10 cents if the cents part is specific
            if len(decimal_part) >= 2:
                cents = int(decimal_part[:2])
                if cents % 10 != 0:
                    cents = round(cents / 10) * 10
                    if cents == 100:  # Handle rounding up to next dollar
                        rounded += 1
                        cents = 0
                result = f"{rounded:,}.{cents:02d}"
            else:
                result = f"{rounded:,}.{decimal_part}"
        
        # Reconstruct the money figure with the correct prefix
        prefix = match.group(1)
        return f"{prefix}{result}"

    def _process_question_answer(self, qa_text: str) -> Tuple[str, bool]:
        """Process a single question-answer pair, rounding money figures.
        Returns the processed text and whether it should be excluded."""
        
        # Check for company references first - if found, mark for exclusion
        if self.exclude_company_questions and self._contains_company_reference(qa_text):
            return qa_text, True
        
        # Find and replace money figures - handles both $X and X dollars formats
        # Fix: Modified regex patterns to better match dollar amounts
        
        # First pattern: $X,XXX.XX or $X,XXX format (more permissive)
        # Now matches $ followed by digits with optional commas and decimal point
        dollar_pattern = r'(\$)([\d,]+(?:\.\d+)?)'
        result = re.sub(dollar_pattern, lambda m: self._round_money_figure(m), qa_text)
        
        # Second pattern: X,XXX.XX dollars or X,XXX dollars format
        # Now matches digits with optional commas and decimal point followed by "dollars"
        dollar_word_pattern = r'()([\d,]+(?:\.\d+)?)\s+dollars'
        result = re.sub(dollar_word_pattern, lambda m: f"{self._round_money_figure(m)} dollars", result)
        
        # Count replacements
        figures_in_original = len(re.findall(dollar_pattern, qa_text)) + len(re.findall(dollar_word_pattern, qa_text))
        self.stats["figures_rounded"] += figures_in_original
        
        # Debug output to check if any replacements were made
        if figures_in_original > 0:
            console.print(f"[dim]Found {figures_in_original} figures to round[/dim]")
        
        return result, False

    def process_file(self) -> None:
        """Process the entire input file, separating question-answer pairs and rounding figures."""
        if not self.input_file.exists():
            console.print(f"[bold red]Error:[/bold red] Input file not found: {self.input_file}")
            return
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Display a sample of the content for debugging
            console.print(f"[dim]Sample of content ({min(200, len(content))} chars):[/dim] {content[:200]}...")
            
            # Fix: Ensure proper separator detection
            # Split the content by separation lines (80 dashes)
            separator = "-" * 80
            
            # Check if the separator exists in the content
            if separator not in content:
                console.print("[bold yellow]Warning:[/bold yellow] Could not find the separator pattern in the file. Using newline separation instead.")
                qa_blocks = [block for block in content.split("\n\n") if block.strip()]
            else:
                qa_blocks = [block for block in content.split(separator) if block.strip()]
            
            # Log the number of QA blocks found
            console.print(f"Found {len(qa_blocks)} QA blocks to process")
            
            # Process each Q&A block
            processed_blocks = []
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("[cyan]Processing Q&A pairs...", total=len(qa_blocks))
                
                for block in qa_blocks:
                    if not block.strip():
                        continue
                    
                    self.stats["questions_processed"] += 1
                    processed_block, should_exclude = self._process_question_answer(block)
                    
                    if should_exclude:
                        self.stats["questions_excluded"] += 1
                        try:
                            first_line = block.split('\n')[0]
                            self.stats["excluded_questions"].append(first_line)
                            console.print(f"[yellow]Excluding question containing company reference[/yellow]")
                        except:
                            console.print(f"[yellow]Excluding question with company reference (no first line)[/yellow]")
                    else:
                        processed_blocks.append(processed_block)
                    
                    progress.update(task, advance=1)
            
            # Combine the processed blocks
            if processed_blocks:
                # If we used newline separation instead of dashes, join with double newlines
                if separator not in content:
                    output_content = "\n\n".join(processed_blocks)
                else:
                    output_content = separator.join(processed_blocks) + separator
                
                # Write the processed content
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    f.write(output_content)
                
                # Report stats
                console.print(f"\n[bold green]Processing complete![/bold green]")
                console.print(f"Questions processed: {self.stats['questions_processed']}")
                console.print(f"Questions excluded due to company references: {self.stats['questions_excluded']}")
                console.print(f"Financial figures rounded: {self.stats['figures_rounded']}")
                console.print(f"Output saved to: {self.output_file}")
            else:
                console.print(f"\n[bold red]Error:[/bold red] No processed blocks to write. Check input format.")
            
        except Exception as e:
            console.print(f"[bold red]Error during processing:[/bold red] {str(e)}")
            import traceback
            console.print(traceback.format_exc())

def main():
    """Entry point for running the script from command line."""
    parser = argparse.ArgumentParser(
        description="Round financial figures in text files to more natural values."
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="/home/zahemen/datasets/advanced_finance_questions.txt",
        help="Input file path containing financial text with specific figures"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="/home/zahemen/datasets/advanced_finance_questions_rounded.txt",
        help="Output file path for the processed text"
    )
    
    parser.add_argument(
        "--include-companies",
        action="store_true",
        help="Include questions containing company references (excluded by default)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information during processing"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    rounder = FinancialFigureRounder(
        input_file=args.input,
        output_file=args.output,
        exclude_company_questions=not args.include_companies
    )
    
    rounder.process_file()

if __name__ == "__main__":
    main()
