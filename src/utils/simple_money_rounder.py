import re
import argparse
from pathlib import Path
from rich.console import Console
import traceback

console = Console()

def round_money_amount(match):
    """Round a monetary amount to a cleaner value."""
    # Get the full match
    full_match = match.group(0)
    
    # Check if it's the $ format or digits+dollars format
    if '$' in full_match:
        # Extract the number part
        num_str = full_match.replace('$', '').replace(',', '')
        prefix = '$'
    else:
        # Extract the number part (before "dollars")
        num_str = full_match.replace(' dollars', '').replace(',', '')
        prefix = ''
        
    # Convert to float
    try:
        num = float(num_str)
    except ValueError:
        # If conversion fails, return original text
        return full_match
    
    # Determine rounding precision based on magnitude
    if num < 100:
        # Keep numbers under 100 as is
        rounded = num
    elif num < 1000:
        # Round to nearest 10
        rounded = round(num / 10) * 10
    elif num < 10000:
        # Round to nearest 100
        rounded = round(num / 100) * 100
    elif num < 100000:
        # Round to nearest 100
        rounded = round(num / 100) * 100
    elif num < 1000000:
        # Round to nearest 1,000
        rounded = round(num / 1000) * 1000
    else:
        # Round to nearest 10,000
        rounded = round(num / 10000) * 10000
    
    # Format as integer if it's a whole number
    if rounded == int(rounded):
        rounded = int(rounded)
    
    # Format with commas
    if prefix == '$':
        # Return in $X,XXX format
        result = f"${rounded:,}"
    else:
        # Return in X,XXX dollars format
        result = f"{rounded:,} dollars"
    
    return result

def process_file(input_path, output_path, exclude_companies=True):
    """Process the file and round all monetary values."""
    try:
        # Read the input file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        console.print(f"Read {len(content)} characters from {input_path}")
        
        # Define company names to exclude if needed
        company_names = [
            "Apple", "Microsoft", "Amazon", "Google", "Meta", "Tesla", 
            "JPMorgan", "Goldman Sachs", "Visa", "Bank of America", 
            "Johnson & Johnson", "Pfizer", "Walmart", "Target", "ExxonMobil"
        ]
        
        # Process content in chunks (split by dashes or paragraphs)
        separator = "-" * 80
        
        if separator in content:
            console.print("Found dash separator in the content")
            chunks = content.split(separator)
        else:
            console.print("Using paragraph separation")
            chunks = [p for p in content.split("\n\n") if p.strip()]
        
        console.print(f"Split content into {len(chunks)} chunks")
        
        # Process each chunk
        processed_chunks = []
        excluded_chunks = 0
        figures_found = 0
        
        for chunk in chunks:
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            # Check if chunk contains company references
            if exclude_companies and any(company in chunk for company in company_names):
                excluded_chunks += 1
                continue
            
            # Count money figures in chunk for reporting
            dollar_pattern = r'\$[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s+dollars\b'
            matches = re.findall(dollar_pattern, chunk)
            figures_found += len(matches)
            
            # Replace $ amounts: $X,XXX.XX or $X,XXX
            processed = re.sub(r'\$[\d,]+(?:\.\d+)?', round_money_amount, chunk)
            
            # Replace X dollars amounts: X,XXX.XX dollars or X,XXX dollars
            processed = re.sub(r'\b[\d,]+(?:\.\d+)?\s+dollars\b', round_money_amount, processed)
            
            processed_chunks.append(processed)
        
        # Write the processed content
        if processed_chunks:
            if separator in content:
                output_content = separator.join(processed_chunks) + separator
            else:
                output_content = "\n\n".join(processed_chunks)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            console.print(f"[green]Successfully processed file with {figures_found} financial figures[/green]")
            console.print(f"Excluded {excluded_chunks} chunks with company references")
            console.print(f"Output saved to {output_path}")
        else:
            console.print("[red]No valid content to write after processing![/red]")
        
    except Exception as e:
        console.print(f"[red]Error processing file:[/red] {str(e)}")
        console.print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Round financial figures in text to more natural values")
    
    parser.add_argument(
        "--input",
        type=str,
        default="/home/zahemen/datasets/advanced_finance_questions.txt",
        help="Input file containing financial text"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="/home/zahemen/datasets/advanced_finance_questions_rounded.txt",
        help="Output file path"
    )
    
    parser.add_argument(
        "--include-companies",
        action="store_true",
        help="Include text containing company references"
    )
    
    args = parser.parse_args()
    
    process_file(
        args.input,
        args.output,
        not args.include_companies
    )

if __name__ == "__main__":
    main()
