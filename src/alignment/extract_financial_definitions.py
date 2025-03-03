import os
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from rich.logging import RichHandler
from tqdm import tqdm
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")

class FinancialDefinitionsExtractor:
    """Extract financial definitions from text files and convert to JSONL format."""
    
    def __init__(
        self, 
        input_dir: str = "/home/zahemen/datasets/enhanced_q_and_a",
        output_file: str = "/home/zahemen/datasets/financial_definitions.jsonl",
        exclude_files: List[str] = ["finance_questions.txt"]
    ):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.definitions = []
        self.exclude_files = exclude_files
        
        # Ensure input directory exists
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
            
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(exist_ok=True, parents=True)

    def format_numbered_lists(self, definition: str) -> str:
        """Format numbered lists in definitions with newlines."""
        # Match patterns like "1)", "1.", "(1)", etc. that indicate a numbered list
        patterns = [
            (r'(\s+)(\d+\))', r'\n\2'),  # Matches " 1)" and adds newline
            (r'(\s+)(\d+\.)', r'\n\2'),  # Matches " 1." and adds newline
            (r'(\s+)\((\d+)\)', r'\n(\2)'),  # Matches " (1)" and adds newline
            # Handle first item specially (no space before it)
            (r'([.;:])(\s*)(\d+\))', r'\1\n\3'),  # Matches ". 1)" after another sentence
            (r'([.;:])(\s*)(\d+\.)', r'\1\n\3'),  # Matches ". 1." after another sentence
            (r'([.;:])(\s*)\((\d+)\)', r'\1\n(\3)')  # Matches ". (1)" after another sentence
        ]
        
        # Apply each pattern
        result = definition.replace("1)", "\n1)")
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
            
        return result

    def extract_definitions_from_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Extract term-definition pairs from a single file."""
        extracted_defs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Split content into potential definition blocks
            # This pattern looks for text blocks that start with a term followed by a colon
            
            # First, normalize newlines
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            
            # Add a newline at the end to help with parsing
            if not content.endswith('\n'):
                content = content + '\n'
            
            # This regex pattern looks for:
            # 1. A term at the start of a line
            # 2. Followed by a colon
            # 3. Capture everything until either:
            #    a) Another term followed by a colon at the start of a line, or
            #    b) End of the file
            term_pattern = r'(?:^|\n)([^:\n]+):\s*(.+?)(?=\n[^:\n]+:|$)'
            matches = re.findall(term_pattern, content, re.DOTALL)
            
            # Process each match
            for term, definition in matches:
                # Clean up the term and definition
                term = term.strip()
                definition = self.cleanup_definition(definition)
                
                # Skip if either term or definition is empty
                if not term or not definition:
                    continue
                
                # Skip very short definitions (likely errors)
                if len(definition.split()) < 3:
                    continue
                
                extracted_defs.append({
                    "term": term,
                    "definition": definition
                })
                
            logger.info(f"Extracted {len(extracted_defs)} definitions from {file_path.name}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            
        return extracted_defs

    def cleanup_definition(self, definition: str) -> str:
        """Clean up and format a definition while preserving structure."""
        # Remove any leading/trailing whitespace
        definition = definition.strip()
        
        # Preserve bullet points and lists but clean up excessive whitespace
        lines = []
        for line in definition.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line):
                # This is a bullet/list item - preserve indentation
                lines.append(line)
            elif line:  # Only add non-empty lines
                lines.append(line)
        
        # Join with appropriate spacing
        # If there are bullet points or numbered lists, preserve the newlines
        if any(line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line) for line in lines):
            # Join with newlines for lists
            cleaned = '\n'.join(lines)
        else:
            # Join with spaces for regular text
            cleaned = ' '.join(lines)
        
        # Ensure the definition ends with a period if it doesn't already
        if not cleaned.endswith('.') and not cleaned.endswith('?') and not cleaned.endswith('!'):
            cleaned += '.'
            
        return cleaned

    def is_similar_definition(self, def1: str, def2: str, threshold: float = 0.85) -> bool:
        """Check if two definitions are similar."""
        # Use SequenceMatcher to calculate similarity ratio
        similarity = SequenceMatcher(None, def1.lower(), def2.lower()).ratio()
        return similarity > threshold

    def clean_definition(self, definition: str) -> str:
        """Clean and format a definition."""
        # Remove starting phrases like "ATM is defined as" or "Bank means"
        patterns = [
            r'^It\s+is\s+', 
            r'^It\s+means\s+', 
            r'^It\s+refers\s+to\s+',
            r'^It\s+is\s+defined\s+as\s+'
            
        ]
        
        for pattern in patterns:
            definition = re.sub(pattern, '', definition, flags=re.IGNORECASE)
            
        # Capitalize first letter
        if definition and len(definition) > 0:
            definition = definition[0].upper() + definition[1:]
            
        return definition

    def format_definition(self, term: str, definition: str) -> str:
        """Format a definition with a term, accounting for singular/plural and articles."""
        # Check if the term is plural by checking common plural endings
        is_plural = term.lower().endswith(('s', 'es', 'ies')) and not term.lower().endswith(('ss', 'ics', 'ness', 'itis', 'osis'))
        
        # Check if the term starts with a vowel sound (simplified approach)
        starts_with_vowel = term.lower()[0] in 'aeiou'
        
        # Create lowercase version of the definition if it starts with uppercase
        definition_lower = definition.lower() if definition and definition[0].isupper() else definition
        
        # Different templates based on plurality and starting letter
        if is_plural:
            # Plural templates
            formats = [
                f"{definition}",
                f"{term} are {definition_lower}",
                f"{term} refer to {definition_lower}",
                f"{term} represent {definition_lower}",
                f"In finance, {term} are {definition_lower}"
            ]
        else:
            # Singular templates with proper article
            article = "an" if starts_with_vowel else "a"
            
            formats = [
                f"{definition}",
                f"{term} is {definition_lower}",
                f"{term} refers to {definition_lower}",
                f"{term} represents {definition_lower}",
                f"{article.capitalize()} {term} is {definition_lower}",
                f"In finance, {term} is {definition_lower}",
                f"The term {term} refers to {definition_lower}"
            ]
        
        # Choose a random format from the appropriate list
        return random.choice(formats)

    def group_duplicate_terms(self) -> None:
        """Group duplicate terms with multiple definitions."""
        term_groups = {}
        
        # Group definitions by term
        for definition in self.definitions:
            term = definition["term"]
            if term not in term_groups:
                term_groups[term] = []
            term_groups[term].append(definition["definition"])
        
        # Process each term group
        grouped_definitions = []
        for term, definitions in term_groups.items():
            # If only one definition, keep it simple
            if len(definitions) == 1:
                grouped_definitions.append({
                    "term": term,
                    "definition": self.format_definition(term, definitions[0])
                })
                continue
                
            # For multiple definitions, check for duplicates and merge them
            unique_definitions = []
            for definition in definitions:
                cleaned_def = self.clean_definition(definition)
                is_duplicate = False
                
                # Check if this definition is similar to any already added
                for existing_def in unique_definitions:
                    if self.is_similar_definition(cleaned_def, existing_def):
                        is_duplicate = True
                        break
                        
                if not is_duplicate and cleaned_def:
                    unique_definitions.append(cleaned_def)
            
            # If we have multiple unique definitions, combine them
            if len(unique_definitions) > 1:
                combined_def = self.format_definition(term, unique_definitions[0])
                
                for additional_def in unique_definitions[1:]:
                    # Check if the additional definition already starts with the term
                    term_pattern = re.compile(rf"(?:a|an|the)?\s*{re.escape(term)}\s+(?:is|are|refers to|mean)", re.IGNORECASE)
                    
                    if term_pattern.match(additional_def):
                        # If it does, add it directly without the connector phrase
                        combined_def += f" {additional_def}"
                    else:
                        # Check if the term is plural
                        is_plural = term.lower().endswith(('s', 'es', 'ies')) and not term.lower().endswith(('ss', 'ics', 'ness', 'itis', 'osis'))
                        
                        # Choose appropriate connector based on plurality
                        if is_plural:
                            connector = random.choice([
                                f" They can also refer to {additional_def.lower() if additional_def[0].isupper() else additional_def}",
                                f" They are also defined as {additional_def.lower() if additional_def[0].isupper() else additional_def}",
                                f" Another definition is that they are {additional_def.lower() if additional_def[0].isupper() else additional_def}"
                            ])
                        else:
                            connector = random.choice([
                                f" It can also refer to {additional_def.lower() if additional_def[0].isupper() else additional_def}",
                                f" It's also defined as {additional_def.lower() if additional_def[0].isupper() else additional_def}",
                                f" Another definition is that it is {additional_def.lower() if additional_def[0].isupper() else additional_def}"
                            ])
                        
                        combined_def += connector
                
                grouped_definitions.append({
                    "term": term,
                    "definition": combined_def
                })
            elif len(unique_definitions) == 1:
                # Just one unique definition
                grouped_definitions.append({
                    "term": term,
                    "definition": self.format_definition(term, unique_definitions[0])
                })
                
        logger.info(f"Reduced from {len(self.definitions)} to {len(grouped_definitions)} definitions after combining duplicates")
        self.definitions = grouped_definitions

    def process_all_files(self) -> None:
        """Process all text files in the input directory."""
        all_files = [f for f in self.input_dir.glob("*.txt") if f.name not in self.exclude_files]
        logger.info(f"Found {len(all_files)} text files to process")
        
        for file_path in tqdm(all_files, desc="Processing files"):
            file_definitions = self.extract_definitions_from_file(file_path)
            if file_definitions:
                self.definitions.extend(file_definitions)
            else:
                logger.warning(f"No definitions extracted from {file_path}")
            
        logger.info(f"Extracted {len(self.definitions)} financial definitions in total")

    def save_to_jsonl(self) -> None:
        """Save the extracted definitions to a JSONL file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for definition in self.definitions:
                f.write(json.dumps(definition, ensure_ascii=False) + '\n')
                
        logger.info(f"Saved {len(self.definitions)} definitions to {self.output_file}")

    # def deduplicate_definitions(self) -> None:
    #     """Remove duplicate definitions based on term names."""
    #     term_dict = {}
    #     for item in self.definitions:
    #         term = item['term'].lower()  # Case-insensitive comparison
    #         if term not in term_dict:
    #             term_dict[term] = item
    #         else:
    #             # If definition already exists, keep the longer one
    #             existing_def = term_dict[term]['definition']
    #             new_def = item['definition']
    #             if len(new_def) > len(existing_def):
    #                 term_dict[term] = item
        
    #     original_count = len(self.definitions)
    #     self.definitions = list(term_dict.values())
    #     logger.info(f"Deduplicated definitions: {original_count} → {len(self.definitions)}")

    def run(self) -> None:
        """Execute the full extraction process."""
        logger.info("Starting financial definitions extraction")
        self.process_all_files()
        # self.deduplicate_definitions()
        self.save_to_jsonl()
        logger.info("Extraction complete!")
        
        # Print a sample of extracted definitions
        sample_size = min(5, len(self.definitions))
        logger.info(f"\nSample of {sample_size} extracted definitions:")
        for i in range(sample_size):
            logger.info(f"Term: {self.definitions[i]['term']}")
            logger.info(f"Definition: {self.definitions[i]['definition']}")
            logger.info("")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract financial definitions from text files')
    parser.add_argument('--input', default="/home/zahemen/datasets/enhanced_q_and_a", 
                        help='Directory containing definition files')
    parser.add_argument('--output', default="/home/zahemen/datasets/sft_datasets/financial_definitions.jsonl",
                        help='Output JSONL file path')
    parser.add_argument('--exclude', nargs='*', default=['finance_questions.txt'],
                        help='Files to exclude from processing')
    
    args = parser.parse_args()
    
    extractor = FinancialDefinitionsExtractor(
        input_dir=args.input,
        output_file=args.output,
        exclude_files=args.exclude
    )
    extractor.run()
