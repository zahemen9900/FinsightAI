import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
import random
import json
from dataclasses import dataclass
import logging
from rich.logging import RichHandler
import hashlib
import jsonlines
from tqdm import tqdm
from create_intro_dataset import IntroDatasetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')

@dataclass
class Conversation:
    messages: List[Dict[str, str]]
    metadata: Dict

class FinanceQAProcessor:

    def __init__(self, dataset_path: Path, num_samples: int = 5000):
        self.data = pd.read_csv(dataset_path)
        self.num_samples = num_samples
        self.used_samples = set()  # Track used question-answer pairs

        # Generate a list of varying system_prompt dicts
        self.system_prompt_variations = [
            {
                "role": "system",
                "content": (
                    "You are FinSight, a professional financial advisor chatbot specialized in "
                    "company analysis and financial insights. Provide accurate, factual responses "
                    "and use lists when appropriate to organize information clearly."
                )
            },
            {
                "role": "system",
                "content": (
                    "You are FinSight, an AI trained to provide clear, structured financial opinions & advice. "
                    "When appropriate, present information in well-organized lists."
                )
            },
            {
                "role": "system",
                "content": (
                    "You are a finance AI Assistant called FinSight, specializing in delivering financial insights through structured responses, including organized lists when beneficial."
                )
            }
        ]

        self.conversation_counter = 0  # Add counter for unique IDs
        self.sample_usage_counter = {}  # Track how many times each sample is used
        self.max_sample_usage = 3  # Maximum times a sample can be used
        
        # Company-specific question templates
        self.company_question_templates = [
            "What was {company}'s revenue growth in their latest report?",
            "How did {company} perform in terms of {metric}?",
            "What are {company}'s main products or services?",
            "How does {company} generate most of its revenue?",
            "What is {company}'s market position in {industry}?",
            "How has {company}'s strategy evolved recently?",
            "What are {company}'s competitive advantages?",
            "What risks does {company} face in their business?",
        ]
        
        # Load company names mapping
        self.company_names = self.load_company_names()
        
        # Minimum words required for valid responses
        self.min_words = 10
        self.max_words = 200
        
    def load_company_names(self) -> Dict[str, str]:
        """Load company ticker to name mapping"""
        try:
            with open('/home/zahemen/datasets/company_tickers.json', 'r') as f:
                data = json.load(f)
            
            # Create ticker -> company name mapping
            mapping = {}
            for item in data.values():
                ticker = item['ticker']
                # Clean company name (remove common corporate suffixes)
                name = re.sub(r'\s+(CORP|INC|LTD|LLC|CO|CORPORATION|LIMITED)\.?$', '', 
                            item['title'], flags=re.IGNORECASE)
                mapping[ticker] = name
                
            logger.info(f"Loaded {len(mapping)} company name mappings")
            return mapping
        except Exception as e:
            logger.error(f"Failed to load company names: {e}")
            return {}
            
    def get_company_name(self, ticker: str) -> str:
        """Get company name from ticker, fallback to ticker if not found"""
        company_name = self.company_names.get(ticker, ticker)
        if company_name == ticker:
            return company_name
        else:
            return company_name.title()

    def generate_prompt_id(self) -> str:
        """Generate a unique 64-character prompt ID"""
        # Combine counter with some random bytes for uniqueness
        unique_string = f"{self.conversation_counter}_{random.getrandbits(32)}"
        # Create SHA-256 hash (64 characters)
        prompt_id = hashlib.sha256(unique_string.encode()).hexdigest()
        self.conversation_counter += 1
        return prompt_id

    def create_greetings(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Create a greeting template"""
        usr_greetings = [
            "Hello",
            "Hi",
            "Greetings",
            "Hey",
            "Good day",
            "Hello there",
            "Hi there",
            "Howdy",
        ]

        bot_greetings = [
            "Hello! I'm your financial analysis assistant. I specialize in analyzing company filings and financial data. How can I help you today?",
            "Hi there! I'm a financial analysis bot ready to help you understand company data and financial statements. What would you like to explore?",
            "Greetings! I'm specialized in analyzing financial reports and company performance metrics. How may I assist you with your financial inquiries?",
            "Hello there! As your financial analysis assistant, I can help you understand company filings, financial metrics, and business performance. What would you like to know?",
            "Good day! I'm here to help you navigate through financial data and company information. How can I assist you with your analysis today?",
        ]
        return {"role": "user", "content": random.choice(usr_greetings)}, {"role": "assistant", "content": random.choice(bot_greetings)}

    def create_conversation_starter(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Create a conversation starter template"""
        starters = [
            "I have some questions about company financials.",
            "Could you help me understand some financial information?",
            "I'm analyzing some company data and need assistance.",
            "I'd like to learn more about this company's financial position."
        ]

        bot_responses = [
            "I'd be happy to help you understand company financials. Please feel free to ask any specific questions about financial statements, performance metrics, or business operations.",
            "Of course! I specialize in analyzing financial information and can help you understand company data. What specific aspects would you like to explore?",
            "I'll be glad to assist with your financial analysis. I can help with interpreting financial statements, understanding key metrics, or analyzing business performance. What would you like to know?",
            "Absolutely! I'm well-versed in corporate financial analysis and can help explain company data in detail. What particular aspects would you like to understand better?",
        ]
        return {"role": "user", "content": random.choice(starters)}, {"role": "assistant", "content": random.choice(bot_responses)}

    def create_system_message(self) -> Dict[str, str]:
        """Create a consistent system message"""
        return random.choice(self.system_prompt_variations)

    def format_company_question(self, question: str, company_name: str) -> str:
        """Format a question to include company name naturally"""
        if not isinstance(question, str) or not question.strip():
            return ""
            
        # Replace generic company references with specific name
        patterns = [
            (r'\bthe company\'s\b', f"{company_name}'s"),
            (r'\bthe company\b', company_name),
            (r'\bthey\b', company_name),
            (r'\btheir\b', f"{company_name}'s"),
            (r'\bcompany\b', company_name),
            (r'\bfirm\'s\b', f"{company_name}'s"),
            (r'\bfirm\b', company_name),
        ]
        
        formatted = question
        for pattern, replacement in patterns:
            formatted = re.sub(pattern, replacement, formatted, flags=re.IGNORECASE)
            
        # Add company name at start if no reference exists
        if company_name.lower() not in formatted.lower():
            formatted = f"For {company_name}, {formatted}"
            
        return formatted.strip()

    def is_valid_response(self, text: str) -> bool:
        """Check if a response is valid (not too short, not too long)"""
        if not isinstance(text, str) or not text.strip():
            return False
            
        # Check length
        words = text.split()
        if len(words) < self.min_words or len(words) > self.max_words:
            return False
            
        # Check for unwanted formatting or references
        unwanted_patterns = [
            r'10-[kq]',
            r'form 10',
            r'^\$\d+',  # Starting with dollar amounts
            r'^\d{4}',  # Starting with years
            r'<[^>]+>',  # HTML tags
            r'sorry, i don\'t'
        ]
        
        for pattern in unwanted_patterns:
            if re.search(pattern, text.lower()):
                return False
                
        return True

    def create_multi_turn_conversation(self, company_data: pd.DataFrame, num_turns: int) -> Conversation:
        """Create a multi-turn conversation focused on a single company"""
        use_starter = random.random() < 0.5  # 50% chance to use starter messages
        if use_starter:
            use_basic_starter = random.random() < 0.3
            if use_basic_starter:
                messages = []
                greeting, response = self.create_greetings()
                usr, ast = self.create_conversation_starter()
                
                # Add system message
                messages.append(self.create_system_message())
                messages.extend([greeting, response, usr, ast])
            else:
                intro_generator = IntroDatasetGenerator(None)
                messages = intro_generator.generate_conversation()
        else:
            messages = [self.create_system_message()]

        # Get company name
        company_name = self.get_company_name(company_data['ticker'].iloc[0])
        
        # Select valid questions and answers for this company
        available_indices = []
        for idx in company_data.index:
            if self.can_use_sample(idx) and self.is_valid_response(company_data.loc[idx, 'answer']):
                available_indices.append(idx)
                
        # Adjust number of turns if needed
        num_turns = min(num_turns, len(available_indices))
        if num_turns == 0:
            return None
            
        # Randomly select indices for this conversation
        selected_indices = random.sample(available_indices, num_turns)
        
        # Add the question-answer pairs
        for idx in selected_indices:
            row = company_data.loc[idx]
            formatted_question = self.format_company_question(row['question'], company_name)
            
            messages.extend([
                {"role": "user", "content": formatted_question or ""},
                {"role": "assistant", "content": row['answer'] or ""}
            ])
            self.mark_sample_used(idx)
        
        return Conversation(
            messages=messages,
            metadata={
                "ticker": company_data['ticker'].iloc[0],
                "company_name": company_name,
                "filing_year": company_data['filing'].iloc[0],
                "conversation_id": self.generate_prompt_id(),
                "turns": num_turns
            }
        )

    def can_use_sample(self, idx: int) -> bool:
        """Check if a sample can still be used"""
        return self.sample_usage_counter.get(idx, 0) < self.max_sample_usage
        
    def mark_sample_used(self, idx: int):
        """Mark a sample as used"""
        self.sample_usage_counter[idx] = self.sample_usage_counter.get(idx, 0) + 1
        if self.sample_usage_counter[idx] >= self.max_sample_usage:
            self.used_samples.add(idx)

    def process_dataset(self, output_path: Path, max_samples_per_company: int = 20):
        """Process the dataset and create company-focused conversations"""
        processed_conversations = []
        
        # Group by ticker
        grouped = self.data.groupby('ticker')
        tickers = list(grouped.groups.keys())
        
        conversation_count = 0
        company_count = 0
        target_companies = min(len(tickers), self.num_samples // 2)
        
        logger.info(f"Processing {target_companies} companies")
        
        # Process each company
        for ticker, company_data in tqdm(grouped, desc="Processing companies"):
            company_count += 1
            if company_count > target_companies:
                break
                
            logger.debug(f"Processing company: {ticker}")
            
            # Skip companies with too few samples
            if len(company_data) < 3:
                continue
                
            # Create conversations with varying number of turns
            samples_for_company = min(max_samples_per_company, len(company_data) // 3)
            
            for _ in range(samples_for_company):
                num_turns = random.randint(3, min(7, len(company_data)))
                
                try:
                    # Check if we have enough usable samples
                    available = [idx for idx in company_data.index if self.can_use_sample(idx)]
                    if len(available) < num_turns:
                        continue
                        
                    conv = self.create_multi_turn_conversation(company_data, num_turns)
                    if conv:
                        processed_conversations.append(conv)
                        conversation_count += 1
                        
                        # Log progress periodically
                        if conversation_count % 100 == 0:
                            logger.info(f"Generated {conversation_count} conversations so far...")
                            
                except Exception as e:
                    logger.error(f"Error creating conversation for {ticker}: {e}")
                    continue
        
        # Save processed conversations
        logger.info(f"Generated {len(processed_conversations)} conversations")
        self.save_conversations(processed_conversations, output_path)

    def save_conversations(self, conversations: List[Conversation], output_path: Path):
        """Save processed conversations to JSONL file with proper string handling"""
        def ensure_string_content(message):
            """Ensure message content is always a string"""
            if not isinstance(message.get('content', ''), str):
                message['content'] = str(message['content'])
            return message

        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                # Convert messages and ensure content is string
                conv_dict = {
                    "messages": [ensure_string_content(msg.copy()) for msg in conv.messages],
                    "metadata": conv.metadata
                }
                # Validate before writing
                if all(isinstance(msg['content'], str) for msg in conv_dict['messages']):
                    f.write(json.dumps(conv_dict, ensure_ascii=False) + '\n')
                else:
                    logger.warning(f"Skipping conversation with invalid content types: {conv.metadata.get('conversation_id', 'unknown')}")

        logger.info(f"Saved {len(conversations)} conversations to {output_path}")
        
        # Validate the saved file
        try:
            # Test load first few lines
            with open(output_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Check first 10 lines
                        break
                    data = json.loads(line)
                    assert all(isinstance(msg['content'], str) for msg in data['messages']), "Content validation failed"
            logger.info("Output file validation successful")
        except Exception as e:
            logger.error(f"Output file validation failed: {e}")
            raise

def main():
    dataset_path = Path('/home/zahemen/datasets/Financial-QA-10k.csv')
    output_path = Path('/home/zahemen/datasets/sft_datasets/company_focused_conversations.jsonl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processor = FinanceQAProcessor(dataset_path, num_samples=2500)
    processor.process_dataset(output_path)

if __name__ == "__main__":
    main()
