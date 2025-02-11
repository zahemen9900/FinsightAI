import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
import random
import json
from dataclasses import dataclass, asdict
import logging
from rich.logging import RichHandler
import hashlib
import jsonlines
from tqdm import tqdm

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
    messages: List[Dict[str, str]]  # Remove prompt field
    metadata: Dict

class FinanceQAProcessor:
    def __init__(self, dataset_path: Path, num_cross_company_samples: int = 2000):
        self.data = pd.read_csv(dataset_path)
        self.num_cross_company_samples = num_cross_company_samples
        self.used_samples = set()  # Track used question-answer pairs
        self.all_tickers = None  # Will store all unique tickers
        self.system_prompts = [
                "You are a finance Bot trained on a dataset of 10-K filings. You are tasked with answering questions related to the 10-K filings.",
                "You are FinSight, a finance Bot trained on a dataset of 10-K filings. You are tasked with answering questions related to the filings in a professional and concise manner.",
                "You are a finance Bot trained on a dataset of 10-K filings. You are tasked with answering questions related to the 10-K filings in an accurate and concise manner.",
                "You are FinSight (Financial Insight), a finance Bot trained on a dataset of 10-K filings of various S&P500 Companies. You are expected to answer questions related to the 10-K filings in a professional and accurate manner.",
                "You are a finance Bot trained on 10-K filings of various S&P500 Companies. Your Name is FinSight - Finacial Insights Bot.",
            ]
        self.context_prompt_templates = [
            "Based on this context: {context}\n",
            "According to the company's filing: {context}\n",
            "Given this information from the 10-K filing: {context}\n",
            "Considering this excerpt from the filing: {context}\n"
        ]
        
        self.system_prompt_with_context = [
            "You are a finance Bot trained on 10-K filings. Here's relevant information from the company's filing: {context}",
            "You are FinSight, an AI trained on financial documents. According to the company's 10-K: {context}",
            "You are a financial advisor with access to company filings. Based on their 10-K: {context}",
            "You are FinSight, an AI assistant specialized in financial analysis. From the company's filing: {context}"
        ]
        
        self.system_prompt_no_context = [
            "You are a finance Bot trained on 10-K filings of various S&P500 Companies.",
            "You are FinSight, an AI trained to analyze and explain company financials.",
            "You are a financial advisor bot trained to answer questions about company financials.",
            "You are an AI assistant specialized in financial analysis and company information."
        ]
        self.conversation_counter = 0  # Add counter for unique IDs
        self.sample_usage_counter = {}  # Track how many times each sample is used
        self.max_sample_usage = 3  # Maximum times a sample can be used
        
    def generate_prompt_id(self) -> str:
        """Generate a unique 64-character prompt ID"""
        # Combine counter with some random bytes for uniqueness
        unique_string = f"{self.conversation_counter}_{random.getrandbits(32)}"
        # Create SHA-256 hash (64 characters)
        prompt_id = hashlib.sha256(unique_string.encode()).hexdigest()
        self.conversation_counter += 1
        return prompt_id
        
    def format_with_context(self, question: str, context: str) -> str:
        """Format a question with its context"""
        # Handle NaN or invalid context
        if pd.isna(context) or not isinstance(context, str):
            return question.strip()
        template = random.choice(self.context_prompt_templates)
        return template.format(context=context.strip()) + question.strip()

    def combine_qa_pairs(self, row1, row2) -> Tuple[str, str]:
        """Combine two Q&A pairs into a single question and answer without context in question"""
        # Combine questions without context
        combined_q = f"{row1['question']} Additionally, {row2['question']}"

        combined_a = np.random.choice([
            f"{row1['answer']} Additionally, {row2['answer']}",
            f"1. {row1['answer']} \n2. {row2['answer']}",
            f"{row1['answer']} \nAlso, {row2['answer']}",
            f"{row1['answer']} In addition, {row2['answer']}",
        ])
        
        # Combine contexts if they're different (for system prompt)
        if row1['context'] != row2['context']:
            self.combined_context = f"{row1['context']} Additionally: {row2['context']}"
        else:
            self.combined_context = row1['context']
            
        return combined_q, combined_a
    
    def create_greetings(self) -> Dict[str, str]:
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
            "Hi! I'm your financial insights assistant, trained to analyze company filings and financial statements. What aspects would you like to explore?",
            "Greetings! As your financial analysis companion, I can help you understand complex financial data and company performance. What can I help you with?",
            "Hello! I specialize in analyzing financial reports and company metrics. I'm here to help you understand the data better. What would you like to know?",
        ]
        return {"role": "user", "content": random.choice(usr_greetings)}, {"role": "assistant", "content": random.choice(bot_greetings)}

    def create_conversation_starter(self) -> Dict[str, str]:
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
            "Certainly! I can help you analyze and understand company financial information. Whether it's about financial statements, business metrics, or company performance, I'm here to assist."
        ]
        return {"role": "user", "content": random.choice(starters)}, {"role": "assistant", "content": random.choice(bot_responses)}

    def create_system_message(self, context: str = None, use_context: bool = True) -> Dict[str, str]:
        """Create a system message, optionally including context"""
        if use_context and context and not pd.isna(context) and isinstance(context, str):
            template = random.choice(self.system_prompt_with_context)
            content = template.format(context=context.strip())
        else:
            content = random.choice(self.system_prompt_no_context)
        return {"role": "system", "content": content or ""}  # Ensure content is never None

    def create_multi_turn_conversation(self, company_data: pd.DataFrame, num_turns: int) -> Conversation:
        """Create a multi-turn conversation for a specific company"""
        messages = []
        greeting, response = self.create_greetings()
        usr, ast = self.create_conversation_starter()
        
        # Decide whether to use context (50% chance)
        use_context = random.random() < 0.8
        
        # Add system prompt with/without context
        context = company_data['context'].iloc[0] if use_context else None
        messages.append(self.create_system_message(context, use_context))
        
        messages.extend([greeting, response, usr, ast])
        
        # Rest of conversation without context in questions
        available_indices = [idx for idx in company_data.index if self.can_use_sample(idx)]
        if len(available_indices) < num_turns:
            num_turns = len(available_indices)
            
        selected_indices = random.sample(available_indices, num_turns)
        
        for idx in selected_indices:
            row = company_data.loc[idx]
            self.mark_sample_used(idx)
            messages.extend([
                {"role": "user", "content": str(row['question'] or "")},  # Convert to string and handle None
                {"role": "assistant", "content": str(row['answer'] or "")}  # Convert to string and handle None
            ])
        
        return Conversation(
            messages=messages,
            metadata={
                "ticker": company_data['ticker'].iloc[0],
                "filing_year": company_data['filing'].iloc[0],
                "has_context": use_context,
                "conversation_id": self.generate_prompt_id()  # Renamed from prompt_id
            }
        )

    def create_cross_company_conversation(
        self,
        company1_data: pd.DataFrame,
        company2_data: pd.DataFrame,
        num_turns: int
    ) -> Conversation:
        """Create a conversation mixing questions from two companies"""
        messages = []
        greeting, response = self.create_greetings()
        usr, ast = self.create_conversation_starter()
        
        # Decide whether to use context and validate context exists
        use_context = random.random() < 0.8
        context1 = company1_data['context'].iloc[0]
        context2 = company2_data['context'].iloc[0]
        
        # Create combined context if using context and both contexts are valid
        if use_context and not pd.isna(context1) and not pd.isna(context2):
            context = f"For {company1_data['ticker'].iloc[0]}: {context1} For {company2_data['ticker'].iloc[0]}: {context2}"
        else:
            context = None
            use_context = False
            
        messages.append(self.create_system_message(context, use_context))
        messages.extend([greeting, response, usr, ast])
        
        # Rest of conversation without context in questions
        company1_indices = [idx for idx in company1_data.index if self.can_use_sample(idx)]
        company2_indices = [idx for idx in company2_data.index if self.can_use_sample(idx)]
        
        num_turns = min(num_turns, len(company1_indices), len(company2_indices))
        
        for i in range(num_turns):
            # Alternate between companies
            current_data = company1_data if i % 2 == 0 else company2_data
            available = [idx for idx in current_data.index if self.can_use_sample(idx)]
            
            if not available:
                continue
                
            idx = random.choice(available)
            row = current_data.loc[idx]
            
            # Add question without context
            messages.extend([
                {"role": "user", "content": row['question']},
                {"role": "assistant", "content": row['answer']}
            ])
            self.mark_sample_used(idx)
            
            # Remove used index from available indices
            if i % 2 == 0:
                company1_indices.remove(idx)
            else:
                company2_indices.remove(idx)
        
        return Conversation(
            messages=messages,
            metadata={
                "ticker": f"{company1_data['ticker'].iloc[0]}, {company2_data['ticker'].iloc[0]}",
                "combined": False,
                "cross_company": True,
                "has_context": use_context,
                "filing_year": f"{company1_data['filing'].iloc[0]}, {company2_data['filing'].iloc[0]}" if str(company1_data['filing'].iloc[0]) != str(company2_data['filing'].iloc[0]) else company1_data['filing'].iloc[0],
                "conversation_id": self.generate_prompt_id()
            }
        )

    def process_dataset(self, output_path: Path, max_samples_per_company: int = 15):
        """Process the entire dataset and create variations"""
        processed_conversations = []
        
        # Group by ticker
        grouped = self.data.groupby('ticker')
        self.all_tickers = list(grouped.groups.keys())
        
        for ticker, company_data in grouped:
            logger.info(f"Processing company: {ticker}")
            
            # Create some combined Q&A pairs
            system_prompts = [
                "You are a Finsight, a finance AI trained on a dataset of 10-K filings. You are tasked with answering questions related to the 10-K filings.",
                "You are FinSight, a finance Bot trained on a dataset of 10-K filings. You are tasked with answering questions related to the filings in a professional and concise manner.",
                "You are a finance AI trained on a dataset of 10-K filings. You are tasked with answering questions related to the 10-K filings in an accurate and concise manner.",
                "You are FinSight, a finance AI model trained on a dataset of 10-K filings of various S&P500 Companies. You are expected to answer questions related to the 10-K filings in a professional and accurate manner.",
            ]

            if len(company_data) >= 2:
                num_combinations = random.randint(3, 7)
                for _ in range(min(num_combinations, len(company_data) // 2)):
                    available = [idx for idx in company_data.index if self.can_use_sample(idx)]
                    if len(available) >= 2:
                        idx1, idx2 = random.sample(available, 2)
                        q, a = self.combine_qa_pairs(
                            company_data.loc[idx1],
                            company_data.loc[idx2]
                        )
                        
                        # Decide whether to use context
                        use_context = random.random() < 0.8
                        system_msg = self.create_system_message(
                            self.combined_context if use_context else None,
                            use_context
                        )
                        
                        greeting, response = self.create_greetings()
                        usr, ast = self.create_conversation_starter()
                        
                        messages = [
                            system_msg,
                            greeting, response,
                            usr, ast,
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a}
                        ] if np.random.rand() < 0.8 else [
                            system_msg,
                            usr, ast,
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a}
                        ]
                        
                        processed_conversations.append(Conversation(
                            messages=messages,
                            metadata={
                                "ticker": ticker,
                                "combined": True,
                                "filing_year": company_data['filing'].iloc[0],
                                "conversation_id": self.generate_prompt_id()  # Add prompt ID
                            }
                        ))
                        
                        self.mark_sample_used(idx1)
                        self.mark_sample_used(idx2)
            # Create multi-turn conversations
            num_conversations = random.randint(3, max_samples_per_company)
            for _ in range(num_conversations):
                num_turns = random.randint(3, 7)
                try:
                    available = [idx for idx in company_data.index if self.can_use_sample(idx)]
                    if len(available) >= num_turns:
                        conv = self.create_multi_turn_conversation(company_data, num_turns)
                        processed_conversations.append(conv)
                except ValueError:
                    continue  # Skip if not enough samples left
                    
        # Now create cross-company conversations
        logger.info("Generating cross-company conversations...")
        num_cross_company = self.num_cross_company_samples  # Number of cross-company conversations to generate
        
        for _ in range(num_cross_company):
            # Select two random companies
            company1_ticker, company2_ticker = random.sample(self.all_tickers, 2)
            company1_data = self.data[self.data['ticker'] == company1_ticker]
            company2_data = self.data[self.data['ticker'] == company2_ticker]
            
            # Generate a conversation with 3-7 turns
            num_turns = random.randint(3, 7)
            try:
                available1 = [idx for idx in company1_data.index if self.can_use_sample(idx)]
                available2 = [idx for idx in company2_data.index if self.can_use_sample(idx)]
                if len(available1) >= num_turns//2 and len(available2) >= num_turns//2:
                    conv = self.create_cross_company_conversation(
                        company1_data,
                        company2_data,
                        num_turns
                    )
                    processed_conversations.append(conv)
            except (ValueError, IndexError):
                continue
                
            if _ % 20 == 0:  # Log progress every 20 conversations
                logger.info(f"Generated {_}/{num_cross_company} cross-company conversations")
        
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

    def can_use_sample(self, idx: int) -> bool:
        """Check if a sample can still be used"""
        return self.sample_usage_counter.get(idx, 0) < self.max_sample_usage
        
    def mark_sample_used(self, idx: int):
        """Mark a sample as used"""
        self.sample_usage_counter[idx] = self.sample_usage_counter.get(idx, 0) + 1
        if self.sample_usage_counter[idx] >= self.max_sample_usage:
            self.used_samples.add(idx)

    def validate_conversation(self, messages: List[Dict[str, str]]) -> bool:
        """Validate conversation format and content"""
        if not isinstance(messages, list):
            return False
            
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if 'role' not in msg or 'content' not in msg:
                return False
            if not isinstance(msg['role'], str) or not isinstance(msg['content'], str):
                return False
            if msg['role'] not in ['system', 'user', 'assistant']:
                return False
            if not msg['content'].strip():
                return False
        return True

    def clean_message(self, content: str) -> str:
        """Clean message content"""
        if not isinstance(content, str):
            return ""
        return content.strip()

    def process_conversation(self, data: Dict[str, Any], source: str) -> Dict:
        """Process a single conversation"""
        try:
            # Ensure messages exist and are properly formatted
            messages = data.get('messages', [])
            if not self.validate_conversation(messages):
                raise ValueError("Invalid conversation format")
            
            # Clean message content
            cleaned_messages = []
            for msg in messages:
                cleaned_msg = {
                    "role": msg["role"],
                    "content": self.clean_message(msg["content"])
                }
                if cleaned_msg["content"]:  # Only add if content is not empty
                    cleaned_messages.append(cleaned_msg)
            
            if not cleaned_messages:
                raise ValueError("No valid messages after cleaning")
            
            # Generate conversation ID
            conv_id = hashlib.sha256(
                f"{source}_{cleaned_messages[0]['content']}".encode()
            ).hexdigest()
            
            return {
                "messages": cleaned_messages,
                "metadata": {
                    "source": source,
                    "conversation_id": conv_id,
                    "type": "qa"
                }
            }
        except Exception as e:
            logger.warning(f"Error processing conversation: {e}")
            return None

    def process_files(self):
        """Process all input files and write to output"""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_conversations = []
        total_processed = 0
        total_failed = 0
        
        try:
            # Process each input file
            for input_file in self.input_files:
                logger.info(f"Processing {input_file}")
                source = Path(input_file).stem
                
                try:
                    # Read input file line by line
                    with open(input_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(tqdm(f), 1):
                            try:
                                # Parse and validate each line
                                data = json.loads(line.strip())
                                processed = self.process_conversation(data, source)
                                
                                if processed:
                                    processed_conversations.append(processed)
                                    total_processed += 1
                                else:
                                    total_failed += 1
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON at line {line_num}: {e}")
                                total_failed += 1
                                continue
                                
                except Exception as e:
                    logger.error(f"Error processing file {input_file}: {e}")
                    continue
            
            # Write processed conversations
            with jsonlines.open(output_path, mode='w') as writer:
                for conv in processed_conversations:
                    writer.write(conv)
            
            logger.info(f"Processing complete:")
            logger.info(f"Successfully processed: {total_processed}")
            logger.info(f"Failed to process: {total_failed}")
            logger.info(f"Output saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise

def main():
    dataset_path = Path('/home/zahemen/datasets/Financial-QA-10k.csv')
    output_path = Path('/home/zahemen/datasets/finance_qa_conversations.jsonl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processor = FinanceQAProcessor(dataset_path, num_cross_company_samples=5000)
    processor.process_dataset(output_path)

if __name__ == "__main__":
    main()
