import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import random
import json
from dataclasses import dataclass, asdict
import logging
from rich.logging import RichHandler
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

@dataclass
class Conversation:
    prompt: str  # Add prompt field
    messages: List[Dict[str, str]]
    metadata: Dict

class FinanceQAProcessor:
    def __init__(self, dataset_path: Path):
        self.data = pd.read_csv(dataset_path)
        self.used_samples = set()  # Track used question-answer pairs
        self.all_tickers = None  # Will store all unique tickers
        self.system_prompts = [
                "You are a finance Bot trained on a dataset of 10-K filings. You are tasked with answering questions related to the 10-K filings.",
                "You are FinSight, a finance Bot trained on a dataset of 10-K filings. You are tasked with answering questions related to the filings in a professional and concise manner.",
                "You are a finance Bot trained on a dataset of 10-K filings. You are tasked with answering questions related to the 10-K filings in an accurate and concise manner.",
                "You are FinSight, a finance Bot trained on a dataset of 10-K filings of various S&P500 Companies. You are expected to answer questions related to the 10-K filings in a professional and accurate manner.",
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
            "You are an AI assistant specialized in financial analysis. From the company's filing: {context}"
        ]
        
        self.system_prompt_no_context = [
            "You are a finance Bot trained on 10-K filings of various S&P500 Companies.",
            "You are FinSight, an AI trained to analyze and explain company financials.",
            "You are a financial advisor bot trained to answer questions about company financials.",
            "You are an AI assistant specialized in financial analysis and company information."
        ]
        self.conversation_counter = 0  # Add counter for unique IDs
        
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
        combined_a = f"{row1['answer']} Furthermore, {row2['answer']}"
        
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
            "Hello! How can I help you today?",
            "Hi there! What can I assist you with?",
            "Greetings! How may I assist you?",
            "Hey there! How can I help you today?",
            "Good day! What can I assist you with?",
            "Hello there! How may I assist you?",
            "Hi there! How can I help you today?",
            "Howdy! What can I assist you with?",
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
            "Sure, I can help with that.",
            "Of course, I'd be happy to assist.",
            "Absolutely, I'm here to help.WHat's your question?",
            "Yes, I can provide information on that.What would you like to know?",
            "Certainly, I'm ready to assist with your queries",

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
        use_context = random.random() < 0.5
        
        # Add system prompt with/without context
        context = company_data['context'].iloc[0] if use_context else None
        messages.append(self.create_system_message(context, use_context))
        
        messages.extend([greeting, response, usr, ast])
        
        # Rest of conversation without context in questions
        available_indices = company_data.index[~company_data.index.isin(self.used_samples)].tolist()
        if len(available_indices) < num_turns:
            num_turns = len(available_indices)
            
        selected_indices = random.sample(available_indices, num_turns)
        
        for idx in selected_indices:
            row = company_data.loc[idx]
            self.used_samples.add(idx)
            messages.extend([
                {"role": "user", "content": str(row['question'] or "")},  # Convert to string and handle None
                {"role": "assistant", "content": str(row['answer'] or "")}  # Convert to string and handle None
            ])
        
        # Get the first actual question as the prompt
        row = company_data.loc[random.choice(company_data.index[~company_data.index.isin(self.used_samples)])]
        
        return Conversation(
            prompt=row['question'],
            messages=messages,
            metadata={
                "ticker": company_data['ticker'].iloc[0],
                "filing_year": company_data['filing'].iloc[0],
                "has_context": use_context,
                "prompt_id": self.generate_prompt_id()  # Add prompt ID
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
        use_context = random.random() < 0.5
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
        company1_indices = company1_data.index[~company1_data.index.isin(self.used_samples)].tolist()
        company2_indices = company2_data.index[~company2_data.index.isin(self.used_samples)].tolist()
        
        num_turns = min(num_turns, len(company1_indices), len(company2_indices))
        
        for i in range(num_turns):
            # Alternate between companies
            current_data = company1_data if i % 2 == 0 else company2_data
            current_indices = company1_indices if i % 2 == 0 else company2_indices
            
            idx = random.choice(current_indices)
            row = current_data.loc[idx]
            
            # Add question without context
            messages.extend([
                {"role": "user", "content": row['question']},
                {"role": "assistant", "content": row['answer']}
            ])
            self.used_samples.add(idx)
            
            # Remove used index from available indices
            if i % 2 == 0:
                company1_indices.remove(idx)
            else:
                company2_indices.remove(idx)
        
        # Get first question as prompt (without context)
        first_row = company1_data.loc[random.choice(company1_data.index[~company1_data.index.isin(self.used_samples)])]
        
        return Conversation(
            prompt=first_row['question'],  # Just the question without context
            messages=messages,
            metadata={
                "ticker": f"{company1_data['ticker'].iloc[0]}, {company2_data['ticker'].iloc[0]}",
                "combined": False,
                "cross_company": True,
                "has_context": use_context,
                "filing_year": f"{company1_data['filing'].iloc[0]}, {company2_data['filing'].iloc[0]}",
                "prompt_id": self.generate_prompt_id()  # Add prompt ID
            }
        )

    def process_dataset(self, output_path: Path, max_samples_per_company: int = 4):
        """Process the entire dataset and create variations"""
        processed_conversations = []
        
        # Group by ticker
        grouped = self.data.groupby('ticker')
        self.all_tickers = list(grouped.groups.keys())
        
        for ticker, company_data in grouped:
            logger.info(f"Processing company: {ticker}")
            
            # Create some combined Q&A pairs
            system_prompts = [
                "You are a finance Bot trained on a dataset of 10-K filings. You are tasked with answering questions related to the 10-K filings.",
                "You are FinSight, a finance Bot trained on a dataset of 10-K filings. You are tasked with answering questions related to the filings in a professional and concise manner.",
                "You are a finance Bot trained on a dataset of 10-K filings. You are tasked with answering questions related to the 10-K filings in an accurate and concise manner.",
                "You are FinSight, a finance Bot trained on a dataset of 10-K filings of various S&P500 Companies. You are expected to answer questions related to the 10-K filings in a professional and accurate manner.",
            ]

            if len(company_data) >= 2:
                for _ in range(min(2, len(company_data) // 2)):
                    available = company_data.index[~company_data.index.isin(self.used_samples)]
                    if len(available) >= 2:
                        idx1, idx2 = random.sample(list(available), 2)
                        q, a = self.combine_qa_pairs(
                            company_data.loc[idx1],
                            company_data.loc[idx2]
                        )
                        
                        # Decide whether to use context
                        use_context = random.random() < 0.5
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
                        ] if np.random.rand() < 0.5 else [
                            system_msg,
                            usr, ast,
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a}
                        ]
                        
                        processed_conversations.append(Conversation(
                            prompt=q,  # Use the combined question as prompt
                            messages=messages,
                            metadata={
                                "ticker": ticker,
                                "combined": True,
                                "filing_year": company_data['filing'].iloc[0],
                                "prompt_id": self.generate_prompt_id()  # Add prompt ID
                            }
                        ))
                        
                        self.used_samples.update([idx1, idx2])
            # Create multi-turn conversations
            num_conversations = random.randint(3, max_samples_per_company)
            for _ in range(num_conversations):
                num_turns = random.randint(3, 5)
                try:
                    conv = self.create_multi_turn_conversation(company_data, num_turns)
                    processed_conversations.append(conv)
                except ValueError:
                    continue  # Skip if not enough samples left
                    
        # Now create cross-company conversations
        logger.info("Generating cross-company conversations...")
        num_cross_company = 300  # Number of cross-company conversations to generate
        
        for _ in range(num_cross_company):
            # Select two random companies
            company1_ticker, company2_ticker = random.sample(self.all_tickers, 2)
            company1_data = self.data[self.data['ticker'] == company1_ticker]
            company2_data = self.data[self.data['ticker'] == company2_ticker]
            
            # Generate a conversation with 3-5 turns
            num_turns = random.randint(3, 5)
            try:
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
        """Save processed conversations to JSON file"""
        output_data = [asdict(conv) for conv in conversations]
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved processed dataset to {output_path}")

def main():
    dataset_path = Path('/home/zahemen/datasets/Financial-QA-10k.csv')
    output_path = Path('/home/zahemen/datasets/finance_qa_conversations.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processor = FinanceQAProcessor(dataset_path)
    processor.process_dataset(output_path)

if __name__ == "__main__":
    main()
