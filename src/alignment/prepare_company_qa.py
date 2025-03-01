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
    messages: List[Dict[str, str]]  # Remove prompt field
    metadata: Dict

class FinanceQAProcessor:

    def __init__(self, dataset_path: Path, num_cross_company_samples: int = 2000):
        self.data = pd.read_csv(dataset_path)
        self.num_cross_company_samples = num_cross_company_samples
        self.used_samples = set()  # Track used question-answer pairs
        self.all_tickers = None  # Will store all unique tickers

        #Generate a list of varying system_prompt dicts

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
        
        # Add list-specific system prompts
        self.list_system_prompts = [
            "You are FinSight, an AI trained to provide clear, structured financial opinions & advice. When appropriate, present information in well-organized lists.",
            "You are a finance Bot that excels at breaking down complex information into clear, numbered or bulleted lists when suitable.",
            "You are FinSight, specializing in delivering financial insights through structured responses, including organized lists when beneficial.",
        ]
        
        # Add list response templates
        self.list_templates = [
            "Here are {num} key points about {topic}:\n{items}",
            "Let me break down {topic} into {num} main aspects:\n{items}",
            "Consider these {num} important factors regarding {topic}:\n{items}",
            "{topic} can be understood through these {num} elements:\n{items}",
        ]
        
        # Add list marker patterns
        self.list_markers = [
            r'^\d+\.\s+',  # Numbered lists (1. 2. etc)
            r'^\*\s+',     # Bullet points
            r'^\-\s+',     # Dash lists
            r'^\[\d+\]\s+' # Bracketed numbers
        ]
        
        # Add cross-company comparison questions
        self.cross_company_questions = [
            "How do {company1} and {company2} compare in terms of {aspect}?",
            "What are the key differences between {company1} and {company2} regarding {aspect}?",
            "Can you compare {company1} and {company2}'s performance in {aspect}?",
            "Break down the differences between {company1} and {company2} in {aspect}.",
            "List the main points of comparison between {company1} and {company2} for {aspect}.",
            "What advantages does {company1} have over {company2} in {aspect}?",
            "How do {company1}'s {aspect} stack up against {company2}?",
            "Analyze the strengths of both {company1} and {company2} in {aspect}.",
            "Which company should I invest in: {company1} or {company2}?",
        ]
        
        self.comparison_aspects = [
            "recent developments",
            "key activities", 
            "current status",
            "overall performance",
            "main priorities",
            "notable highlights",
            "general direction",
            "major initiatives",
            "significant changes",
            "important factors",
            "core strengths",
            "key characteristics",
            "main focus areas",
            "strategic elements"
        ]
        
        self.list_question_templates = [
            "What are the top {num} {aspect} of {company}?",
            "List {num} key {aspect} that {company} demonstrates.",
            "Can you break down {num} main {aspect} of {company}?",
            "What {num} factors make {company} stand out in terms of {aspect}?",
            "Identify {num} critical {aspect} that define {company}.",
            "Enumerate {num} significant {aspect} of {company}.",
            "What are {num} notable {aspect} that characterize {company}?",
            "Outline {num} essential {aspect} of {company}.",
        ]
        
        self.list_aspects = [
            "strengths",
            "challenges",
            "growth drivers",
            "risk factors",
            "competitive advantages",
            "market opportunities",
            "strategic initiatives",
            "performance indicators",
            "business segments",
            "revenue sources",
            "operational highlights",
            "investment considerations",
            "market positions",
            "industry trends",
        ]
        
        self.min_words_for_list = 8  # Minimum words required for list items
        
        # Add company fact question templates
        self.company_fact_templates = [
            "Tell me some interesting facts about {company}",
            "What are some key things to know about {company}?",
            "Share some important information about {company}",
            "What should investors know about {company}?",
            "Give me an overview of {company}'s business",
            "What makes {company} stand out in their industry?",
            "What are {company}'s main strengths and characteristics?",
            "Highlight some notable aspects of {company}",
        ]

        # Remove context-related attributes and add company-specific question templates
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
        
        # Add investment comparison questions templates
        self.investment_preference_questions = [
            "Which company should I invest in: {company1} or {company2}?",
            "Is {company1} or {company2} a better investment choice right now?",
            "If you had to pick between {company1} and {company2} for investing, which would you choose?",
            "Should I put my money in {company1} or {company2}?",
            "For long-term growth, would you recommend {company1} or {company2}?",
            "Between {company1} and {company2}, which stock has more potential?",
            "I'm torn between investing in {company1} and {company2}. Which would you suggest?",
            "As an investor, should I focus on {company1} or {company2}?",
            "From an investment standpoint, is {company1} or {company2} the stronger option?"
        ]
        
        # Add investment preference response templates
        self.investment_preference_responses = [
            "Both {company1} and {company2} have their strengths, but {selected_company} might be more suitable based on these factors:\n\n{selected_facts}\n\nThat said, {other_company} also offers some compelling points:\n\n{other_facts}\n\nUltimately, your investment choice should align with your personal financial goals, risk tolerance, and investment timeline. Which aspects are most important to you?",
            
            "When comparing {company1} and {company2}, I would highlight these advantages for {selected_company}:\n\n{selected_facts}\n\nHowever, {other_company} also has notable strengths:\n\n{other_facts}\n\nYour investment decision should reflect your specific financial objectives and risk profile. What particular factors matter most for your investment strategy?",
            
            "While both companies have merit, {selected_company} stands out for these reasons:\n\n{selected_facts}\n\nThat's not to dismiss {other_company}, which offers these potential benefits:\n\n{other_facts}\n\nThe best choice depends entirely on your investment goals and time horizon. What are you primarily looking for in this investment?",
            
            "From an analysis perspective, {selected_company} shows these promising aspects:\n\n{selected_facts}\n\nAlthough {other_company} also demonstrates these positive qualities:\n\n{other_facts}\n\nRemember that past performance doesn't guarantee future results, and your personal investment strategy should guide your decision. What specific outcomes are you hoping to achieve?"
        ]
        
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
        company_name =  self.company_names.get(ticker, ticker)
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
        

    def combine_qa_pairs(self, row1, row2) -> Tuple[str, str]:
        """Combine two Q&A pairs into a single question and answer without context in question"""
        # Combine questions without context
        combined_q = f"{row1['question']} Additionally, {row2['question']}"

        combined_a = np.random.choice([
            f"{row1['answer']} Additionally, {row2['answer']}",
            f"1. {row1['answer']} \n\n2. {row2['answer']}",
            f"{row1['answer']} \n\nAlso, {row2['answer']}",
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

    def create_system_message(self) -> Dict[str, str]:
        """Create a consistent system message without context"""
        return random.choice(self.system_prompt_variations)

    def create_company_fact_question(self, company_name: str) -> str:
        """Create a question asking for facts about a company"""
        return random.choice(self.company_fact_templates).format(company=company_name)

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
            (r'\benterprise\'s\b', f"{company_name}'s"),
            (r'\benterprise\b', company_name),
            (r'\bbusiness\'s\b', f"{company_name}'s"),
            (r'\bbusiness\b', company_name),
            (r'\bcorporation\'s\b', f"{company_name}'s"),
            (r'\bcorporation\b', company_name),
            (r'\borganization\'s\b', f"{company_name}'s"),
            (r'\borganization\b', company_name),
        ]
        
        formatted = question
        for pattern, replacement in patterns:
            formatted = re.sub(pattern, replacement, formatted, flags=re.IGNORECASE)
            
        # Add company name at start if no reference exists
        if company_name.lower() not in formatted.lower():
            formatted = f"For {company_name}, {formatted}"
            
        return formatted.strip()

    def create_multi_turn_conversation(self, company_data: pd.DataFrame, num_turns: int) -> Conversation:
        """Modified to include fact-based questions and remove context"""

        use_starter = random.random() < 0.5 # 50% chance to use starter messages
        if use_starter:
            use_basic_starter = random.random() < 0.3
            if use_basic_starter:
                messages = []
                greeting, response = self.create_greetings()
                usr, ast = self.create_conversation_starter()
                
                # Add system message without context
                messages.append(self.create_system_message())
                messages.extend([greeting, response, usr, ast])
            else:
                intro_generator = IntroDatasetGenerator(None)
                messages = intro_generator.generate_conversation()
        else:
            messages = []

        # Add a fact-based question about the company
        company_name = self.get_company_name(company_data['ticker'].iloc[0])
        fact_question = self.create_company_fact_question(company_name)
        
        # Filter out answers containing "10-K" references
        def is_valid_fact(idx):
            answer = str(company_data.loc[idx, 'answer']).lower()
            return (
                self.can_use_sample(idx) and 
                self.is_valid_list_item(answer) and
                all(fin_term not in answer for fin_term in ["10-k", "10k", "form 10", "10-K", "10-Q", "10-Q"]) 
            )
        # Get filtered indices
        valid_indices = [
            idx for idx in company_data.index 
            if is_valid_fact(idx)
        ]
        
        if len(valid_indices) >= 3:
            fact_indices = random.sample(
                valid_indices, 
                min(random.randint(3, 5), len(valid_indices))
            )
            fact_responses = [company_data.loc[idx, 'answer'] for idx in fact_indices]
            fact_response = self.format_list_response(fact_responses, f"Facts about {company_name}")
            
            ending_vars = [
                "I hope you found these facts helpful.",
                "Any other questions you'd like to ask?",
                "Feel free to inquire about any other details.",
                "Let me know if you need more information.",
                "I'm here to assist with any other queries.",
                "Don't hesitate to ask if you need more insights"
            ]

            should_use_ending = random.random() < 0.5  # 50% chance to add ending

            if should_use_ending:
                fact_response += f"\n\n{random.choice(ending_vars)}"
            messages.extend([
                {"role": "user", "content": fact_question},
                {"role": "assistant", "content": fact_response}
            ])
            
            # Mark these facts as used
            for idx in fact_indices:
                self.mark_sample_used(idx)
        
        # Rest of the conversation remains unchanged
        available_indices = [idx for idx in company_data.index if self.can_use_sample(idx)]
        if len(available_indices) < num_turns:
            num_turns = len(available_indices)
            
        selected_indices = random.sample(available_indices, num_turns)
        
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
                "conversation_id": self.generate_prompt_id()
            }
        )
    def create_cross_company_conversation(
        self,
        company1_data: pd.DataFrame,
        company2_data: pd.DataFrame,
        num_turns: int
    ) -> Conversation:
        """Create a conversation mixing questions from two companies with enhanced list formatting"""
        messages = []
        greeting, response = self.create_greetings()
        usr, ast = self.create_conversation_starter()
        
        # Decide whether to add intro messages or not

        should_add_intro = random.random() < 0.5  # 50% chance to add intro messages
        if should_add_intro:
            intro_generator = IntroDatasetGenerator(None)
            messages.extend(intro_generator.generate_conversation())
        else:
            #Just add a basic system message
            messages.append(self.create_system_message())

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
            
            # Enhanced question generation for comparison
            if i % 2 == 1:  # When we have info from both companies
                # 70% chance to use comparison question
                aspect = random.choice(self.comparison_aspects)
                if random.random() < 0.7:
                    aspect = random.choice(self.comparison_aspects)
                    question = random.choice(self.cross_company_questions).format(
                        company1=self.get_company_name(company1_data['ticker'].iloc[0]),
                        company2=self.get_company_name(company2_data['ticker'].iloc[0]),
                        aspect=aspect
                    )
                else:
                    question = row['question']
                
                # 60% chance to format response as a list
                use_list_format = random.random() < 0.7

                prev_idx = messages[-2]['content']
                prev_row = company1_data[company1_data['question'] == prev_idx].iloc[0]                

                if use_list_format:              
                    # Validate both answers have sufficient words for list items
                    if not (self.is_valid_list_item(str(prev_row['answer'])) and 
                            self.is_valid_list_item(str(row['answer']))):
                        use_list_format = False
                        response = row['answer']
                    else:
                        # Enhanced list formatting with varied structures
                        list_style = random.choice(['number', 'bullet', 'detailed'])
                        
                        if list_style == 'number':
                            response = (
                                f"Here's a comparison of {aspect} between {self.get_company_name(company1_data['ticker'].iloc[0])} "
                                f"and {self.get_company_name(company2_data['ticker'].iloc[0])}:\n\n"
                                f"1. {self.get_company_name(company1_data['ticker'].iloc[0])}:\n"
                                f"   - {prev_row['answer']}\n\n"
                                f"2. {self.get_company_name(company2_data['ticker'].iloc[0])}:\n"
                                f"   - {row['answer']}"
                            )
                        elif list_style == 'bullet':
                            response = (
                                f"Comparing {aspect}:\n\n"
                                f"* {self.get_company_name(company1_data['ticker'].iloc[0])}:\n"
                                f"  - {prev_row['answer']}\n\n"
                                f"* {self.get_company_name(company2_data['ticker'].iloc[0])}:\n"
                                f"  - {row['answer']}"
                            )
                        else:  # detailed
                            response = (
                                f"Detailed comparison of {aspect}:\n\n"
                                f"[{self.get_company_name(company1_data['ticker'].iloc[0])}]\n"
                                f"• {prev_row['answer']}\n\n"
                                f"[{self.get_company_name(company2_data['ticker'].iloc[0])}]\n"
                                f"• {row['answer']}\n\n"
                                f"Key Differences:\n"
                                f"- {random.choice(['Stronger', 'Different', 'Unique'])} focus on specific aspects\n"
                                f"- Distinct approaches to {aspect}"
                            )
                else:
                    response = (
                        f"For {self.get_company_name(company1_data['ticker'].iloc[0])} {prev_row['answer']}. \
                        In comparison, {self.get_company_name(company2_data['ticker'].iloc[0])}: {row['answer']}"
                    )

            else:
                question = row['question']
                response = row['answer']
            
            messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
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
                "company_names": f"{self.get_company_name(company1_data['ticker'].iloc[0])}, {self.get_company_name(company2_data['ticker'].iloc[0])}",
                "combined": False,
                "cross_company": True,
                "has_lists": True,  # Add flag for list formatting
                "filing_year": f"{company1_data['filing'].iloc[0]}, {company2_data['filing'].iloc[0]}" if str(company1_data['filing'].iloc[0]) != str(company2_data['filing'].iloc[0]) else company1_data['filing'].iloc[0],
                "conversation_id": self.generate_prompt_id()
            }
        )

    def format_list_response(self, items: List[str], topic: str) -> str:
        """Format a list response with proper structure"""
        num_items = len(items)
        template = random.choice(self.list_templates)
        
        # Format items with consistent markers
        formatted_items = []
        marker_type = random.choice(['number', 'bullet'])
        
        for i, item in enumerate(items, 1):
            if marker_type == 'number':
                formatted_items.append(f"{i}. {item.strip()}")
            else:
                formatted_items.append(f"* {item.strip()}")
                
        formatted_list = '\n'.join(formatted_items)
        
        return template.format(
            num=num_items,
            topic=topic,
            items=formatted_list
        )

    def is_valid_list_item(self, text: str) -> bool:
        """Check if a text is valid for use in lists"""
        if not isinstance(text, str):
            return False
        # Count words (split by whitespace and filter out empty strings)
        word_count = len([w for w in text.strip().split() if w])
        return word_count >= self.min_words_for_list

    def create_list_conversation(self, company_data: pd.DataFrame) -> Conversation:
        """Create a conversation focusing on enhanced list-based responses"""
        messages = []
        
        # Use list-specific system prompt
        messages.append({
            "role": "system",
            "content": random.choice(self.list_system_prompts)
        })
        
        # Add greeting and conversation starter
        greeting, response = self.create_greetings()
        usr, ast = self.create_conversation_starter()
        messages.extend([greeting, response, usr, ast])
        
        # Generate enhanced list-based QA with word count validation
        available_indices = [
            idx for idx in company_data.index 
            if self.can_use_sample(idx) and self.is_valid_list_item(str(company_data.loc[idx, 'answer']))
        ]
        
        if len(available_indices) >= 3:
            num_items = random.randint(2, 4)
            items = []
            attempts = 0
            max_attempts = 10  # Prevent infinite loops
            
            while len(items) < num_items and attempts < max_attempts:
                if not available_indices:
                    break
                    
                idx = random.choice(available_indices)
                row = company_data.loc[idx]
                answer = str(row['answer'])
                
                if self.is_valid_list_item(answer):
                    items.append(answer)
                    self.mark_sample_used(idx)
                
                available_indices.remove(idx)
                attempts += 1
            
            if len(items) >= 2:  # Ensure we have at least 2 valid items
                aspect = random.choice(self.list_aspects)
                list_question = random.choice(self.list_question_templates).format(
                    num=len(items),
                    aspect=aspect,
                    company=self.get_company_name(company_data['ticker'].iloc[0])
                )
                
                list_response = self.format_list_response(items, f"{self.get_company_name(company_data['ticker'].iloc[0])}'s {aspect}")
                
                messages.extend([
                    {"role": "user", "content": list_question},
                    {"role": "assistant", "content": list_response}
                ])
                
                return Conversation(
                    messages=messages,
                    metadata={
                        "ticker": company_data['ticker'].iloc[0],
                        "company_name": self.get_company_name(company_data['ticker'].iloc[0]),
                        "filing_year": company_data['filing'].iloc[0],
                        "has_lists": True,
                        "conversation_id": self.generate_prompt_id()
                    }
                )
        
        return None  # Return None if we couldn't create a valid list conversation

    def create_investment_preference_conversation(
        self,
        company1_data: pd.DataFrame,
        company2_data: pd.DataFrame
    ) -> Conversation:
        """Create a conversation focused on investment preference between two companies"""
        # Start with system prompt
        messages = [self.create_system_message()]
        
        # Get company names
        company1_name = self.get_company_name(company1_data['ticker'].iloc[0])
        company2_name = self.get_company_name(company2_data['ticker'].iloc[0])
        
        # Create investment preference question
        question = random.choice(self.investment_preference_questions).format(
            company1=company1_name,
            company2=company2_name
        )
        
        # Randomly select which company to favor
        selected_company = random.choice([company1_name, company2_name])
        other_company = company2_name if selected_company == company1_name else company1_name
        
        selected_data = company1_data if selected_company == company1_name else company2_data
        other_data = company2_data if selected_company == company1_name else company1_data
        
        # Find good fact candidates for both companies
        def get_valid_facts(company_data, num_facts=5):
            valid_indices = [
                idx for idx in company_data.index 
                if self.can_use_sample(idx) and 
                self.is_valid_list_item(str(company_data.loc[idx, 'answer']))
            ]
            
            if len(valid_indices) < num_facts:
                num_facts = max(2, len(valid_indices))
                
            if not valid_indices:
                return []
                
            selected_indices = random.sample(valid_indices, min(num_facts, len(valid_indices)))
            facts = [company_data.loc[idx, 'answer'] for idx in selected_indices]
            
            # Mark as used
            for idx in selected_indices:
                self.mark_sample_used(idx)
                
            return facts
            
        # Get facts for both companies
        selected_company_facts = get_valid_facts(selected_data, random.randint(3, 5))
        other_company_facts = get_valid_facts(other_data, random.randint(2, 4))
        
        if not selected_company_facts or not other_company_facts:
            # Fallback to creating a cross-company conversation if we don't have enough facts
            return self.create_cross_company_conversation(company1_data, company2_data, 3)
        
        # Format facts into bullet points
        def format_facts(facts):
            return "\n".join([f"• {fact}" for fact in facts])
            
        selected_facts_formatted = format_facts(selected_company_facts)
        other_facts_formatted = format_facts(other_company_facts)
        
        # Create response using template
        response_template = random.choice(self.investment_preference_responses)
        response = response_template.format(
            company1=company1_name,
            company2=company2_name,
            selected_company=selected_company,
            other_company=other_company,
            selected_facts=selected_facts_formatted,
            other_facts=other_facts_formatted
        )
        
        # Add the question and response to messages
        messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ])
        
        # Define followup questions for both companies
        selected_company_questions = [
            f"Why do you think {selected_company} has better growth potential?",
            f"What makes {selected_company} stand out compared to other companies in the sector?",
            f"What risks should I be aware of when investing in {selected_company}?",
            f"How has {selected_company} performed in recent quarters?",
            f"What factors might affect {selected_company}'s stock price in the near future?"
        ]
        
        other_company_questions = [
            f"What about {other_company}? What are its strengths?",
            f"Are there any reasons I might prefer {other_company} instead?",
            f"What risks would I face investing in {other_company}?",
            f"How has {other_company} been performing recently?",
            f"What market factors would benefit {other_company}?",
            f"Are there any upcoming catalysts for {other_company}?"
        ]
        
        # Determine how many follow-up pairs to include (2-5 pairs)
        num_followups = random.randint(2, 5)
        
        # Function to get valid answer that's more than 5 words
        def get_valid_answer(data_df):
            max_attempts = 10
            attempts = 0
            
            while attempts < max_attempts:
                remaining_indices = [idx for idx in data_df.index if self.can_use_sample(idx)]
                if not remaining_indices:
                    return "Based on the company's financial statements and market position, they've demonstrated consistent performance in key metrics that investors typically look for. Their strategic initiatives and management approach suggest potential for continued growth and adaptation to market changes."
                
                idx = random.choice(remaining_indices)
                answer = str(data_df.loc[idx, 'answer'])

                company_terms = [
                    "company", "business", "firm", "enterprise", "organization", 
                    "corporation", "industry", "market", "sector"
                ]
                
                company_relevant = any(term in answer.lower() for term in company_terms)

                # Check if answer is substantial (more than 5 words), relevant to company, and not raw financial data
                word_count = len(answer.split())
                if word_count > 10 and company_relevant and not answer.startswith('$'):
                    self.mark_sample_used(idx)
                    return answer
                
                attempts += 1
            
            # Fallback response if we can't find a valid answer
            return "The company has shown strong fundamentals in their financial reporting, with promising growth metrics across multiple quarters. Their strategic initiatives align well with current market trends, suggesting potential for continued expansion in their market segment."
        
        # First, add follow-ups for the selected company (always first)
        selected_q = random.choice(selected_company_questions)
        selected_a = get_valid_answer(selected_data)
        
        messages.extend([
            {"role": "user", "content": selected_q},
            {"role": "assistant", "content": selected_a}
        ])
        
        # Then, add follow-ups for the other company
        other_q = random.choice(other_company_questions)
        other_a = get_valid_answer(other_data)
        
        messages.extend([
            {"role": "user", "content": other_q},
            {"role": "assistant", "content": other_a}
        ])
        
        # Add additional follow-up pairs to reach desired conversation length
        for _ in range(num_followups - 2):  # -2 because we already added one pair for each company
            if random.random() < 0.5 and selected_company_questions:
                # Add another selected company question
                q = random.choice(selected_company_questions)
                selected_company_questions.remove(q)  # Avoid repetition
                a = get_valid_answer(selected_data)
                
                messages.extend([
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ])
            elif other_company_questions:
                # Add another other company question
                q = random.choice(other_company_questions)
                other_company_questions.remove(q)  # Avoid repetition
                a = get_valid_answer(other_data)
                
                messages.extend([
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ])
        
        # Add a final question about overall investment strategy
        final_questions = [
            f"Given all this information, how would you recommend I approach investing in either {selected_company} or {other_company}?",
            f"What would be a good investment strategy if I want exposure to both {selected_company} and {other_company}?",
            f"How would you recommend I allocate my portfolio between {selected_company} and {other_company}?",
            f"What time horizon would you recommend for investing in these companies?",
            f"Should I consider dollar-cost averaging into either of these stocks?",
            f"Ultimately, I wanna decide between {selected_company} and {other_company}. What's your take?"
        ]
        
        final_q = random.choice(final_questions)
        
        final_responses = [
            f"Based on our discussion, a balanced approach might work well. {selected_company} appears to offer stronger growth potential, making it suitable for a larger portion of your investment (perhaps 60-70%), while allocating a smaller position (30-40%) to {other_company} could provide diversification benefits. Remember to align this strategy with your overall investment goals, risk tolerance, and time horizon. Regular portfolio reviews would be advisable as market conditions and company performances evolve over time.",
            
            f"I'd suggest considering your investment time horizon first. For long-term growth, {selected_company} might warrant a more substantial allocation in your portfolio. However, maintaining some exposure to {other_company} could help manage sector-specific risks. Consider starting with a 70/30 or 60/40 split favoring your preferred company, and rebalance periodically as their relative valuations and prospects change. Always ensure these investments align with your broader financial strategy.",
            
            f"Given the strengths and challenges of both companies, you might consider a phased approach to investing. Start with a position in your preferred company ({selected_company} based on our analysis), and monitor its performance for 3-6 months. Then gradually build a smaller position in {other_company} to diversify. This strategy allows you to average into positions while staying responsive to changing market conditions and company developments.",
            
            f"For a balanced approach, consider allocating investments proportionally to your conviction level for each company. Based on our discussion, perhaps 65% to {selected_company} and 35% to {other_company} would be reasonable. However, your personal financial situation, investment timeline, and risk tolerance should ultimately guide this decision. Remember that regular portfolio review is essential regardless of your initial allocation.",

            f"Given the information we've discussed, a diversified approach might be beneficial. Consider allocating a larger portion of your investment to {selected_company} due to its growth potential and market position. A smaller allocation to {other_company} could provide diversification benefits and help manage sector-specific risks. Regularly review your portfolio to ensure it aligns with your investment goals and risk tolerance.",

            f"Based on our analysis, a balanced approach could be effective. Allocate a larger portion of your investment to {selected_company} to capitalize on its growth potential. A smaller allocation to {other_company} could provide diversification benefits. Regularly review your portfolio to ensure it aligns with your investment goals and risk tolerance.",

            f"Considering the information we've discussed, a balanced approach might be suitable. Allocate a larger portion of your investment to {selected_company} to benefit from its growth potential. A smaller allocation to {other_company} could provide diversification benefits. Regularly review your portfolio to ensure it aligns with your investment goals and risk tolerance."
        ]
        
        final_a = random.choice(final_responses)
        
        messages.extend([
            {"role": "user", "content": final_q},
            {"role": "assistant", "content": final_a}
        ])
        
        return Conversation(
            messages=messages,
            metadata={
                "ticker": f"{company1_data['ticker'].iloc[0]}, {company2_data['ticker'].iloc[0]}",
                "company_names": f"{company1_name}, {company2_name}",
                "investment_comparison": True,
                "conversation_id": self.generate_prompt_id()
            }
        )

    def process_dataset(self, output_path: Path, max_samples_per_company: int = 20):
        """Process the entire dataset and create variations"""
        processed_conversations = []
        
        # Group by ticker
        grouped = self.data.groupby('ticker')
        self.all_tickers = list(grouped.groups.keys())
        
        for ticker, company_data in grouped:
            logger.info(f"Processing company: {ticker}")
            
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
                        system_msg = self.create_system_message()

                        
                        greeting, response = self.create_greetings()
                        usr, ast = self.create_conversation_starter()
                        
                        messages = [
                            system_msg,
                            greeting, response,
                            usr, ast,
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a}
                        ] if random.random() < 0.8 else [
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
            
            # Add list-based conversations with validation
            num_list_conversations = random.randint(2, 5)
            successful_list_convs = 0
            max_attempts = num_list_conversations * 2  # Allow some extra attempts
            
            for _ in range(max_attempts):
                if successful_list_convs >= num_list_conversations:
                    break
                    
                try:
                    conv = self.create_list_conversation(company_data)
                    if conv is not None:  # Only add if we got a valid conversation
                        processed_conversations.append(conv)
                        successful_list_convs += 1
                except Exception as e:
                    logger.warning(f"Failed to create list conversation for {ticker}: {e}")
                    continue
            
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
        
        # Split the cross-company samples between regular cross-company and investment preference
        num_investment_preference = int(num_cross_company * 0.35)  # Allocate 35% to investment preference
        num_regular_cross = num_cross_company - num_investment_preference
        
        logger.info(f"Generating {num_regular_cross} regular cross-company conversations...")
        for _ in range(num_regular_cross):
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
                logger.info(f"Generated {_}/{num_regular_cross} regular cross-company conversations")
        
        logger.info(f"Generating {num_investment_preference} investment preference conversations...")
        for _ in range(num_investment_preference):
            # Select two random companies
            company1_ticker, company2_ticker = random.sample(self.all_tickers, 2)
            company1_data = self.data[self.data['ticker'] == company1_ticker]
            company2_data = self.data[self.data['ticker'] == company2_ticker]
            
            try:
                available1 = [idx for idx in company1_data.index if self.can_use_sample(idx)]
                available2 = [idx for idx in company2_data.index if self.can_use_sample(idx)]
                if len(available1) >= 3 and len(available2) >= 2:  # Minimum required facts
                    conv = self.create_investment_preference_conversation(
                        company1_data,
                        company2_data
                    )
                    processed_conversations.append(conv)
            except (ValueError, IndexError):
                continue
                
            if _ % 20 == 0:  # Log progress every 20 conversations
                logger.info(f"Generated {_}/{num_investment_preference} investment preference conversations")
        
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
    
    processor = FinanceQAProcessor(dataset_path, num_cross_company_samples=2000)
    processor.process_dataset(output_path)

if __name__ == "__main__":
    main()
