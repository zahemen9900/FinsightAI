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

class FinanceQAGenerator:
    def __init__(self, dataset_path: Path, num_cross_company_samples: int = 2000):
        self.data = pd.read_csv(dataset_path)
        self.num_cross_company_samples = num_cross_company_samples
        self.min_words_for_fact = 5
        
        # Standard system prompt - no context needed
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are FinSight, a professional financial advisor specializing in "
                "company analysis and market insights. Provide clear, factual responses "
                "and use lists when appropriate to organize information."
            )
        }

        # Company fact-finding questions
        self.fact_questions = [
            "What are some key facts about {company}?",
            "Tell me the most important things to know about {company}.",
            "What makes {company} stand out in their industry?",
            "What are {company}'s main business highlights?",
            "Share some interesting facts about {company}'s operations.",
        ]
        
        # Company-specific question templates
        self.company_questions = [
            "How has {company} performed in terms of {metric}?",
            "What strategies has {company} implemented for {aspect}?",
            "How does {company} approach their {area} operations?",
            "What are {company}'s primary revenue sources?",
            "How has {company}'s market position evolved?",
        ]

        # Cross-company comparison templates
        self.comparison_questions = [
            "How do {company1} and {company2} compare in {aspect}?",
            "What differentiates {company1} from {company2} in terms of {aspect}?",
            "Compare the performance of {company1} and {company2} in {aspect}.",
            "What advantages does {company1} have over {company2}?",
            "How do {company1}'s strategies differ from {company2}'s?",
        ]

        # Business aspects for questions
        self.business_aspects = [
            "revenue growth",
            "market share",
            "profitability",
            "operational efficiency",
            "innovation strategy",
            "competitive position",
            "business model",
            "risk management",
            "expansion plans",
            "customer base",
        ]

        # Performance metrics
        self.metrics = [
            "revenue growth",
            "profit margins",
            "market share",
            "operational costs",
            "return on investment",
            "customer acquisition",
        ]

        # Business areas
        self.areas = [
            "global expansion",
            "product development",
            "customer service",
            "supply chain",
            "digital transformation",
            "sustainability",
        ]

    def is_valid_fact(self, text: str) -> bool:
        """Check if a fact meets quality criteria"""
        if not isinstance(text, str):
            return False
        words = [w for w in text.strip().split() if w]
        return len(words) >= self.min_words_for_fact

    def generate_company_facts(self, company_data: pd.DataFrame) -> Dict[str, str]:
        """Generate a fact-based Q&A about a company"""
        facts = []
        available_rows = company_data[
            company_data['answer'].apply(self.is_valid_fact)
        ]
        
        if len(available_rows) >= 3:
            # Select 3-5 high-quality facts
            selected_rows = available_rows.sample(
                n=random.randint(3, min(5, len(available_rows)))
            )
            facts = [row['answer'] for _, row in selected_rows.iterrows()]
            
            # Format as a list response
            response = "Here are the key facts:\n\n" + "\n".join(
                f"{i+1}. {fact}" for i, fact in enumerate(facts)
            )
            
            return {
                "role": "user", 
                "content": random.choice(self.fact_questions).format(
                    company=company_data['ticker'].iloc[0]
                )
            }, {
                "role": "assistant",
                "content": response
            }
        return None

    def generate_company_specific_qa(self, company_data: pd.DataFrame) -> Dict[str, str]:
        """Generate company-specific Q&A pairs"""
        available_rows = company_data[
            company_data['answer'].apply(self.is_valid_fact)
        ]
        
        if not available_rows.empty:
            row = available_rows.sample(n=1).iloc[0]
            
            # Format question with company name
            template = random.choice(self.company_questions)
            question = template.format(
                company=company_data['ticker'].iloc[0],
                metric=random.choice(self.metrics),
                aspect=random.choice(self.business_aspects),
                area=random.choice(self.areas)
            )
            
            return {
                "role": "user",
                "content": question
            }, {
                "role": "assistant",
                "content": row['answer']
            }
        return None

    def generate_comparison_qa(
        self,
        company1_data: pd.DataFrame,
        company2_data: pd.DataFrame
    ) -> Dict[str, str]:
        """Generate cross-company comparison Q&A"""
        # ... implementation continues ...
