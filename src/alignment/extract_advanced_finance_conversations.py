import re
import json
import logging
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from tqdm import tqdm
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")

@dataclass
class AdvancedQAPair:
    """Represents a question-answer pair with metadata."""
    question: str
    answer: str
    category: str  # e.g., 'investment', 'corporate', 'market', etc.
    has_formula: bool
    has_steps: bool
    complexity: str  # 'basic', 'intermediate', 'advanced'
    topics: List[str]

class AdvancedFinanceConversationExtractor:
    """Extract and process advanced finance conversations."""
    
    def __init__(
        self,
        input_file: str = "/home/zahemen/datasets/advanced_finance_questions.txt",
        output_file: str = "/home/zahemen/datasets/sft_datasets/advanced_finance_conversations.jsonl",
        min_turns: int = 3,
        max_turns: int = 8,
        max_reuses: int = 15,
        target_conversations: int = 1000,
        use_progress: bool = True,
        diversify_currency: bool = True,  # New parameter to control currency diversification
        introduce_errors: bool = True,    # New parameter to control error introduction
        error_rate: float = 0.3           # Percentage of questions that will have errors
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.max_reuses = max_reuses
        self.target_conversations = target_conversations
        self.use_progress = use_progress
        self.qa_usage_counter = {}
        self.conversation_counter = 0
        
        # System prompt variations focused on structured response style
        self.system_prompt_variations = [
            "You are FinSight, a financial advisor focused on providing clear, structured explanations. "
            "Always break down complex concepts into steps and include relevant formulas when appropriate.",
            
            "As FinSight, provide detailed financial analysis using a structured approach. "
            "When applicable, organize your responses with clear sections, bullet points, and formulas.",
            
            "You are FinSight, an AI financial expert that excels at explaining complex financial concepts. "
            "Present information in a clear, organized manner with numbered steps and formulas when relevant.",
            
            "You are FinSight, a financial education assistant. Explain concepts thoroughly using examples, "
            "comparisons, and structured breakdowns. Include calculations and formulas when they help illustrate points.",
            
            "As FinSight, your goal is to make financial concepts accessible to everyone. Use clear language, "
            "step-by-step explanations, and visual organization techniques like bullet points and numbered lists.",
            
            "You are FinSight, a data-driven financial analysis assistant. Present information with logical flow, "
            "reference relevant metrics, and provide formulas that support your analysis when appropriate.",
            
            "As FinSight, focus on practical financial advice with clear reasoning. Structure your responses "
            "with organized sections, highlight key points, and include calculations that demonstrate concepts.",
            
            "You are FinSight, a balanced financial advisor. Present multiple perspectives when relevant, "
            "use structured explanations with clear headings, and include mathematical expressions when necessary."
        ]
        
        # Categories and their complexities
        self.categories = {
            "corporate": ["WACC", "capital structure", "valuation", "enterprise value", "dividend policy"],
            "investment": ["portfolio", "stock", "bond", "ETF", "diversification"],
            "market": ["analysis", "P/E ratio", "market cap", "dividend yield"],
            "personal": ["budget", "savings", "retirement", "debt"],
            "tax": ["tax", "deduction", "IRA", "401(k)"]
        }
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Financial topic keywords for topic extraction
        self.topic_keywords = {
            "investment": [
                "invest", "stock", "bond", "etf", "fund", "portfolio", "dividend", 
                "yield", "return", "asset", "allocation", "diversification"
            ],
            "retirement": [
                "retire", "401k", "ira", "pension", "social security", "roth"
            ],
            "personal_finance": [
                "budget", "saving", "emergency fund", "income", "expense", "debt", "credit"
            ],
            "tax": [
                "tax", "deduction", "credit", "write-off", "irs", "withholding", "capital gain"
            ],
            "corporate_finance": [
                "wacc", "capital structure", "enterprise value", "dcf", "discounted cash flow",
                "ebitda", "roe", "roa", "fcf", "free cash flow", "dividend policy"
            ],
            "market_analysis": [
                "p/e ratio", "market cap", "technical analysis", "fundamental analysis", 
                "sector", "industry", "growth rate", "valuation", "multiple"
            ],
            "options_derivatives": [
                "option", "call", "put", "strike price", "expiration", "premium", "futures", 
                "forward", "swap", "hedge"
            ],
            "real_estate": [
                "mortgage", "property", "real estate", "rent", "landlord", "tenant", "reit"
            ],
            "banking": [
                "bank", "interest rate", "loan", "deposit", "checking", "saving account",
                "cd", "certificate of deposit"
            ],
            "economics": [
                "inflation", "gdp", "recession", "economic", "federal reserve", "monetary policy",
                "fiscal policy", "unemployment"
            ]
        }
        
        # Follow-up question templates for making multi-turn conversations
        self.follow_up_templates = {
            "investment": [
                "What allocation would you recommend for {topic} in a balanced portfolio?",
                "How does {topic} perform during periods of high inflation?",
                "What are the tax implications of investing in {topic}?",
                "What's a reasonable expected return for {topic} over a 10-year horizon?",
                "How does {topic} compare to index fund investing?"
            ],
            "retirement": [
                "When should I start taking withdrawals from my {topic}?",
                "What are the penalties for early withdrawals from {topic}?",
                "How much should I be contributing to my {topic} at my age?",
                "Can you explain the tax advantages of {topic}?",
                "How should my {topic} allocation change as I get closer to retirement?"
            ],
            "personal_finance": [
                "What percentage of my income should go toward {topic}?",
                "How can I improve my {topic} situation?",
                "What's a good strategy for managing {topic} on a variable income?",
                "How does {topic} affect my credit score?",
                "What tools do you recommend for tracking {topic}?"
            ],
            "tax": [
                "Are there any deductions related to {topic} I should know about?",
                "How can I minimize taxes on my {topic}?",
                "When should I consult a tax professional about {topic}?",
                "How do tax laws for {topic} differ across states?",
                "Can you explain how {topic} is taxed differently from regular income?"
            ],
            "corporate_finance": [
                "How is {topic} calculated?",
                "What's a good benchmark for {topic} in the technology sector?",
                "How does {topic} affect investment decisions?",
                "Can you explain how changes in interest rates impact {topic}?",
                "How do analysts use {topic} to compare companies?"
            ],
            "market_analysis": [
                "What does a high {topic} indicate about a company?",
                "How reliable is {topic} as a predictor of future performance?",
                "How does {topic} vary across different industries?",
                "Can {topic} be manipulated by companies?",
                "What other metrics should be considered alongside {topic}?"
            ],
            "generic": [
                "Could you elaborate on the risks associated with this approach?",
                "How would your recommendation change for someone with a lower risk tolerance?",
                "What are some common misconceptions about this topic?",
                "Can you provide a real-world example of how this works?",
                "Are there any alternatives I should consider?",
                "How has this strategy performed historically?",
                "What factors might cause this advice to change in the future?",
                "Could you explain the underlying principles in more detail?"
            ]
        }

        self.diversify_currency = diversify_currency
        self.currency_options = [
            {"name": "dollars", "symbol": "$", "regex": r'(\$|\bdollars?\b)'},
            {"name": "pounds", "symbol": "£", "regex": r'(£|\bpounds?\b)'},
            {"name": "euros", "symbol": "€", "regex": r'(€|\beuros?\b)'},
            {"name": "cedis", "symbol": "₵", "regex": r'(₵|\bcedis\b)'},
        ]
        self.currency_distribution = {i: 0 for i, _ in enumerate(self.currency_options)}

        self.introduce_errors = introduce_errors
        self.error_rate = error_rate

        # List of common company names to detect in questions
        self.company_names = [
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
        ]

    def extract_qa_pairs(self) -> List[AdvancedQAPair]:
        """Extract Q&A pairs while preserving formatting and structure."""
        qa_pairs = []
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # More robust pattern to extract QA pairs
            qa_pattern = re.compile(r'Question\s+\d+:(.*?)Answer:(.*?)(?=-{80}|\Z)', re.DOTALL)
            matches = qa_pattern.findall(content)
            
            # Process extracted matches
            for question_text, answer_text in tqdm(matches, desc="Extracting QA pairs"):
                question = question_text.strip()
                answer = answer_text.strip()
                
                # Skip empty pairs
                if not question or not answer:
                    continue
                
                # Analyze the QA pair
                category = self._detect_category(question)
                has_formula = bool(re.search(r'[=+\-*/÷×]', answer))
                has_steps = bool(re.search(r'\d+\.|•|\*|\-|\b(Step|First|Second|Third|Finally)\b', answer))
                complexity = self._determine_complexity(question, answer)
                topics = self._extract_topics(question)
                
                qa_pairs.append(AdvancedQAPair(
                    question=question,
                    answer=answer,
                    category=category,
                    has_formula=has_formula,
                    has_steps=has_steps,
                    complexity=complexity,
                    topics=topics
                ))
            
        except Exception as e:
            logger.error(f"Error extracting QA pairs: {e}")
            logger.error("Try running fix_question_format.py on the input file first")
            return []
        
        if not qa_pairs:
            logger.warning("No QA pairs extracted. Check input file format or run fix_question_format.py")
            try:
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    sample = f.read(500)  # Read first 500 chars for debugging
                    logger.debug(f"Sample of file content:\n{sample}")
            except Exception as e:
                logger.error(f"Error reading sample: {e}")
        else:
            logger.info(f"Extracted {len(qa_pairs)} QA pairs")
        
        return qa_pairs

    def _detect_category(self, text: str) -> str:
        """Detect the category of a question based on keywords."""
        text_lower = text.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                return category
        
        return "general"

    def _determine_complexity(self, question: str, answer: str) -> str:
        """Determine the complexity level of a Q&A pair."""
        # Analyze various factors
        word_count = len(answer.split())
        has_formulas = bool(re.search(r'[=+\-*/÷×]', answer))
        has_technical_terms = len(re.findall(r'\b(WACC|DCF|EBITDA|ROE|CAPM)\b', answer))
        has_steps = bool(re.search(r'\d+\.|•|\*|Step', answer))
        
        # Score complexity
        complexity_score = 0
        complexity_score += word_count // 100  # Length factor
        complexity_score += 2 if has_formulas else 0  # Formula complexity
        complexity_score += has_technical_terms  # Technical terminology
        complexity_score += 1 if has_steps else 0  # Structured steps
        
        # Determine level
        if complexity_score <= 2:
            return "basic"
        elif complexity_score <= 4:
            return "intermediate"
        else:
            return "advanced"

    def _extract_topics(self, text: str) -> List[str]:
        """Extract relevant financial topics from text."""
        text_lower = text.lower()
        detected_topics = []
        
        # Check for each topic category based on keywords
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        # Extract specific topics from the text (like "ETFs", "401k", etc.)
        specific_topics = []
        for topic_category, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower and len(keyword) > 3:  # Avoid short general terms
                    # Don't add duplicates or very similar terms
                    if keyword not in specific_topics and not any(keyword in t for t in specific_topics):
                        specific_topics.append(keyword)
        
        # If no topics were detected, use "general_finance"
        if not detected_topics:
            detected_topics.append("general_finance")
            
        return detected_topics + specific_topics[:3]  # Limit specific topics to top 3

    def generate_conversation_id(self) -> str:
        """Generate a unique conversation identifier."""
        unique_string = f"adv_finance_{self.conversation_counter}_{random.getrandbits(32)}"
        conv_id = hashlib.sha256(unique_string.encode()).hexdigest()[:16]
        self.conversation_counter += 1
        return conv_id

    def _generate_follow_up(self, qa_pair: AdvancedQAPair) -> str:
        """Generate a relevant follow-up question based on topics."""
        # If the QA pair has identified topics, use those to generate a follow-up
        if not qa_pair.topics:
            return random.choice(self.follow_up_templates["generic"])
        
        # Try to use the detected category or the first topic
        category = qa_pair.category
        if category not in self.follow_up_templates:
            category = qa_pair.topics[0] if qa_pair.topics else "generic"
        
        if category not in self.follow_up_templates:
            category = "generic"
            
        # Get templates for the chosen category
        templates = self.follow_up_templates[category]
        
        # Extract potential topic terms from the question and answer
        text = qa_pair.question + " " + qa_pair.answer
        text_lower = text.lower()
        
        # Find specific financial terms/phrases that can be used in follow-ups
        potential_terms = []
        for topic_category, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower and len(keyword) > 3:
                    potential_terms.append(keyword)
        
        # If we found terms, randomly select one
        selected_term = random.choice(potential_terms) if potential_terms else "this strategy"
        
        # Pick a template and format it
        template = random.choice(templates)
        if "{topic}" in template:
            return template.format(topic=selected_term)
        
        return template

    def _convert_currency(self, text: str, from_regex: str, to_symbol: str, to_name: str, use_symbol_in_question: bool) -> str:
        """
        Convert currency references in text from one currency to another.
        With exceptions for "dollar-cost" and company-related questions.
        
        Args:
            text: The text to modify
            from_regex: Regex pattern to match the original currency
            to_symbol: Target currency symbol
            to_name: Target currency name
            use_symbol_in_question: Whether to use symbol or name in questions
            
        Returns:
            Modified text with new currency
        """
        # First check if this is a company-related question
        # If it contains any company names, don't convert currency at all
        if any(company.lower() in text.lower() for company in self.company_names):
            return text
        
        # Check for special case "dollar-cost" which shouldn't be replaced
        # Preserve "dollar-cost" term before any replacements
        text = text.replace("dollar-cost", "DOLLARCOSTPLACEHOLDER")
        text = text.replace("Dollar-cost", "DOLLARCOSTPLACEHOLDER_CAP")
        text = text.replace("DOLLAR-COST", "DOLLARCOSTPLACEHOLDER_ALLCAP")
        
        # Check if this is a question (heuristic)
        is_question = text.strip().endswith('?') or len(text.split()) < 30
        
        # Preserve original text if no currency found (to avoid adding metadata for currency that isn't used)
        has_currency_reference = bool(re.search(r'(\$|\bdollars?\b)', text))
        if not has_currency_reference:
            # Restore "dollar-cost" placeholders before returning
            text = text.replace("DOLLARCOSTPLACEHOLDER", "dollar-cost")
            text = text.replace("DOLLARCOSTPLACEHOLDER_CAP", "Dollar-cost")
            text = text.replace("DOLLARCOSTPLACEHOLDER_ALLCAP", "DOLLAR-COST")
            return text
        
        # For questions, use either symbol or name based on parameter
        if is_question:
            # Replace currency symbol/name with appropriate replacement
            if use_symbol_in_question:
                # Convert all currency references to symbol
                text = re.sub(from_regex, to_symbol, text)
            else:
                # Special handling for "$X" format vs "X dollars" format
                text = re.sub(r'\$(\d[\d,]*)', r'\1 ' + to_name, text)  # $100 -> 100 euros
                text = re.sub(r'\b(dollar|dollars)\b', to_name, text)    # dollars -> euros
        else:
            # For answers, always use symbol - but match the currency from the question!
            text = re.sub(r'\$(\d[\d,]*)', to_symbol + r'\1', text)  # $100 -> €100
            text = re.sub(r'(\d[\d,]*)\s+\b(dollar|dollars)\b', to_symbol + r'\1', text)  # 100 dollars -> €100
            text = re.sub(r'\b(dollar|dollars)\b', to_name, text)  # dollars -> euros
        
        # Restore "dollar-cost" term in all its variations
        text = text.replace("DOLLARCOSTPLACEHOLDER", "dollar-cost")
        text = text.replace("DOLLARCOSTPLACEHOLDER_CAP", "Dollar-cost")
        text = text.replace("DOLLARCOSTPLACEHOLDER_ALLCAP", "DOLLAR-COST")
        
        return text

    def _fix_dollar_cost_terms(self, text: str) -> str:
        """
        Fix any incorrectly converted 'dollar-cost' terms in the text.
        This should be called at the final stage before using text in a conversation.
        
        Args:
            text: The text to fix
            
        Returns:
            Text with dollar-cost terms properly fixed
        """
        # Replace any incorrectly converted terms with the correct "dollar-cost" 
        text = text.replace("cedis-cost", "dollar-cost")
        text = text.replace("pounds-cost", "dollar-cost")
        text = text.replace("euros-cost", "dollar-cost")
        text = text.replace("Cedis-cost", "Dollar-cost")
        text = text.replace("Pounds-cost", "Dollar-cost")
        text = text.replace("Euros-cost", "Dollar-cost")
        text = text.replace("CEDIS-COST", "DOLLAR-COST")
        text = text.replace("POUNDS-COST", "DOLLAR-COST")
        text = text.replace("EUROS-COST", "DOLLAR-COST")
        text = text.replace("cedis-Cost", "Dollar-Cost")
        text = text.replace("pounds-Cost", "Dollar-Cost")
        text = text.replace("euros-Cost", "Dollar-Cost")
        
        return text

    def _introduce_random_errors(self, text: str) -> str:
        """
        Deliberately introduce errors to text to improve model robustness.
        Applies up to 2 types of the following errors:
        1. Remove punctuation
        2. Convert to lowercase
        3. Convert to uppercase
        4. Distort spelling by omitting letters
        
        Args:
            text: Original text to modify
            
        Returns:
            Modified text with introduced errors
        """
        if not self.introduce_errors or random.random() > self.error_rate:
            return text
        
        # Choose how many error types to apply (1 or 2)
        num_errors = random.randint(1, 2)
        
        # Select random error types
        error_types = random.sample([
            "remove_punctuation",
            "lowercase",
            "uppercase",
            "distort_spelling"
        ], num_errors)
        
        modified_text = text
        
        for error_type in error_types:
            if error_type == "remove_punctuation":
                # Remove most punctuation except apostrophes in words
                modified_text = re.sub(r'[.,?!;:"()\[\]{}]', '', modified_text)
            
            elif error_type == "lowercase":
                modified_text = modified_text.lower()
            
            elif error_type == "uppercase":
                modified_text = modified_text.upper()
            
            elif error_type == "distort_spelling":
                words = modified_text.split()
                distorted_words = []
                
                for word in words:
                    # Only distort words that are at least 4 characters
                    if len(word) >= 4 and random.random() < 0.4:
                        # Remove 1-2 random letters (not first or last)
                        if len(word) > 4:
                            positions_to_remove = random.sample(range(1, len(word) - 1), min(2, len(word) - 2))
                            distorted_word = ''.join([char for idx, char in enumerate(word) if idx not in positions_to_remove])
                            distorted_words.append(distorted_word)
                        else:
                            # For shorter words, just remove one letter
                            position = random.randint(1, len(word) - 2)
                            distorted_word = word[:position] + word[position+1:]
                            distorted_words.append(distorted_word)
                    else:
                        distorted_words.append(word)
                        
                modified_text = ' '.join(distorted_words)
        
        return modified_text

    def create_structured_conversation(self, 
                                      qa_pairs: List[AdvancedQAPair],
                                      complexity_preference: str = None,
                                      category_focus: str = None) -> Dict:
        """Create a structured conversation with controlled complexity and focus."""
        # Special case for single QA pair - create a simple conversation
        if len(qa_pairs) == 1:
            logger.warning("Only one QA pair available - creating single-turn conversation")
            qa = qa_pairs[0]
            system_prompt = random.choice(self.system_prompt_variations)
            
            # Before applying currency conversion, apply error introduction to question
            question = self._introduce_random_errors(qa.question)
            
            conversation = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},  # Use potentially modified question
                    {"role": "assistant", "content": qa.answer}
                ],
                "metadata": {
                    "source": "advanced_finance_qa",
                    "conversation_id": self.generate_conversation_id(), # Move ID into metadata
                    "turns": 1,
                    "categories": qa.category,  # Use string directly instead of list
                    "complexity": qa.complexity,
                    "topics": ", ".join(qa.topics),  # Convert list to string
                    "has_formulas": str(qa.has_formula).lower(),  # Convert boolean to string
                    "has_steps": str(qa.has_steps).lower(),  # Convert boolean to string
                    "single_qa_mode": "true"  # Use string instead of boolean
                }
            }
            
            # Add currency diversification if enabled
            if self.diversify_currency:
                # Check if the question actually contains currency references before applying conversion
                has_currency = bool(re.search(r'(\$|\bdollars?\b)', question))
                
                if has_currency:
                    # Select least used currency to maintain even distribution
                    currency_idx = min(self.currency_distribution, key=self.currency_distribution.get)
                    currency = self.currency_options[currency_idx]
                    self.currency_distribution[currency_idx] += 1
                    
                    # Decide whether to use symbol or name in questions (alternate)
                    use_symbol = random.choice([True, False])
                    
                    # Convert currency in question and answer
                    new_question = self._convert_currency(
                        question,  # Use the potentially modified question
                        r'(\$|\bdollars?\b)', 
                        currency["symbol"], 
                        currency["name"],
                        use_symbol
                    )
                    
                    new_answer = self._convert_currency(
                        qa.answer,
                        r'(\$|\bdollars?\b)',
                        currency["symbol"],
                        currency["name"],
                        False  # Always use symbols in answers
                    )
                    
                    # Only update metadata if we actually converted something
                    if new_question != question or new_answer != qa.answer:
                        # Update conversation with new currency
                        conversation["messages"][1]["content"] = self._fix_dollar_cost_terms(new_question)
                        conversation["messages"][2]["content"] = self._fix_dollar_cost_terms(new_answer)
                        conversation["metadata"]["currency"] = currency["name"]
                    
                return conversation
        
        # Normal case - multiple QA pairs
        # Filter QA pairs based on preferences
        filtered_pairs = qa_pairs.copy()
        
        # If we have enough pairs total, apply filtering. Otherwise, skip filtering.
        if len(filtered_pairs) >= self.min_turns * 2:
            if complexity_preference:
                complexity_filtered = [qa for qa in filtered_pairs if qa.complexity == complexity_preference]
                if len(complexity_filtered) >= self.min_turns:
                    filtered_pairs = complexity_filtered
            
            if category_focus:
                category_filtered = [qa for qa in filtered_pairs if qa.category == category_focus 
                                 or category_focus in qa.topics]
                if len(category_filtered) >= self.min_turns:
                    filtered_pairs = category_filtered
        
        # Select available QA pairs (not used too many times)
        available_indices = [i for i, qa in enumerate(filtered_pairs) 
                            if self.qa_usage_counter.get(i, 0) < self.max_reuses]
        
        # If we don't have enough unused pairs, try using pairs that are less used
        if len(available_indices) < self.min_turns:
            # Fall back - use any pairs with the lowest usage count
            if filtered_pairs:
                min_usage = min(self.qa_usage_counter.get(i, 0) for i in range(len(filtered_pairs)))
                available_indices = [i for i, qa in enumerate(filtered_pairs) 
                                   if self.qa_usage_counter.get(i, 0) == min_usage]
        
        # If still not enough, just fail
        if len(available_indices) < self.min_turns:
            logger.debug(f"Only {len(available_indices)} QA pairs available, need at least {self.min_turns}")
            return None
        
        # Determine number of turns for this conversation
        adjusted_max_turns = min(self.max_turns, len(available_indices))
        adjusted_min_turns = min(self.min_turns, adjusted_max_turns)
        
        # This will now always provide a valid range
        num_turns = random.randint(adjusted_min_turns, adjusted_max_turns)
        
        # Choose system prompt
        system_prompt = random.choice(self.system_prompt_variations)
        
        # Initialize conversation with starting QA pair
        start_idx = random.choice(available_indices)
        self.mark_qa_pair_used(start_idx)
        start_qa = filtered_pairs[start_idx]
        
        # Apply error introduction to initial question
        initial_question = self._introduce_random_errors(start_qa.question)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_question},  # Use potentially modified question
            {"role": "assistant", "content": start_qa.answer}
        ]
        
        # Track what pairs have been used in this conversation to avoid repetition
        used_in_conv = {start_idx}
        
        # Add additional turns using follow-up questions
        for turn in range(num_turns - 1):
            # Generate a follow-up and possibly introduce errors
            follow_up = self._generate_follow_up(start_qa)
            follow_up = self._introduce_random_errors(follow_up)
            
            messages.append({"role": "user", "content": follow_up})
            
            # Find an appropriate response from the QA pairs
            # Prioritize unused pairs and those matching the category/topic
            candidates = []
            for i, qa in enumerate(filtered_pairs):
                if i not in used_in_conv and self.qa_usage_counter.get(i, 0) < self.max_reuses:
                    # Score candidate based on relevance to follow-up
                    relevance = 0
                    
                    # Check if topics match
                    follow_up_topics = self._extract_topics(follow_up)
                    matching_topics = len(set(qa.topics).intersection(follow_up_topics))
                    relevance += matching_topics * 2
                    
                    # Check for keyword matches
                    if any(word in follow_up.lower() for word in qa.question.lower().split()):
                        relevance += 1
                        
                    # Check complexity matches previous answer complexity
                    if qa.complexity == start_qa.complexity:
                        relevance += 1
                        
                    candidates.append((i, qa, relevance))
            
            # Sort by relevance score (descending) and pick top candidate
            if candidates:
                candidates.sort(key=lambda x: x[2], reverse=True)
                next_idx, next_qa, _ = candidates[0]
            else:
                # Fall back to any unused QA pair if no good candidates
                unused_indices = [i for i, qa in enumerate(filtered_pairs) 
                                if i not in used_in_conv and self.qa_usage_counter.get(i, 0) < self.max_reuses]
                
                if not unused_indices:
                    # If no unused pairs, break the conversation
                    break
                
                next_idx = random.choice(unused_indices)
                next_qa = filtered_pairs[next_idx]
            
            # Mark the QA pair as used and add to conversation
            self.mark_qa_pair_used(next_idx)
            used_in_conv.add(next_idx)
            
            messages.append({"role": "assistant", "content": next_qa.answer})
            
            # Update the reference QA pair for the next follow-up
            start_qa = next_qa
        
        # Create the conversation object
        # Get categories and topics from all QA pairs in the conversation
        categories_set = set([qa.category for qa in [start_qa] + 
                            [filtered_pairs[i] for i in used_in_conv if i != start_idx]])
        all_topics = sum([qa.topics for qa in [start_qa] + 
                        [filtered_pairs[i] for i in used_in_conv if i != start_idx]], [])
        has_formulas = any(qa.has_formula for qa in [start_qa] + 
                         [filtered_pairs[i] for i in used_in_conv if i != start_idx])
        has_steps = any(qa.has_steps for qa in [start_qa] + 
                       [filtered_pairs[i] for i in used_in_conv if i != start_idx])
        
        # Convert to strings
        categories_str = ", ".join(categories_set)
        topics_str = ", ".join(set(all_topics))
        
        # Create the conversation object
        conversation = {
            "messages": messages,
            "metadata": {
                "source": "advanced_finance_qa",
                "conversation_id": self.generate_conversation_id(), # Move ID into metadata
                "turns": len(messages) // 2,  # Number of turns
                "categories": categories_str,  # String instead of list
                "complexity": start_qa.complexity,
                "topics": topics_str,  # String instead of list
                "has_formulas": str(has_formulas).lower(),  # String instead of boolean
                "has_steps": str(has_steps).lower()  # String instead of boolean
            }
        }
        
        # Add currency diversification for multi-turn conversations if enabled
        if self.diversify_currency:
            # Check if any message in the conversation actually contains currency references
            has_currency_refs = False
            contains_company_refs = False
            
            # First check if any message contains company references
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    if any(company.lower() in msg["content"].lower() for company in self.company_names):
                        contains_company_refs = True
                        break
                        
            # If the conversation mentions companies, skip currency diversification
            if contains_company_refs:
                return conversation
                
            # Otherwise continue with regular currency diversification
            for msg in messages:
                if msg["role"] == "user" and re.search(r'(\$|\bdollars?\b)', msg["content"]):
                    has_currency_refs = True
                    break
            
            if has_currency_refs:
                # Select currency for this conversation
                currency_idx = min(self.currency_distribution, key=self.currency_distribution.get)
                currency = self.currency_options[currency_idx]
                self.currency_distribution[currency_idx] += 1
                
                # Decide whether to use symbol or name in questions
                use_symbol = random.choice([True, False])
                
                # Track if any messages were actually converted
                actually_converted = False
                
                # Update messages with converted currency
                for i, msg in enumerate(messages):
                    original_content = msg["content"]
                    
                    if msg["role"] == "user":
                        current_text = msg["content"]  # Already has errors introduced
                        messages[i]["content"] = self._convert_currency(
                            current_text,
                            r'(\$|\bdollars?\b)',
                            currency["symbol"],
                            currency["name"],
                            use_symbol
                        )
                        # Apply dollar-cost term fixing
                        messages[i]["content"] = self._fix_dollar_cost_terms(messages[i]["content"])
                        # Check if any conversion happened
                        if messages[i]["content"] != original_content:
                            actually_converted = True
                            
                    elif msg["role"] == "assistant":
                        messages[i]["content"] = self._convert_currency(
                            msg["content"],
                            r'(\$|\bdollars?\b)',
                            currency["symbol"],
                            currency["name"],
                            False  # Always use symbols in answers
                        )
                        # Apply dollar-cost term fixing
                        messages[i]["content"] = self._fix_dollar_cost_terms(messages[i]["content"])
                        # Check if any conversion happened
                        if messages[i]["content"] != original_content:
                            actually_converted = True
                
                # Only add currency to metadata if conversions actually happened
                if actually_converted:
                    # Add currency information to metadata
                    conversation["metadata"]["currency"] = currency["name"]
            
        return conversation

    def mark_qa_pair_used(self, qa_index: int) -> None:
        """Mark a QA pair as used by incrementing its usage counter."""
        self.qa_usage_counter[qa_index] = self.qa_usage_counter.get(qa_index, 0) + 1

    def validate_currency_consistency(self, conversation: Dict) -> Dict:
        """
        Check and fix currency consistency in a conversation.
        Ensures currencies used in questions match those used in answers.
        """
        if "metadata" not in conversation or "currency" not in conversation["metadata"]:
            return conversation  # No currency to validate
        
        messages = conversation["messages"]
        currency_name = conversation["metadata"]["currency"]
        currency_info = None
        
        # Find the currency info based on name
        for idx, curr in enumerate(self.currency_options):
            if curr["name"] == currency_name:
                currency_info = curr
                break
                
        if not currency_info:
            # Currency not found in options, remove it from metadata
            del conversation["metadata"]["currency"]
            return conversation
            
        # Check if user messages actually contain the currency
        has_user_currency = False
        for msg in messages:
            if msg["role"] == "user" and (
                currency_info["symbol"] in msg["content"] or 
                currency_info["name"] in msg["content"]
            ):
                has_user_currency = True
                break
        
        if not has_user_currency:
            # Currency not used in user messages, remove from metadata
            del conversation["metadata"]["currency"]
            return conversation
            
        # Ensure assistant responses use matching currency
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                # Replace any dollar symbols with the correct currency symbol
                if "$" in msg["content"] and currency_info["symbol"] != "$":
                    # Replace dollar symbols with the correct currency symbol
                    messages[i]["content"] = re.sub(
                        r'\$(\d[\d,\.]*)', 
                        currency_info["symbol"] + r'\1', 
                        msg["content"]
                    )
                    
                # Replace any "dollars" text with the correct currency name
                if "dollar" in msg["content"].lower() and currency_info["name"] != "dollars":
                    messages[i]["content"] = re.sub(
                        r'\b(dollar|dollars)\b', 
                        currency_info["name"], 
                        messages[i]["content"], 
                        flags=re.IGNORECASE
                    )
                    
                # Fix any broken dollar-cost terms
                messages[i]["content"] = self._fix_dollar_cost_terms(messages[i]["content"])
        
        conversation["messages"] = messages
        return conversation

    def generate_conversations(self, qa_pairs: List[AdvancedQAPair]) -> List[Dict]:
        """Generate multiple conversations with varying characteristics."""
        conversations = []
        
        # Track progress if enabled
        progress_context = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) if self.use_progress else None
        
        with progress_context as progress:
            task = progress.add_task(
                "[green]Generating conversations...", 
                total=self.target_conversations
            ) if self.use_progress else None
            
            # Define distribution of complexity and category focus
            complexity_distribution = {
                "basic": 0.3,
                "intermediate": 0.5,
                "advanced": 0.2
            }
            
            # Extract available categories from QA pairs and ensure we have more than one
            categories = list(set([qa.category for qa in qa_pairs]))
            if not categories:
                categories = ["general"]
            
            # Use equal distribution for available categories
            category_distribution = {cat: 1/len(categories) for cat in categories}
            
            # Counter to avoid infinite loops
            attempts = 0
            max_attempts = self.target_conversations * 5  # Reasonable limit
            
            # Generate conversations with varying complexity and focus
            while len(conversations) < self.target_conversations and attempts < max_attempts:
                attempts += 1
                
                # For 40% of conversations, don't filter by complexity to increase success rate
                if random.random() < 0.4:
                    complexity = None
                else:
                    complexity = random.choices(
                        list(complexity_distribution.keys()),
                        weights=list(complexity_distribution.values()),
                        k=1
                    )[0]
                
                # For 60% of conversations, don't filter by category to increase success rate
                if random.random() < 0.6 or not categories or len(categories) == 1:
                    category = None
                else:
                    # Safely handle the category selection
                    try:
                        category = random.choice(categories)  # Use simpler random.choice instead of choices
                    except (IndexError, ValueError):
                        category = None
                
                # Create the conversation
                conversation = self.create_structured_conversation(
                    qa_pairs=qa_pairs,
                    complexity_preference=complexity,
                    category_focus=category
                )
                
                # Add valid conversation to the list
                if conversation:
                    # Check that the conversation doesn't end with a user message
                    # (Which would indicate a question without an answer)
                    messages = conversation["messages"]
                    if messages and messages[-1]["role"] == "user":
                        # Remove the last message
                        messages.pop()
                        # Update the metadata to reflect the actual number of turns
                        conversation["metadata"]["turns"] = len(messages) // 2
                    
                    # Make sure currency is consistent between questions and answers
                    conversation = self.validate_currency_consistency(conversation)
                    
                    conversations.append(conversation)
                    
                    # Update progress
                    if self.use_progress:
                        progress.update(task, completed=len(conversations))
                
                # Break if all QA pairs have been used too many times
                if all(self.qa_usage_counter.get(i, 0) >= self.max_reuses for i in range(len(qa_pairs))):
                    logger.warning("All QA pairs have reached maximum reuse limit")
                    break
        
        logger.info(f"Generated {len(conversations)} conversations in {attempts} attempts")
        return conversations

    def save_conversations(self, conversations: List[Dict]) -> None:
        """Save conversations to JSONL file with metadata."""
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for conversation in conversations:
                f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(conversations)} conversations to {self.output_file}")
        
        # Generate dataset statistics
        total_turns = sum(len(conv["messages"]) // 2 for conv in conversations)
        avg_turns = total_turns / len(conversations) if conversations else 0
        
        categories = {}
        complexities = {}
        has_formula_count = 0
        has_steps_count = 0
        
        for conv in conversations:
            metadata = conv["metadata"]
            # Split the categories string to count individual categories
            for cat in metadata["categories"].split(", "):
                if cat:  # Only count non-empty categories
                    categories[cat] = categories.get(cat, 0) + 1
            
            complexity = metadata["complexity"]
            complexities[complexity] = complexities.get(complexity, 0) + 1
            
            # Parse the string boolean values back to actual booleans for counting
            if metadata["has_formulas"] == "true":
                has_formula_count += 1
                
            if metadata["has_steps"] == "true":
                has_steps_count += 1
        
        # Log statistics
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total conversations: {len(conversations)}")
        logger.info(f"  Average turns per conversation: {avg_turns:.2f}")
        logger.info(f"  Category distribution: {categories}")
        logger.info(f"  Complexity distribution: {complexities}")
        logger.info(f"  Conversations with formulas: {has_formula_count} ({has_formula_count/len(conversations)*100:.1f}%)")
        logger.info(f"  Conversations with steps: {has_steps_count} ({has_steps_count/len(conversations)*100:.1f}%)")
        
        # Add currency distribution statistics if enabled
        if self.diversify_currency:
            currency_stats = {self.currency_options[idx]["name"]: count 
                             for idx, count in self.currency_distribution.items()}
            logger.info(f"Currency distribution in dataset: {currency_stats}")

    def run(self) -> None:
        """Execute the full extraction and conversation creation process."""
        logger.info(f"Extracting QA pairs from {self.input_file}")
        qa_pairs = self.extract_qa_pairs()
        
        if not qa_pairs:
            logger.error("No QA pairs extracted. Exiting.")
            return
        
        logger.info(f"Generating conversations from {len(qa_pairs)} QA pairs")
        conversations = self.generate_conversations(qa_pairs)
        
        if not conversations:
            logger.error("No conversations generated. Exiting.")
            return
        
        logger.info(f"Saving {len(conversations)} conversations")
        self.save_conversations(conversations)
        
        logger.info("Process completed successfully")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract and create advanced multi-turn finance conversations"
    )
    
    # Add command line arguments
    parser.add_argument(
        "--input",
        type=str,
        default="/home/zahemen/datasets/advanced_finance_questions.txt",
        help="Input file containing advanced finance QA pairs"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="/home/zahemen/datasets/sft_datasets/advanced_finance_conversations.jsonl",
        help="Output JSONL file path"
    )
    
    parser.add_argument(
        "--min-turns",
        type=int,
        default=3,
        help="Minimum number of turns in a conversation"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum number of turns in a conversation"
    )
    
    parser.add_argument(
        "--max-reuses",
        type=int,
        default=15,
        help="Maximum times a QA pair can be reused"
    )
    
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=1000,
        help="Target number of conversations to generate"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    parser.add_argument(
        "--no-currency-diversity",
        action="store_true",
        help="Disable currency diversification in generated conversations"
    )
    
    parser.add_argument(
        "--no-error-introduction",
        action="store_true",
        help="Disable introduction of random errors in questions"
    )
    
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.3,
        help="Percentage of questions to introduce errors to (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    extractor = AdvancedFinanceConversationExtractor(
        input_file=args.input,
        output_file=args.output,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_reuses=args.max_reuses,
        target_conversations=args.num_conversations,
        use_progress=not args.no_progress,
        diversify_currency=not args.no_currency_diversity,
        introduce_errors=not args.no_error_introduction,
        error_rate=args.error_rate
    )
    
    extractor.run()

if __name__ == "__main__":
    main()
