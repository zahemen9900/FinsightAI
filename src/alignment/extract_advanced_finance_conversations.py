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
        use_progress: bool = True
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

    def extract_qa_pairs(self) -> List[AdvancedQAPair]:
        """Extract Q&A pairs while preserving formatting and structure."""
        qa_pairs = []
        current_question = None
        current_answer = []
        question_pattern = re.compile(r"Question \d+:(.*?)(?=Answer:)", re.DOTALL)
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split by delimiter (the line of dashes)
            sections = content.split("-" * 80)
            
            for section in tqdm(sections, desc="Extracting QA pairs"):
                if not section.strip():
                    continue
                    
                # Extract question
                question_match = question_pattern.search(section)
                if not question_match:
                    continue
                    
                question = question_match.group(1).strip()
                
                # Extract answer
                answer_start = section.find("Answer:")
                if answer_start == -1:
                    continue
                    
                answer = section[answer_start + 7:].strip()
                
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
            return []
        
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

    def create_structured_conversation(self, 
                                      qa_pairs: List[AdvancedQAPair],
                                      complexity_preference: str = None,
                                      category_focus: str = None) -> Dict:
        """Create a structured conversation with controlled complexity and focus."""
        # Filter QA pairs based on preferences
        filtered_pairs = qa_pairs
        
        if complexity_preference:
            filtered_pairs = [qa for qa in filtered_pairs if qa.complexity == complexity_preference]
        
        if category_focus:
            filtered_pairs = [qa for qa in filtered_pairs if qa.category == category_focus 
                             or category_focus in qa.topics]
            
        # If filters eliminated too many pairs, fall back to all pairs
        if len(filtered_pairs) < self.min_turns:
            filtered_pairs = qa_pairs
            
        # Select a random starting QA pair
        available_indices = [i for i, qa in enumerate(filtered_pairs) 
                            if self.qa_usage_counter.get(i, 0) < self.max_reuses]
        
        if not available_indices:
            logger.warning("No available QA pairs for new conversation")
            return None
        
        # Determine number of turns for this conversation
        num_turns = random.randint(self.min_turns, min(self.max_turns, len(available_indices)))
        
        # Choose system prompt
        system_prompt = random.choice(self.system_prompt_variations)
        
        # Initialize conversation with starting QA pair
        start_idx = random.choice(available_indices)
        self.mark_qa_pair_used(start_idx)
        start_qa = filtered_pairs[start_idx]
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": start_qa.question},
            {"role": "assistant", "content": start_qa.answer}
        ]
        
        # Track what pairs have been used in this conversation to avoid repetition
        used_in_conv = {start_idx}
        
        # Add additional turns using follow-up questions
        for turn in range(num_turns - 1):
            # Generate a follow-up based on the last QA pair
            follow_up = self._generate_follow_up(start_qa)
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
        conversation = {
            "id": self.generate_conversation_id(),
            "messages": messages,
            "metadata": {
                "source": "advanced_finance_qa",
                "turns": len(messages) // 2,  # Number of turns
                "categories": list(set([qa.category for qa in [start_qa] + 
                                      [filtered_pairs[i] for i in used_in_conv if i != start_idx]])),
                "complexity": start_qa.complexity,
                "topics": list(set(sum([qa.topics for qa in [start_qa] + 
                                      [filtered_pairs[i] for i in used_in_conv if i != start_idx]], []))),
                "has_formulas": any(qa.has_formula for qa in [start_qa] + 
                                  [filtered_pairs[i] for i in used_in_conv if i != start_idx]),
                "has_steps": any(qa.has_steps for qa in [start_qa] + 
                                [filtered_pairs[i] for i in used_in_conv if i != start_idx])
            }
        }
        
        return conversation

    def mark_qa_pair_used(self, qa_index: int) -> None:
        """Mark a QA pair as used by incrementing its usage counter."""
        self.qa_usage_counter[qa_index] = self.qa_usage_counter.get(qa_index, 0) + 1

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
            
            # Extract available categories from QA pairs
            categories = list(set([qa.category for qa in qa_pairs]))
            category_distribution = {cat: 1/len(categories) for cat in categories}
            
            # Generate conversations with varying complexity and focus
            while len(conversations) < self.target_conversations:
                # For 60% of conversations, choose specific complexity
                if random.random() < 0.6:
                    complexity = random.choices(
                        list(complexity_distribution.keys()),
                        weights=list(complexity_distribution.values()),
                        k=1
                    )[0]
                else:
                    complexity = None
                
                # For 50% of conversations, focus on a specific category
                if random.random() < 0.5:
                    category = random.choices(
                        list(category_distribution.keys()),
                        weights=list(category_distribution.values()),
                        k=1
                    )[0]
                else:
                    category = None
                
                # Create the conversation
                conversation = self.create_structured_conversation(
                    qa_pairs=qa_pairs,
                    complexity_preference=complexity,
                    category_focus=category
                )
                
                # Add valid conversation to the list
                if conversation:
                    conversations.append(conversation)
                    
                    # Update progress
                    if self.use_progress:
                        progress.update(task, completed=len(conversations))
                
                # Break if all QA pairs have been used too many times
                if all(self.qa_usage_counter.get(i, 0) >= self.max_reuses for i in range(len(qa_pairs))):
                    logger.warning("All QA pairs have reached maximum reuse limit")
                    break
        
        logger.info(f"Generated {len(conversations)} conversations")
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
            for cat in metadata["categories"]:
                categories[cat] = categories.get(cat, 0) + 1
            
            complexity = metadata["complexity"]
            complexities[complexity] = complexities.get(complexity, 0) + 1
            
            if metadata["has_formulas"]:
                has_formula_count += 1
                
            if metadata["has_steps"]:
                has_steps_count += 1
        
        # Log statistics
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total conversations: {len(conversations)}")
        logger.info(f"  Average turns per conversation: {avg_turns:.2f}")
        logger.info(f"  Category distribution: {categories}")
        logger.info(f"  Complexity distribution: {complexities}")
        logger.info(f"  Conversations with formulas: {has_formula_count} ({has_formula_count/len(conversations)*100:.1f}%)")
        logger.info(f"  Conversations with steps: {has_steps_count} ({has_steps_count/len(conversations)*100:.1f}%)")

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
    
    args = parser.parse_args()
    
    extractor = AdvancedFinanceConversationExtractor(
        input_file=args.input,
        output_file=args.output,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_reuses=args.max_reuses,
        target_conversations=args.num_conversations,
        use_progress=not args.no_progress
    )
    
    extractor.run()

if __name__ == "__main__":
    main()
