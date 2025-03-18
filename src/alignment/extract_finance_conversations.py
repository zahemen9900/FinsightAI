import re
import json
import logging
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from create_intro_dataset import IntroDatasetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")

class FinanceConversationExtractor:
    """Extract finance conversations from text files in the specified format."""
    
    def __init__(
        self,
        input_file: str = "/home/zahemen/datasets/enhanced_q_and_a/finance_questions.txt",
        output_file: str = "/home/zahemen/datasets/sft_datasets/finance_conversations.jsonl",
        system_prompt: str = "You are FinSight, an AI financial advisor. Provide accurate and helpful financial guidance.",
        min_turns: int = 3,
        max_turns: int = 10,
        max_reuses: int = 20,  # Increased from 10 to 20
        target_conversations: int = 1000,  # Added target number of conversations
        use_progress: bool = True,  # Add parameter to control progress bars
        diversify_currency: bool = True  # Add currency diversification parameter
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.conversations = []
        self.system_prompt = system_prompt
        self.conversation_counter = 0
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.max_reuses = max_reuses
        self.target_conversations = target_conversations
        self.qa_usage_counter = {}  # Track how many times each QA pair is used
        self.intro_generator = IntroDatasetGenerator(None)  # Initialize intro generator
        self.use_progress = use_progress  # Whether to use progress bars
        
        # System prompt variations
        self.system_prompt_variations = [
            "You are FinSight, an AI financial advisor. Provide accurate and helpful financial guidance.",
            "As FinSight, your role is to deliver expert financial insights and advice tailored to each user's needs.",
            "You are FinSight, a specialized AI designed to help users understand complex financial concepts and make informed decisions.",
            "Acting as FinSight, provide thoughtful financial guidance and explanations that help users navigate their financial questions.",
            "You are FinSight, an AI assistant specialized in financial education and advice. Provide clear and accurate information."
        ]
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Verify input file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        # Add currency diversification settings
        self.diversify_currency = diversify_currency
        self.currency_options = [
            {"name": "dollars", "symbol": "$", "regex": r'(\$|\bdollars?\b)'},
            {"name": "pounds", "symbol": "£", "regex": r'(£|\bpounds?\b)'},
            {"name": "euros", "symbol": "€", "regex": r'(€|\beuros?\b)'},
            {"name": "cedis", "symbol": "₵", "regex": r'(₵|\bcedis\b)'},
        ]
        self.currency_distribution = {i: 0 for i, _ in enumerate(self.currency_options)}

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
    
    def generate_conversation_id(self) -> str:
        """Generate a unique identifier for a conversation."""
        unique_string = f"finance_conv_{self.conversation_counter}_{random.getrandbits(32)}"
        conv_id = hashlib.sha256(unique_string.encode()).hexdigest()
        self.conversation_counter += 1
        return conv_id
    
    def extract_qa_pairs(self) -> List[Dict[str, str]]:
        """Extract QA pairs from the input file."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split the content by "User:" to find each conversation start
            parts = content.split('\nUser: ')
            
            # If the file starts with "User:", the first part will be empty
            if parts[0].strip() == '':
                parts = parts[1:]  # Remove the empty first part
            else:
                # If the file doesn't start with "User:", fix the first part
                parts[0] = parts[0].lstrip()
            
            # Now prepend "User: " to all parts except potentially the first one
            parts = ["User: " + part if i > 0 or parts[0].startswith("User: ") else part 
                    for i, part in enumerate(parts)]
            
            logger.info(f"Found {len(parts)} potential QA pairs")
            
            # Process each part to extract User/Assistant pairs
            qa_pairs = []
            skipped = 0
            
            # Use tqdm conditionally based on use_progress flag
            parts_iter = tqdm(parts, desc="Extracting QA pairs") if self.use_progress else parts
            
            for part in parts_iter:
                # Use regex to extract the user question and assistant response
                match = re.match(r'User:\s*(.*?)\s*\nAssistant:\s*(.*?)(?=\n\s*User:|$)', 
                                part, re.DOTALL)
                
                if match:
                    user_question = match.group(1).strip()
                    assistant_response = match.group(2).strip()
                    
                    # Skip if either part is empty
                    if not user_question or not assistant_response:
                        skipped += 1
                        continue
                    
                    # Skip placeholders like [leave-blank]
                    if "[leave-blank]" in assistant_response:
                        skipped += 1
                        continue
                    
                    # Determine if this is a company-specific question
                    company_names = self.extract_company_names(user_question)
                    
                    # Add additional metadata to help create varied conversations
                    topics = self.detect_topics(user_question)
                    
                    qa_pairs.append({
                        "question": user_question,
                        "answer": assistant_response,
                        "companies": company_names,
                        "topics": topics
                    })
                else:
                    skipped += 1
            
            logger.info(f"Extracted {len(qa_pairs)} QA pairs, skipped {skipped} invalid parts")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error extracting QA pairs: {e}")
            return []
    
    def extract_company_names(self, text: str) -> List[str]:
        """Extract potential company names from the question."""
        # List of major companies to look for
        major_companies = [
            "Apple", "Microsoft", "Amazon", "Alphabet", "Google", "Meta", "Facebook", 
            "Tesla", "Nvidia", "AMD", "Intel", "Cisco", "Oracle", "IBM", "Dell", "HP", 
            # ... existing companies ...
        ]

        found_companies = []
        for company in major_companies:
            # Search for company name as a whole word
            pattern = r'\b' + re.escape(company) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found_companies.append(company)
        
        return found_companies
    
    def detect_topics(self, text: str) -> List[str]:
        """Detect financial topics in the question for better organization."""
        text_lower = text.lower()
        
        topic_keywords = {
            "investment": ["invest", "stock", "portfolio", "market", "share", "return", "fund", "etf", "dividend"],
            "retirement": ["retire", "401k", "ira", "pension", "social security", "savings", "retirement"],
            "taxation": ["tax", "irs", "deduction", "write-off", "credit", "liability", "withholding"],
            "budgeting": ["budget", "saving", "spending", "income", "expense", "debt", "credit", "money"],
            "crypto": ["crypto", "bitcoin", "ethereum", "blockchain", "token", "nft", "cryptocurrency"],
            "real_estate": ["real estate", "house", "mortgage", "property", "rent", "lease", "housing"],
            "banking": ["bank", "account", "interest", "deposit", "withdraw", "checking", "saving"],
            "insurance": ["insurance", "policy", "premium", "coverage", "risk", "claim", "life insurance"],
            "macroeconomics": ["inflation", "interest rate", "fed", "recession", "gdp", "economy"],
            "personal_finance": ["personal finance", "financial plan", "emergency fund", "savings"]
        }
        
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics if detected_topics else ["general"]
    
    def can_use_qa_pair(self, qa_index: int) -> bool:
        """Check if a QA pair can still be used based on usage count."""
        return self.qa_usage_counter.get(qa_index, 0) < self.max_reuses
    
    def mark_qa_pair_used(self, qa_index: int) -> None:
        """Mark a QA pair as used by incrementing its usage counter."""
        self.qa_usage_counter[qa_index] = self.qa_usage_counter.get(qa_index, 0) + 1
    
    def group_by_topic(self, qa_pairs: List[Dict[str, str]]) -> Dict[str, List[int]]:
        """Group QA pairs by topics for more coherent conversations."""
        topic_indices = {}
        
        # First, build a mapping of topics to indices
        for i, qa in enumerate(qa_pairs):
            # For company-specific questions, add to a company topic
            if qa["companies"]:
                if "company_specific" not in topic_indices:
                    topic_indices["company_specific"] = []
                topic_indices["company_specific"].append(i)
            
            # Add to each detected topic
            for topic in qa["topics"]:
                if topic not in topic_indices:
                    topic_indices[topic] = []
                topic_indices[topic].append(i)
        
        # Ensure there's a general category for fallback
        if "general" not in topic_indices:
            topic_indices["general"] = list(range(len(qa_pairs)))
            
        return topic_indices
    
    def create_varied_multi_turn_conversation(self, 
                                             qa_pairs: List[Dict[str, str]], 
                                             topic_indices: Dict[str, List[int]],
                                             variation_type: str = "random") -> Dict:
        """Create a multi-turn conversation from QA pairs with more variation."""
        # Decide whether to use intro generator conversation
        use_intro = random.random() < 0.3  # 30% chance to use intro
        
        # Choose conversation variation type if not specified
        if variation_type == "random":
            variation_type = random.choice(["topic_focused", "mixed_topics", "company_specific", "progressive"])
        
        # Get available indices based on variation type
        available_indices = []
        
        if variation_type == "company_specific" and "company_specific" in topic_indices:
            # Company-focused conversation
            company_indices = [idx for idx in topic_indices["company_specific"] if self.can_use_qa_pair(idx)]
            if len(company_indices) >= self.min_turns:
                available_indices = company_indices
                topic = "company_specific"
            else:
                # Fall back to a different variation
                variation_type = "topic_focused"
        
        if variation_type == "topic_focused":
            # Choose a specific topic with enough available questions
            valid_topics = [topic for topic, indices in topic_indices.items() 
                         if len([idx for idx in indices if self.can_use_qa_pair(idx)]) >= self.min_turns
                         and topic != "general"]  # Prefer specific topics over general
            
            if valid_topics:
                topic = random.choice(valid_topics)
                available_indices = [idx for idx in topic_indices[topic] if self.can_use_qa_pair(idx)]
            else:
                # Fall back to mixed approach
                variation_type = "mixed_topics"
        
        if variation_type == "mixed_topics" or variation_type == "progressive" or not available_indices:
            # Mix questions from different topics or ensure we have enough
            all_available = [i for i, _ in enumerate(qa_pairs) if self.can_use_qa_pair(i)]
            if len(all_available) < self.min_turns:
                # Not enough questions available
                logger.warning("Not enough available QA pairs to create a conversation")
                return None
            
            if variation_type == "mixed_topics":
                # Randomly select from all available questions
                available_indices = all_available
                topic = "mixed"
            elif variation_type == "progressive":
                # Try to create a progression through related topics
                # Start with one topic and expand to related ones
                # For simplicity, we'll use the order they appear in the file as a proxy
                seed_topic = random.choice(list(topic_indices.keys()))
                available_indices = [idx for idx in topic_indices[seed_topic] if self.can_use_qa_pair(idx)]
                
                # If we need more questions, add from other topics
                if len(available_indices) < self.min_turns:
                    extra_needed = self.min_turns - len(available_indices)
                    # Add questions from other topics
                    other_indices = [i for i in all_available if i not in available_indices]
                    if other_indices:
                        available_indices.extend(random.sample(other_indices, 
                                                          min(extra_needed, len(other_indices))))
                
                topic = "progressive"
        
        # Determine number of turns (use a distribution that favors middle values)
        if self.min_turns == self.max_turns:
            num_turns = self.min_turns
        else:
            # Create a distribution that favors middle values
            weights = []
            for i in range(self.min_turns, self.max_turns + 1):
                # Calculate weight - higher for middle values
                mid_point = (self.min_turns + self.max_turns) / 2
                distance = abs(i - mid_point)
                max_distance = (self.max_turns - self.min_turns) / 2
                weight = 1 - (distance / max_distance) * 0.5  # Higher weight for middle values
                weights.append(weight)
            
            # Normalize weights
            total = sum(weights)
            weights = [w/total for w in weights]
            
            # Choose the number of turns with weights
            num_turns = random.choices(
                range(self.min_turns, self.max_turns + 1),
                weights=weights,
                k=1
            )[0]
        
        num_turns = min(num_turns, len(available_indices))
        
        # Select QA pairs for this conversation
        if variation_type == "progressive":
            # Select with some sequential preference
            # Start with easier/intro questions and move to more complex ones
            # For simplicity, we'll use the order they appear in the file as a proxy
            selected_indices = sorted(random.sample(available_indices, num_turns))
        else:
            # Random selection
            selected_indices = random.sample(available_indices, num_turns)
        
        # Start with system message
        if use_intro:
            # Get intro conversation but keep only system message and first 2-4 messages
            intro_messages = self.intro_generator.generate_conversation()
            system_message = intro_messages[0]  # Keep system message
            
            # Determine how many intro turns to keep (1-2 turns)
            intro_turns = random.randint(1, 2)
            kept_messages = intro_messages[1:intro_turns*2+1]  # +1 because we start from index 1
            
            messages = [system_message] + kept_messages
        else:
            # Use a random system prompt variation
            messages = [{"role": "system", "content": random.choice(self.system_prompt_variations)}]
        
        # Add selected QA pairs
        for idx in selected_indices:
            qa = qa_pairs[idx]
            messages.extend([
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]}
            ])
            self.mark_qa_pair_used(idx)
        
        # Generate conversation ID and metadata
        conversation_id = self.generate_conversation_id()
        
        # Collect all company names mentioned in the conversation
        all_companies = []
        all_topics = set()
        for idx in selected_indices:
            all_companies.extend(qa_pairs[idx]["companies"])
            all_topics.update(qa_pairs[idx]["topics"])
        
        # Convert lists to comma-separated strings for metadata
        unique_companies_str = ", ".join(set(all_companies)) if all_companies else ""
        all_topics_str = ", ".join(all_topics) if all_topics else ""
        
        conversation = {
            "messages": messages,
            "metadata": {
                "source": "finance_questions",
                "conversation_id": conversation_id,
                "type": "finance_qa_multi_turn",
                "topic": topic,
                "subtopics": all_topics_str,  # Convert to string
                "variation": variation_type,
                "turns": len(messages) // 2,  # Count actual turns (user+assistant pairs)
                "companies": unique_companies_str,  # Convert to string
                "has_intro": str(use_intro).lower()  # Convert boolean to string
            }
        }
        
        # Add currency diversification
        if self.diversify_currency:
            # First check if this is a company-specific conversation
            # If it contains any company references, don't convert currency
            contains_company_refs = False
            
            # Check messages and metadata for company references
            if "company_specific" in conversation["metadata"]["variation"] or conversation["metadata"]["companies"]:
                contains_company_refs = True
                
            if not contains_company_refs:
                for msg in conversation["messages"]:
                    if msg["role"] in ["user", "assistant"]:
                        if any(company.lower() in msg["content"].lower() for company in self.company_names):
                            contains_company_refs = True
                            break
            
            # Skip currency conversion for company-related conversations
            if contains_company_refs:
                return conversation
                
            # Continue with regular currency diversification if not company-related
            # Select least used currency to maintain even distribution
            currency_idx = min(self.currency_distribution, key=self.currency_distribution.get)
            currency = self.currency_options[currency_idx]
            self.currency_distribution[currency_idx] += 1
            
            # Decide whether to use symbol or name in questions (alternate)
            use_symbol = random.choice([True, False])
            
            # Update messages with converted currency
            for i, msg in enumerate(conversation["messages"]):
                if msg["role"] == "user":
                    # Apply currency conversion to user messages
                    conversation["messages"][i]["content"] = self._convert_currency(
                        msg["content"],
                        r'(\$|\bdollars?\b)',
                        currency["symbol"],
                        currency["name"],
                        use_symbol
                    )
                    # Fix any broken dollar-cost terms
                    conversation["messages"][i]["content"] = self._fix_dollar_cost_terms(conversation["messages"][i]["content"])
                elif msg["role"] == "assistant":
                    # Apply currency conversion to assistant messages
                    conversation["messages"][i]["content"] = self._convert_currency(
                        msg["content"],
                        r'(\$|\bdollars?\b)',
                        currency["symbol"],
                        currency["name"],
                        False  # Always use symbols in answers
                    )
                    # Fix any broken dollar-cost terms
                    conversation["messages"][i]["content"] = self._fix_dollar_cost_terms(conversation["messages"][i]["content"])
            
            # Add currency information to metadata
            if "metadata" not in conversation:
                conversation["metadata"] = {}
            conversation["metadata"]["currency"] = currency["name"]
        
        return conversation
    
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
        # Note: Enhanced check to be more thorough
        if any(company.lower() in text.lower() for company in self.company_names):
            return text
        
        # Check for special case "dollar-cost" which shouldn't be replaced
        # Preserve "dollar-cost" term before any replacements
        text = text.replace("dollar-cost", "DOLLARCOSTPLACEHOLDER")
        text = text.replace("Dollar-cost", "DOLLARCOSTPLACEHOLDER_CAP")
        text = text.replace("DOLLAR-COST", "DOLLARCOSTPLACEHOLDER_ALLCAP")
        
        # Check if this is a question (heuristic)
        is_question = text.strip().endswith('?') or len(text.split()) < 30
        
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
            # For answers, always use symbol
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
    
    def generate_conversations_with_strategy(self, qa_pairs: List[Dict[str, str]]) -> List[Dict]:
        """Generate conversations with a strategic approach to maximize variety."""
        if not qa_pairs:
            logger.error("No QA pairs available to create conversations")
            return []
            
        # Group QA pairs by topic for better organization
        topic_indices = self.group_by_topic(qa_pairs)
        
        # Calculate how many conversations we need of each type
        total_target = self.target_conversations
        
        # Define variation types and their proportions
        variation_distribution = {
            "topic_focused": 0.4,  # 40% topic-focused
            "mixed_topics": 0.3,   # 30% mixed topics
            "company_specific": 0.15, # 15% company-specific
            "progressive": 0.15    # 15% progressive
        }
        
        conversation_targets = {
            var_type: int(total_target * proportion)
            for var_type, proportion in variation_distribution.items()
        }
        
        # Ensure we add up to the total
        remaining = total_target - sum(conversation_targets.values())
        if remaining > 0:
            # Add the remainder to topic_focused
            conversation_targets["topic_focused"] += remaining
        
        logger.info(f"Conversation targets by type: {conversation_targets}")
        
        conversations = []
        variation_counts = {var_type: 0 for var_type in variation_distribution}
        
        # Track used company combinations to avoid duplicates
        used_company_combinations = set()
        
        # Counter to prevent infinite loops
        attempts = 0
        max_attempts = total_target * 2
        
        # Create conversations until we reach the target or max attempts
        # Use tqdm conditionally based on use_progress flag
        loop_range = tqdm(range(attempts, max_attempts), 
                         desc="Creating conversations", 
                         initial=0, 
                         total=total_target) if self.use_progress else range(attempts, max_attempts)
        
        while sum(variation_counts.values()) < total_target and attempts < max_attempts:
            attempts += 1
            if self.use_progress and attempts % 10 == 0:  # Update progress bar every 10 attempts
                loop_range.update(10)
                loop_range.set_description(f"Created {sum(variation_counts.values())}/{total_target}")
            
            # Choose variation type prioritizing those that need more conversations
            remaining_by_type = {
                var_type: conversation_targets[var_type] - variation_counts[var_type]
                for var_type in variation_distribution
                if conversation_targets[var_type] > variation_counts[var_type]
            }
            
            if not remaining_by_type:
                # All targets met, choose randomly
                variation_type = random.choice(list(variation_distribution.keys()))
            else:
                # Weight by how many more we need of each type
                weights = list(remaining_by_type.values())
                types = list(remaining_by_type.keys())
                variation_type = random.choices(types, weights=weights, k=1)[0]
            
            # Create conversation with selected variation type
            conversation = self.create_varied_multi_turn_conversation(
                qa_pairs, topic_indices, variation_type
            )
            
            if conversation:
                # For company-specific conversations, check for duplicates
                if variation_type == "company_specific" and conversation["metadata"]["companies"]:
                    # Sort companies to create consistent key
                    company_key = tuple(sorted(conversation["metadata"]["companies"]))
                    
                    # Skip if we've already used this exact company combination too many times
                    if company_key in used_company_combinations and random.random() < 0.7:
                        continue
                    
                    used_company_combinations.add(company_key)
                
                conversations.append(conversation)
                variation_counts[variation_type] += 1
                
                # Log progress periodically without progress bar
                total_created = sum(variation_counts.values())
                if total_created % 50 == 0 and not self.use_progress:
                    logger.info(f"Created {total_created}/{total_target} conversations")
            
            # Check if we've used up all QA pairs beyond their reuse limit
            if not any(self.can_use_qa_pair(i) for i in range(len(qa_pairs))):
                # Reset usage counters if we haven't met our targets
                if total_created < total_target * 0.95:  # If we're below 95% of target
                    logger.warning("Resetting QA pair usage counters to meet conversation targets")
                    self.qa_usage_counter = {}
                else:
                    logger.warning("All QA pairs have reached maximum usage")
                    break
        
        # Close progress bar if using it
        if self.use_progress:
            loop_range.close()
            
        # Log final distribution
        logger.info(f"Conversations created by variation type: {variation_counts}")
        logger.info(f"Total conversations created: {sum(variation_counts.values())}/{total_target}")
        
        return conversations
    
    def save_conversations(self) -> None:
        """Save created conversations to a JSONL file."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for conversation in self.conversations:
                    f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
            
            logger.info(f"Successfully saved {len(self.conversations)} conversations to {self.output_file}")
            
            # Provide stats on conversation characteristics
            company_specific = sum(1 for conv in self.conversations if conv["metadata"]["companies"])
            logger.info(f"Company-specific conversations: {company_specific} ({company_specific/len(self.conversations)*100:.1f}%)")
            
            with_intro = sum(1 for conv in self.conversations if conv["metadata"]["has_intro"])
            logger.info(f"Conversations with intro: {with_intro} ({with_intro/len(self.conversations)*100:.1f}%)")
            
            # Stats on conversation length
            turn_counts = [conv["metadata"]["turns"] for conv in self.conversations]
            avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
            logger.info(f"Average conversation turns: {avg_turns:.1f}")
            logger.info(f"Min turns: {min(turn_counts) if turn_counts else 0}, Max turns: {max(turn_counts) if turn_counts else 0}")
            
            # Stats on topics
            topic_counts = {}
            for conv in self.conversations:
                topic = conv["metadata"]["topic"]
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            logger.info("Topic distribution:")
            for topic, count in topic_counts.items():
                logger.info(f"  {topic}: {count} ({count/len(self.conversations)*100:.1f}%)")
            
            # Add currency distribution statistics if enabled
            if self.diversify_currency:
                currency_stats = {self.currency_options[idx]["name"]: count 
                                 for idx, count in self.currency_distribution.items()}
                logger.info(f"Currency distribution in dataset: {currency_stats}")
            
        except Exception as e:
            logger.error(f"Error saving conversations: {e}")
    
    def run(self, num_conversations: int = None) -> None:
        """Execute the full extraction and conversation creation process."""
        if num_conversations is not None:
            self.target_conversations = num_conversations
            
        logger.info(f"Starting finance conversation extraction from {self.input_file}")
        logger.info(f"Target number of conversations: {self.target_conversations}")
        
        # Only use Progress if use_progress is True
        if self.use_progress:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            ) as progress:
                extract_task = progress.add_task("[green]Extracting QA pairs...", total=1)
                create_task = progress.add_task("[cyan]Creating conversations...", total=1, visible=False)
                save_task = progress.add_task("[magenta]Saving conversations...", total=1, visible=False)
                
                # Extract QA pairs
                qa_pairs = self.extract_qa_pairs()
                progress.update(extract_task, completed=1)
                
                if qa_pairs:
                    progress.update(create_task, visible=True)
                    # Create multi-turn conversations with strategic approach
                    self.conversations = self.generate_conversations_with_strategy(qa_pairs)
                    progress.update(create_task, completed=1)
                    
                    if self.conversations:
                        progress.update(save_task, visible=True)
                        # Save conversations
                        self.save_conversations()
                        progress.update(save_task, completed=1)
        else:
            # Simple execution without progress bars
            logger.info("Extracting QA pairs...")
            qa_pairs = self.extract_qa_pairs()
            
            if qa_pairs:
                logger.info("Creating conversations...")
                self.conversations = self.generate_conversations_with_strategy(qa_pairs)
                
                if self.conversations:
                    logger.info("Saving conversations...")
                    self.save_conversations()
        
        # Print sample conversation regardless of progress bar usage
        if self.conversations:
            sample = random.choice(self.conversations)
            logger.info("\nSample conversation:")
            for msg in sample["messages"]:
                if msg["role"] == "system":
                    logger.info(f"System: {msg['content']}")
                elif msg["role"] == "user":
                    logger.info(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    logger.info(f"Assistant: {msg['content'][:100]}...")  # Truncate long responses
            
            logger.info(f"Metadata: {sample['metadata']}")
        else:
            logger.warning("No conversations were created")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and create multi-turn finance conversations from text file")
    
    parser.add_argument(
        "--input", 
        type=str, 
        default="/home/zahemen/datasets/enhanced_q_and_a/finance_questions.txt",
        help="Input file containing finance conversations"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="/home/zahemen/datasets/sft_datasets/finance_conversations.jsonl",
        help="Output JSONL file path"
    )
    
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=2000,  # Changed default to 2000
        help="Number of conversations to generate"
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
        default=10,
        help="Maximum number of turns in a conversation"
    )
    
    parser.add_argument(
        "--max-reuses",
        type=int,
        default=20,  # Increased default from 5 to 20
        help="Maximum times a QA pair can be reused"
    )
    
    parser.add_argument(
        "--use-progress",
        action="store_true",
        help="Use progress bars (default: True)"
    )
    
    # Add currency diversification argument
    parser.add_argument(
        "--no-currency-diversity",
        action="store_true",
        help="Disable currency diversification in generated conversations"
    )
    
    args = parser.parse_args()
    
    extractor = FinanceConversationExtractor(
        input_file=args.input,
        output_file=args.output,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_reuses=args.max_reuses,
        target_conversations=args.num_conversations,
        use_progress=args.use_progress,
        diversify_currency=not args.no_currency_diversity
    )
    
    extractor.run()

if __name__ == "__main__":
    main()
