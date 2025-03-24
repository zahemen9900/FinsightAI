#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import tqdm
import langdetect
from collections import defaultdict
import string
import multiprocessing
from functools import partial
import time

# Improve performance by setting parallel execution for langdetect
langdetect.DetectorFactory.seed = 0

class FinanceInstructDatasetProcessor:
    def __init__(
        self,
        input_file: str,
        output_dir: str,
        dataset_percentage: float = 1.0,
        max_sample_usage: int = 2,
        min_turns: int = 1,
        max_turns: int = 5,
        coherent_percentage: float = 0.5,
        num_workers: int = None,
        batch_size: int = 10000,
        min_response_length: int = 150,  # Minimum character count for assistant responses
        min_response_words: int = 30,    # Minimum word count for assistant responses
        seed: int = 42
    ):
        self.input_file = input_file
        self.output_dir = output_dir
        self.dataset_percentage = dataset_percentage
        self.max_sample_usage = max_sample_usage
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.coherent_percentage = coherent_percentage
        self.batch_size = batch_size
        self.min_response_length = min_response_length
        self.min_response_words = min_response_words
        self.seed = seed
        
        # Set the number of workers (default to CPU count - 1)
        self.num_workers = num_workers if num_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        
        # Set the seed for reproducibility
        random.seed(self.seed)
        
        # Compile regex pattern for faster Chinese detection
        self.chinese_regex = re.compile(r'[\u4e00-\u9fff]')
        
        # System prompts for FinSight
        self.system_prompts = [
            "You are FinSight, a specialized AI designed to help users understand complex financial concepts and make informed decisions.",
            "You are FinSight, an AI financial advisor dedicated to helping users navigate financial markets and improve their financial literacy.",
            "You are FinSight, an AI assistant specialized in providing clear, accurate information about financial markets, investment strategies, and economic concepts.",
            "You are FinSight, a financial AI assistant that helps users with investment decisions, financial planning, and understanding market trends.",
            "You are FinSight, an AI designed to simplify complex financial concepts and provide personalized financial guidance to users.",
            "You are FinSight, a financial AI advisor that provides data-driven insights and recommendations to help users make informed financial decisions.",
            "You are FinSight, an AI assistant specialized in financial analysis, market predictions, and personalized investment advice.",
            "You are FinSight, a financial AI companion that offers expertise in portfolio management, retirement planning, and wealth preservation.",
            "You are FinSight, an AI financial guide with expertise in stock markets, cryptocurrency, real estate, and various investment vehicles.",
            "You are FinSight, an AI specialized in breaking down complex financial jargon and helping users develop sound financial strategies."
        ]
        
        # Prepare financial keywords for faster topic extraction
        self.financial_keywords = {
            'stock': 'Stocks',
            'bond': 'Bonds',
            'invest': 'Investing',
            'market': 'Markets',
            'portfolio': 'Portfolio',
            'retirement': 'Retirement',
            'tax': 'Taxes',
            'credit': 'Credit',
            'debt': 'Debt',
            'mortgage': 'Mortgages',
            'loan': 'Loans',
            'inflation': 'Inflation',
            'recession': 'Economy',
            'gdp': 'Economy',
            'interest rate': 'Interest Rates',
            'mutual fund': 'Funds',
            'etf': 'ETFs',
            'dividend': 'Dividends',
            'crypto': 'Cryptocurrency',
            'bitcoin': 'Cryptocurrency',
            'blockchain': 'Cryptocurrency',
            'insurance': 'Insurance',
            'budget': 'Budgeting',
            'saving': 'Savings',
            'expense': 'Expenses',
            'income': 'Income',
            'real estate': 'Real Estate',
            'property': 'Real Estate',
            'financial statement': 'Financial Statements',
            'balance sheet': 'Financial Statements',
            'income statement': 'Financial Statements',
            'cash flow': 'Cash Flow',
            'option': 'Options Trading',
            'futures': 'Futures',
            'forex': 'Forex',
            'hedge fund': 'Hedge Funds',
            'private equity': 'Private Equity',
            'venture capital': 'Venture Capital',
            'ira': 'Retirement',
            '401k': 'Retirement',
            'annuity': 'Retirement',
            'pension': 'Retirement',
            'social security': 'Retirement',
            'estate planning': 'Estate Planning',
            'will': 'Estate Planning',
            'trust': 'Estate Planning',
            'inheritance': 'Estate Planning'
        }
        
        # Add broader topic keywords for non-financial topics
        self.topic_keywords = {
            # Technology topics
            'computer': 'Technology',
            'software': 'Technology',
            'hardware': 'Technology',
            'internet': 'Technology',
            'programming': 'Technology',
            'code': 'Technology',
            'algorithm': 'Technology',
            'ai': 'Technology',
            'artificial intelligence': 'Technology',
            'machine learning': 'Technology',
            'neural network': 'Technology',
            'app': 'Technology',
            'website': 'Technology',
            'mobile': 'Technology',
            'cyber': 'Technology',
            'cloud': 'Technology',
            'data': 'Technology',
            'automation': 'Technology',
            
            # Health topics
            'health': 'Health',
            'medical': 'Health',
            'doctor': 'Health',
            'medicine': 'Health',
            'disease': 'Health',
            'symptom': 'Health',
            'treatment': 'Health',
            'diet': 'Health',
            'exercise': 'Health',
            'fitness': 'Health',
            'mental health': 'Health',
            'therapy': 'Health',
            'wellness': 'Health',
            'nutrition': 'Health',
            'immune': 'Health',
            'hospital': 'Health',
            'surgery': 'Health',
            
            # Business (non-financial)
            'business': 'Business',
            'company': 'Business',
            'startup': 'Business',
            'entrepreneur': 'Business',
            'management': 'Business',
            'marketing': 'Business',
            'product': 'Business',
            'service': 'Business',
            'customer': 'Business',
            'client': 'Business',
            'strategy': 'Business',
            'leadership': 'Business',
            'hr': 'Business',
            'hiring': 'Business',
            'career': 'Business',
            'work': 'Business',
            'job': 'Business',
            
            # Education
            'education': 'Education',
            'school': 'Education',
            'university': 'Education',
            'college': 'Education',
            'degree': 'Education',
            'learn': 'Education',
            'study': 'Education',
            'student': 'Education',
            'teacher': 'Education',
            'professor': 'Education',
            'academic': 'Education',
            'research': 'Education',
            'thesis': 'Education',
            'dissertation': 'Education',
            'course': 'Education',
            'classroom': 'Education',
            'exam': 'Education',
            'test': 'Education',
            
            # Science
            'science': 'Science',
            'physics': 'Science',
            'chemistry': 'Science',
            'biology': 'Science',
            'astronomy': 'Science',
            'geology': 'Science',
            'experiment': 'Science',
            'theory': 'Science',
            'hypothesis': 'Science',
            'laboratory': 'Science',
            'research': 'Science',
            'discovery': 'Science',
            'scientific': 'Science',
            'molecule': 'Science',
            'atom': 'Science',
            'cell': 'Science',
            'organism': 'Science',
            
            # Politics & Government
            'politics': 'Politics',
            'government': 'Politics',
            'policy': 'Politics',
            'law': 'Politics',
            'legislation': 'Politics',
            'regulation': 'Politics',
            'election': 'Politics',
            'vote': 'Politics',
            'candidate': 'Politics',
            'president': 'Politics',
            'senator': 'Politics',
            'congress': 'Politics',
            'parliament': 'Politics',
            'democracy': 'Politics',
            'republican': 'Politics',
            'democrat': 'Politics',
            'constitution': 'Politics',
            'supreme court': 'Politics',
            
            # Entertainment
            'movie': 'Entertainment',
            'film': 'Entertainment',
            'tv': 'Entertainment',
            'television': 'Entertainment',
            'show': 'Entertainment',
            'actor': 'Entertainment',
            'actress': 'Entertainment',
            'director': 'Entertainment',
            'music': 'Entertainment',
            'song': 'Entertainment',
            'artist': 'Entertainment',
            'band': 'Entertainment',
            'concert': 'Entertainment',
            'celebrity': 'Entertainment',
            'game': 'Entertainment',
            'play': 'Entertainment',
            'performance': 'Entertainment',
            
            # Sports
            'sport': 'Sports',
            'team': 'Sports',
            'player': 'Sports',
            'football': 'Sports',
            'soccer': 'Sports',
            'basketball': 'Sports',
            'baseball': 'Sports',
            'tennis': 'Sports',
            'golf': 'Sports',
            'hockey': 'Sports',
            'match': 'Sports',
            'tournament': 'Sports',
            'championship': 'Sports',
            'olympic': 'Sports',
            'athlete': 'Sports',
            'coach': 'Sports',
            'stadium': 'Sports',
            
            # Travel
            'travel': 'Travel',
            'vacation': 'Travel',
            'destination': 'Travel',
            'trip': 'Travel',
            'tourism': 'Travel',
            'tourist': 'Travel',
            'hotel': 'Travel',
            'flight': 'Travel',
            'airport': 'Travel',
            'country': 'Travel',
            'city': 'Travel',
            'visit': 'Travel',
            'explore': 'Travel',
            'adventure': 'Travel',
            'sightseeing': 'Travel',
            'passport': 'Travel',
            'visa': 'Travel',
            
            # Food & Cooking
            'food': 'Food',
            'cooking': 'Food',
            'recipe': 'Food',
            'ingredient': 'Food',
            'meal': 'Food',
            'restaurant': 'Food',
            'chef': 'Food',
            'cuisine': 'Food',
            'bake': 'Food',
            'dish': 'Food',
            'taste': 'Food',
            'flavor': 'Food',
            'kitchen': 'Food',
            'eat': 'Food',
            'dinner': 'Food',
            'lunch': 'Food',
            'breakfast': 'Food',
            
            # History
            'history': 'History',
            'historical': 'History',
            'ancient': 'History',
            'medieval': 'History',
            'century': 'History',
            'era': 'History',
            'period': 'History',
            'civilization': 'History',
            'empire': 'History',
            'kingdom': 'History',
            'revolution': 'History',
            'war': 'History',
            'battle': 'History',
            'artifact': 'History',
            'archaeology': 'History',
            'dynasty': 'History',
            'heritage': 'History',
            
            # Philosophy & Ethics
            'philosophy': 'Philosophy',
            'ethics': 'Philosophy',
            'moral': 'Philosophy',
            'ethical': 'Philosophy',
            'existential': 'Philosophy',
            'consciousness': 'Philosophy',
            'meaning': 'Philosophy',
            'purpose': 'Philosophy',
            'virtue': 'Philosophy',
            'justice': 'Philosophy',
            'metaphysics': 'Philosophy',
            'epistemology': 'Philosophy',
            'philosopher': 'Philosophy',
            'socrates': 'Philosophy',
            'aristotle': 'Philosophy',
            'kant': 'Philosophy',
            'nietzsche': 'Philosophy',
            
            # Personal Development
            'personal growth': 'Personal Development',
            'self-improvement': 'Personal Development',
            'mindfulness': 'Personal Development',
            'meditation': 'Personal Development',
            'habit': 'Personal Development',
            'productivity': 'Personal Development',
            'motivation': 'Personal Development',
            'goal': 'Personal Development',
            'success': 'Personal Development',
            'psychology': 'Personal Development',
            'behavior': 'Personal Development',
            'relationship': 'Personal Development',
            'communication': 'Personal Development',
            'emotion': 'Personal Development',
            'stress': 'Personal Development',
            'confidence': 'Personal Development',
            'resilience': 'Personal Development'
        }
        
        # Combine all keywords for topic extraction
        self.all_keywords = {**self.financial_keywords, **self.topic_keywords}
            
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def is_chinese(self, text: str) -> bool:
        """Check if the text contains Chinese characters - optimized version."""
        # Skip language detection and just use regex as it's much faster
        return bool(self.chinese_regex.search(text))

    def filter_chinese_content(self, exchange: Dict[str, Any]) -> bool:
        """Return True if exchange doesn't contain Chinese text."""
        # Fast reject: check if user content has Chinese
        if "user" in exchange and self.is_chinese(exchange["user"]):
            return False
        # Only check assistant content if user content passed
        if "assistant" in exchange and self.is_chinese(exchange["assistant"]):
            return False
        return True

    def get_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        # Generate a random conversation_id - optimization: directly use random bytes
        random_bytes = os.urandom(16)
        return hashlib.sha256(random_bytes).hexdigest()

    def generate_system_prompt(self) -> str:
        """Return a randomly selected system prompt."""
        return random.choice(self.system_prompts)
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text - optimized version supporting multiple domains."""
        # Convert to lowercase
        text_lower = text.lower()
        
        # Find keywords - optimization: avoid tokenization and use direct string matching
        topics = set()
        for keyword, topic in self.all_keywords.items():
            if keyword in text_lower:
                topics.add(topic)
        
        # If no topics found, assign a general category based on content analysis
        if not topics:
            # Check for financial terms first (our primary domain of interest)
            if any(term in text_lower for term in ['money', 'finance', 'bank', 'cost', 'price', 'dollar', 'euro', 'payment']):
                return ["General Finance"]
            else:
                return ["General Knowledge"]
                
        return list(topics)
    
    def is_response_too_short(self, response_text: str) -> bool:
        """Check if an assistant response is too short to be useful.
        
        Returns True if the response is too short (should be filtered out).
        """
        # Check character length
        if len(response_text) < self.min_response_length:
            return True
            
        # Check word count (more accurate for determining actual content)
        words = response_text.split()
        if len(words) < self.min_response_words:
            return True
            
        # Check if it's likely a single answer response (yes/no, single value)
        # Common patterns in brief QA responses:
        short_answer_patterns = [
            r"^\s*yes\s*\.?\s*$",  # Just "yes"
            r"^\s*no\s*\.?\s*$",    # Just "no"
            r"^\s*\d+\.?\d*\s*$",   # Just a number
            r"^\s*\$\s*\d+\.?\d*\s*$",  # Just a dollar amount
            r"^\s*[a-zA-Z0-9\s\.\,\:\;\-\_\'\"\(\)]{1,50}\s*$"  # Very short phrase (< 50 chars)
        ]
        
        for pattern in short_answer_patterns:
            if re.match(pattern, response_text, re.IGNORECASE):
                return True
                
        return False
    
    def process_sample_batch(self, batch: List[str], selected_indices: Optional[set] = None) -> List[Dict[str, Any]]:
        """Process a batch of samples in parallel."""
        valid_exchanges = []
        
        for idx, line in enumerate(batch):
            # Skip if this line wasn't selected (when using percentage)
            if selected_indices is not None and idx not in selected_indices:
                continue
                
            try:
                sample = json.loads(line.strip())
                
                # Check if this is a valid exchange (no Chinese)
                if not self.filter_chinese_content(sample):
                    continue
                
                # Ensure we have both user and assistant content
                if "user" not in sample or "assistant" not in sample:
                    continue
                    
                # Filter out short answers / QA style responses
                if self.is_response_too_short(sample["assistant"]):
                    continue
                
                # Generate a hash to track usage (will manage usage counts later)
                exchange_str = json.dumps({
                    "user": sample["user"],
                    "assistant": sample["assistant"]
                }, sort_keys=True)
                exchange_hash = hashlib.md5(exchange_str.encode()).hexdigest()
                
                # Extract topics from the user message
                topics = self.extract_topics(sample["user"])
                
                # Add to valid exchanges
                valid_exchanges.append({
                    "user": sample["user"].strip(),
                    "assistant": sample["assistant"].strip(),
                    "topics": topics,
                    "hash": exchange_hash
                })
                
            except json.JSONDecodeError:
                # Skip malformed JSON
                continue
                
        return valid_exchanges
    
    def collect_valid_exchanges(self) -> List[Dict[str, Any]]:
        """Collect valid exchanges from the dataset - optimized with parallel processing."""
        start_time = time.time()
        print(f"Reading dataset from {self.input_file}...")
        
        # Count total samples
        total_samples = 0
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for _ in f:
                total_samples += 1
        
        # Determine how many samples to use
        samples_to_use = int(total_samples * self.dataset_percentage)
        use_all = self.dataset_percentage >= 1.0
        
        # If not using all, generate indices to sample
        selected_indices = None
        if not use_all:
            all_indices = list(range(total_samples))
            random.shuffle(all_indices)
            selected_indices = set(all_indices[:samples_to_use])
        
        print(f"Collecting valid exchanges from {samples_to_use} samples out of {total_samples} total...")
        
        valid_exchanges = []
        exchange_usage_count = defaultdict(int)
        
        # Read and process in batches
        with open(self.input_file, 'r', encoding='utf-8') as f:
            # Process file in batches for better memory management
            batch = []
            batch_indices = set()
            lines_processed = 0
            
            with tqdm.tqdm(total=total_samples, desc="Reading samples") as pbar:
                for idx, line in enumerate(f):
                    # If we're using a subset and this index isn't selected, skip
                    if not use_all and idx not in selected_indices:
                        pbar.update(1)
                        continue
                    
                    batch.append(line)
                    if not use_all:
                        batch_indices.add(idx)
                    
                    # Process when batch is full
                    if len(batch) >= self.batch_size:
                        # Process batch
                        with multiprocessing.Pool(self.num_workers) as pool:
                            # Split batch for parallel processing
                            batch_size = len(batch)
                            chunk_size = max(1, batch_size // self.num_workers)
                            chunks = [batch[i:i+chunk_size] for i in range(0, batch_size, chunk_size)]
                            
                            # Process chunks in parallel
                            results = pool.map(self.process_sample_batch, chunks)
                            
                            # Combine results
                            batch_exchanges = []
                            for result in results:
                                batch_exchanges.extend(result)
                        
                        # Deduplicate and enforce usage limits
                        for exchange in batch_exchanges:
                            exchange_hash = exchange["hash"]
                            if exchange_usage_count[exchange_hash] < self.max_sample_usage:
                                valid_exchanges.append(exchange)
                                exchange_usage_count[exchange_hash] += 1
                        
                        # Clear batch for next round
                        batch = []
                        batch_indices = set()
                        
                        # Update progress
                        pbar.update(self.batch_size)
                    
                # Process remaining samples
                if batch:
                    batch_exchanges = self.process_sample_batch(batch, batch_indices if not use_all else None)
                    for exchange in batch_exchanges:
                        exchange_hash = exchange["hash"]
                        if exchange_usage_count[exchange_hash] < self.max_sample_usage:
                            valid_exchanges.append(exchange)
                            exchange_usage_count[exchange_hash] += 1
                    
                    # Update progress
                    pbar.update(len(batch))
        
        elapsed = time.time() - start_time
        print(f"Collected {len(valid_exchanges)} valid exchanges in {elapsed:.2f} seconds")
        return valid_exchanges
    
    def group_by_topic(self, exchanges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group exchanges by their primary topic."""
        start_time = time.time()
        topic_groups = defaultdict(list)
        
        # Group exchanges by topic
        for exchange in exchanges:
            # Use the first topic as the primary topic
            primary_topic = exchange["topics"][0] if exchange["topics"] else "General Finance"
            topic_groups[primary_topic].append(exchange)
        
        # Shuffle each topic group for randomness
        for topic in topic_groups:
            random.shuffle(topic_groups[topic])
        
        elapsed = time.time() - start_time
        print(f"Grouped exchanges by {len(topic_groups)} topics in {elapsed:.2f} seconds")
        return topic_groups
    
    def create_multi_turn_conversation(self, exchanges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a multi-turn conversation from a list of exchanges."""
        # Determine number of turns (between min_turns and max_turns)
        num_turns = min(len(exchanges), random.randint(self.min_turns, self.max_turns))
        
        # Select a subset of exchanges for this conversation
        selected_exchanges = exchanges[:num_turns]
        
        # Build the conversation messages
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": self.generate_system_prompt()
        })
        
        # Add user-assistant exchanges
        for exchange in selected_exchanges:
            messages.append({
                "role": "user",
                "content": exchange["user"]
            })
            messages.append({
                "role": "assistant",
                "content": exchange["assistant"]
            })
        
        # Create metadata
        conversation_id = self.get_conversation_id()
        
        # Get topics from all exchanges
        all_topics = set()
        for exchange in selected_exchanges:
            all_topics.update(exchange["topics"])
        
        metadata = {
            "conversation_id": conversation_id,
            "turns": num_turns,  # Number of user-assistant exchange pairs
            "source": "Finance-Instruct-500k",
            "topics": list(all_topics)
        }
        
        return {
            "messages": messages,
            "metadata": metadata
        }
    
    def create_conversations(self, valid_exchanges: List[Dict[str, Any]], topic_groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Create a mix of coherent and random multi-turn conversations."""
        start_time = time.time()
        conversations = []
        
        # Number of conversations to generate (ensure we have at least min_turns valid exchanges)
        num_exchanges = len(valid_exchanges)
        max_conversations = num_exchanges // self.min_turns
        
        # Calculate how many coherent vs random conversations to create
        print(f"Can create up to {max_conversations} conversations from {num_exchanges} exchanges")
        num_coherent = int(max_conversations * self.coherent_percentage)
        num_random = max_conversations - num_coherent
        
        print(f"Creating {num_coherent} coherent and {num_random} random conversations...")
        
        # ------- COHERENT CONVERSATIONS -------
        # Create coherent conversations (grouped by topic)
        coherent_count = 0
        for topic, exchanges in tqdm.tqdm(topic_groups.items(), desc="Creating coherent conversations"):
            # If we have enough exchanges in this topic for at least min_turns
            if len(exchanges) >= self.min_turns:
                # How many conversations can we create from this topic group?
                possible_convos = len(exchanges) // self.min_turns
                # How many conversations to actually create
                to_create = min(possible_convos, (num_coherent - coherent_count))
                
                for i in range(to_create):
                    # Select a contiguous slice of exchanges
                    start_idx = i * self.min_turns
                    end_idx = min(start_idx + self.max_turns, len(exchanges))
                    topic_exchanges = exchanges[start_idx:end_idx]
                    
                    conversation = self.create_multi_turn_conversation(topic_exchanges)
                    conversation["metadata"]["coherent"] = True
                    conversation["metadata"]["primary_topic"] = topic
                    conversations.append(conversation)
                    
                    coherent_count += 1
                    if coherent_count >= num_coherent:
                        break
            
            if coherent_count >= num_coherent:
                break
        
        print(f"Created {coherent_count} coherent conversations")
        
        # ------- RANDOM CONVERSATIONS -------
        # Optimize random conversation creation by processing in larger batches
        remaining_exchanges = valid_exchanges.copy()
        random.shuffle(remaining_exchanges)
        
        # Prepare to create random conversations in bulk
        random_conversations = []
        random_count = 0
        
        # Faster approach: Process random conversations in batches
        # Similar to how we process coherent conversations
        print("Creating random conversations...")
        remaining_len = len(remaining_exchanges)
        
        # Precompute how many random conversations we can create
        possible_random = remaining_len // self.min_turns
        to_create_random = min(possible_random, num_random)
        
        # Process in blocks of min_turns
        with tqdm.tqdm(total=to_create_random, desc="Creating random conversations") as pbar:
            for i in range(to_create_random):
                # Calculate how many turns to use for this conversation (between min and max)
                # Make sure we don't run out of exchanges
                max_possible_turns = min(self.max_turns, 
                                         (remaining_len - i * self.min_turns) // (to_create_random - i))
                if max_possible_turns < self.min_turns:
                    break
                    
                num_turns = random.randint(self.min_turns, max_possible_turns)
                
                # Take a slice of exchanges for this conversation
                start_idx = i * self.min_turns
                end_idx = start_idx + num_turns
                if end_idx > len(remaining_exchanges):
                    break
                    
                selected_exchanges = remaining_exchanges[start_idx:end_idx]
                
                # Create the conversation
                conversation = self.create_multi_turn_conversation(selected_exchanges)
                conversation["metadata"]["coherent"] = False
                conversations.append(conversation)
                
                random_count += 1
                pbar.update(1)
                
                if random_count >= num_random:
                    break
        
        print(f"Created {random_count} random conversations")
        
        elapsed = time.time() - start_time
        print(f"Created {len(conversations)} total conversations in {elapsed:.2f} seconds")
        return conversations

    def process_dataset(self) -> List[Dict[str, Any]]:
        """Process the dataset and create multi-turn conversations."""
        total_start_time = time.time()
        
        # Collect valid exchanges
        valid_exchanges = self.collect_valid_exchanges()
        
        if not valid_exchanges:
            print("No valid exchanges found in the dataset.")
            return []
        
        # Group exchanges by topic
        topic_groups = self.group_by_topic(valid_exchanges)
        
        # Create a mix of coherent and random conversations
        conversations = self.create_conversations(valid_exchanges, topic_groups)
        
        total_elapsed = time.time() - total_start_time
        print(f"Total processing time: {total_elapsed:.2f} seconds")
        
        return conversations

    def save_dataset(self, conversations: List[Dict[str, Any]]) -> None:
        """Save the processed dataset to a JSONL file."""
        start_time = time.time()
        output_path = os.path.join(self.output_dir, 'finance_instruct_processed.jsonl')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for conversation in conversations:
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        elapsed = time.time() - start_time
        print(f"Saved processed dataset to {output_path}")
        print(f"Total conversations saved: {len(conversations)}")
        print(f"Save time: {elapsed:.2f} seconds")

    def run(self) -> None:
        """Run the entire processing pipeline."""
        overall_start = time.time()
        conversations = self.process_dataset()
        self.save_dataset(conversations)
        overall_elapsed = time.time() - overall_start
        print(f"Overall runtime: {overall_elapsed:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Process Finance-Instruct-500k dataset')
    
    parser.add_argument('--input_file', type=str, 
                        default='/home/zahemen/datasets/v2/Finance-Instruct-500k/train.json',
                        help='Path to the input JSON file')
    
    parser.add_argument('--output_dir', type=str, 
                        default='/home/zahemen/datasets/v2',
                        help='Directory to save the processed dataset')
    
    parser.add_argument('--dataset_percentage', type=float, default=0.5,
                        help='Percentage of the dataset to use (0.0-1.0)')
    
    parser.add_argument('--max_sample_usage', type=int, default=2,
                        help='Maximum number of times a sample can be used')
    
    parser.add_argument('--min_turns', type=int, default=1,
                        help='Minimum number of conversation turns to include')
    
    parser.add_argument('--max_turns', type=int, default=5,
                        help='Maximum number of conversation turns to include')
    
    parser.add_argument('--coherent_percentage', type=float, default=0.5,
                        help='Percentage of conversations that should be topic-coherent (0.0-1.0)')
    
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes for parallel processing')
    
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size for processing samples')
    
    parser.add_argument('--min_response_length', type=int, default=150,
                        help='Minimum character length for assistant responses')
    
    parser.add_argument('--min_response_words', type=int, default=20,
                        help='Minimum word count for assistant responses')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    processor = FinanceInstructDatasetProcessor(
        input_file=args.input_file,
        output_dir=args.output_dir,
        dataset_percentage=args.dataset_percentage,
        max_sample_usage=args.max_sample_usage,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        coherent_percentage=args.coherent_percentage,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        min_response_length=args.min_response_length,
        min_response_words=args.min_response_words,
        seed=args.seed
    )
    
    processor.run()


if __name__ == "__main__":
    main()