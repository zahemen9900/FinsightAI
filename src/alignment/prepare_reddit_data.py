import json
import random
import pandas as pd
import re
import numpy as np
import logging
import hashlib
from typing import Dict, List, Tuple
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from rich.logging import RichHandler
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from textblob import TextBlob
import joblib
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from time import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[
        RichHandler(),
        logging.FileHandler('reddit_data_processing.log')  # Add file handler
    ],
)
logger = logging.getLogger('rich')

# Download required NLTK data
# nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet'])
# from nltk.corpus import wordnet

# Before running the script, you need to download these models:
# python -m spacy download en_core_web_lg  # Large English model with word vectors
# python -m spacy download en_core_web_md  # Medium English model with word vectors (alternative)

# Update the model loading section:
try:
    # Load the large model that includes word vectors
    nlp = spacy.load('en_core_web_md')  # Changed from en_core_web_sm
    logger.info("Loaded en_core_web_md model")
except OSError:
    try:
        # Fallback to medium model if large one is not available
        nlp = spacy.load('en_core_web_md')
        logger.info("Loaded en_core_web_md model")
    except OSError:
        logger.error("Could not load spaCy model with word vectors. Please run:")
        logger.error("python -m spacy download en_core_web_lg")
        logger.error("or")
        logger.error("python -m spacy download en_core_web_md")
        raise

# Load financial domain specific model if available (keep existing code)
try:
    nlp_financial = spacy.load('en_core_financial_web_sm')
except:
    logger.warning("Could not load financial domain-specific spaCy model. Using general model.")
    nlp_financial = nlp

class DatasetCleaner:
    def __init__(self, input_file: str, output_file: str, conv_starters_file: str, silent_warnings: bool = True):
        self.input_file = input_file
        self.output_file = output_file
        self.conv_starters_file = conv_starters_file
        self.silent_warnings = silent_warnings
        self.quality_threshold = 0.6  # Lowered from 0.75
        self.max_prompt_length = 1000  # Reduced from 1500 to ensure more focused responses
        
        # Common profanity patterns
        self.profanity_patterns = [
            # General profanity
            r'\b(fuck|shit|damn|bitch|crap|ass|dick|porn|nsfw|cunt|bastard|slut|piss|hell)\b',
            r'\b(wtf|stfu|omfg|btch|wtf|af|lmfao|omg|fml)\b',

            # Sexual or explicit content
            r'\b(sex|sexy|naked|nude|fap|thot|blowjob|handjob|cum|boobs|butt|dildo|vagina|penis|orgasm|xxx)\b',
            r'\b(erotic|fetish|kinky|bdsm|hentai|camgirl|stripper)\b',

            # Hate speech and slurs
            r'\b(nigga|nigger|chink|spick|kike|faggot|tranny|dyke|gook|retard|moron)\b',
            r'\b(white trash|wetback|redneck|camel jockey|sand nigger)\b',

            # Abbreviations or acronyms
            r'\b(idgaf|gtfo|nsfw|nsfl|smh|tmi|fubar|lmao|lmfao)\b',

            # Aggressive or insulting phrases
            r'\b(kill yourself|die in a fire|shut the fuck up|piece of shit|you suck|go to hell)\b',
            r'\b(loser|idiot|stupid|dumbass|jackass|scumbag|prick|twat)\b'
        ]
        
        # Load conversational starters
        self.conv_starters = self.load_conv_starters()

        # Add new parameters
        self.cache_dir = Path("/home/zahemen/datasets/dataset_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.min_financial_relevance = 0.35  # Lowered from 0.7
        self.max_similarity_threshold = 0.80  # For deduplication
        self.complexity_threshold = 0.3  # Lowered from 0.4
        self.min_words = 5
        self.max_words = 200  # Limit response length
        
        # Initialize vectorizer for similarity checking
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Financial domain keywords
        self.financial_keywords = set([
            'investment', 'stock', 'bond', 'market', 'fund', 'dividend',
            'portfolio', 'asset', 'equity', 'risk', 'return', 'trading',
            'finance', 'bank', 'capital', 'debt', 'credit', 'interest',
            'inflation', 'economy', 'security', 'hedge', 'option', 'future',
            'liquidity', 'valuation', 'yield', 'income', 'tax', 'audit',
            'accounting', 'budget', 'loan', 'mortgage', 'retirement', 'pension',
            'savings', 'wealth', 'insurance', 'policy', 'regulation', 'compliance',
            'audit', 'fraud', 'scam', 'money', 'payment', 'transaction', 'currency',
            'exchange', 'crypto', 'blockchain', 'token', 'coin', 'wallet', 'mining',
            'staking', 'defi', 'nft', 'yield farming', 'staking', 'staking pool',
            'liquidity pool', 'impermanent loss', 'rug pull', 'pump and dump',
            'bear market', 'bull market', 'short selling', 'long position',
            'margin trading', 'leverage', 'volatility', 'correlation', 'beta',
            'alpha', 'sharpe ratio', 'sortino ratio', 'treynor ratio', 'jensen alpha',
            'efficient market hypothesis', 'random walk', 'technical analysis',
            'fundamental analysis', 'quantitative analysis', 'quantitative easing',
            'monetary policy', 'fiscal policy', 'central bank', 'interest rate',
            'inflation rate', 'deflation', 'stagflation', 'recession', 'depression',
            'recovery', 'growth', 'expansion', 'peak', 'trough', 'cycle', 'bubble',
            'crash', 'black swan', 'tail risk', 'systemic risk', 'counterparty risk',
            'credit risk', 'liquidity risk', 'market risk', 'operational risk',
            'regulatory risk', 'political risk', 'economic risk', 'geopolitical risk',
            'environmental risk', 'social risk', 'ESG', 'sustainable investing',
            'impact investing', 'green finance', 'carbon footprint', 'carbon offset',
            'carbon credit', 'sustainability', 'climate change', 'global warming',
            'renewable energy', 'clean energy', 'green energy', 'solar power',
            'arbitrage', 'bid-ask spread', 'buyback', 'capital gains', 'commodities',
            'contango', 'backwardation', 'derivatives', 'collateral', 'securitization',
            'special purpose vehicle', 'structured finance', 'credit default swap',
            'mortgage-backed security', 'asset-backed security', 'zero-coupon bond',
            'fixed income', 'floating rate', 'coupon rate', 'par value', 'discount bond',
            'convertible bond', 'sovereign debt', 'municipal bond', 'junk bond',
            'investment grade', 'deleveraging', 'leveraged buyout', 'hostile takeover',
            'mergers and acquisitions', 'initial public offering', 'direct listing',
            'secondary offering', 'private equity', 'venture capital', 'angel investing',
            'seed funding', 'series A funding', 'series B funding', 'unicorn startup',
            'valuation multiple', 'earnings per share', 'price-to-earnings ratio',
            'price-to-book ratio', 'enterprise value', 'discounted cash flow',
            'intrinsic value', 'time value of money', 'net present value', 'internal rate of return',
            'hurdle rate', 'capital asset pricing model', 'weighted average cost of capital',
            'modigliani-miller theorem', 'efficient frontier', 'mean-variance optimization',
            'black-scholes model', 'binomial options pricing', 'delta hedging',
            'gamma exposure', 'vega risk', 'theta decay', 'implied volatility',
            'historical volatility', 'market efficiency', 'behavioral finance',
            'loss aversion', 'overconfidence bias', 'anchoring effect',
            'herd mentality', 'hot hand fallacy', 'representativeness heuristic',
            'recency bias', 'mental accounting', 'prospect theory',
            'endowment effect', 'confirmation bias', 'framing effect',
            'regret aversion', 'status quo bias', 'availability bias',
            'yield curve', 'inverted yield curve', 'steepening yield curve',
            'flattening yield curve', 'negative interest rates', 'quantitative tightening',
            'forward guidance', 'dual mandate', 'open market operations',
            'fractional reserve banking', 'money supply', 'velocity of money',
            'fiat currency', 'gold standard', 'Bretton Woods system',
            'sovereign wealth fund', 'foreign exchange reserves', 'current account deficit',
            'balance of payments', 'trade surplus', 'trade deficit',
            'purchasing power parity', 'interest rate parity', 'carry trade',
            'capital flight', 'hot money flows', 'globalization', 'deglobalization',
            'currency peg', 'dirty float', 'speculative attack', 'devaluation',
            'revaluation', 'capital controls', 'shadow banking system',
            'systemically important financial institution', 'moral hazard',
            'adverse selection', 'principal-agent problem', 'asymmetric information',
            'lemon market', 'creative destruction', 'invisible hand', 'Keynesian economics',
            'Austrian economics', 'neoclassical economics', 'behavioral economics',
            'modern monetary theory', 'supply-side economics', 'demand-side economics',
            'trickle-down economics', 'helicopter money', 'fiscal cliff',
            'debt ceiling', 'government bond', 'gilt', 'treasury yield',
            'sovereign default', 'hyperinflation', 'stagflation', 'currency crisis',
            'bank run', 'contagion effect', 'bailout', 'bail-in',
            'too big to fail', 'stress test', 'macroprudential regulation',
            'Basel III', 'Dodd-Frank Act', 'Glass-Steagall Act',
            'Volcker Rule', 'monetary aggregate', 'broad money', 'narrow money',
            'M1', 'M2', 'M3', 'velocity of circulation', 'GDP deflator',
            'nominal GDP', 'real GDP', 'gross national product', 'disposable income',
            'consumer sentiment', 'purchasing managers index', 'leading economic indicators',
            'coincident economic indicators', 'lagging economic indicators',
            'misery index', 'big mac index', 'tobin’s q', 'real wages',
            'human capital', 'knowledge economy', 'creative economy', 'gig economy',
            'circular economy', 'cryptographic hash', 'hash rate',
            'proof of stake', 'proof of work', 'layer 1 blockchain',
            'layer 2 scaling solution', 'oracle problem', 'flash loan',
            'yield aggregator', 'stablecoin', 'algorithmic stablecoin',
            'central bank digital currency', 'tokenomics', 'smart contract risk',
            'gas fees', 'EIP-1559', 'rollups', 'zk-rollups', 'optimistic rollups',
            'liquidity mining', 'governance token', 'fork', 'soft fork', 'hard fork',
            'wrapped token', 'cross-chain interoperability', 'bridging assets',
            'real-world assets', 'security token', 'tokenized securities',
            'fractional ownership', 'NFT royalties', 'soulbound tokens',
            'AI in finance', 'automated trading', 'algorithmic trading',
            'high-frequency trading', 'market making', 'dark pools',
            'quantitative hedge fund', 'factor investing', 'momentum investing',
            'contrarian investing', 'index investing', 'smart beta', 'robo-advisor',
            'passive investing', 'active investing', 'direct indexing',
            'retail investor', 'institutional investor', 'sovereign investor',
            'family office', 'endowment fund', 'pension fund', 'defined benefit plan',
            'defined contribution plan', 'annuity', 'life insurance',
            'reinsurance', 'insurance underwriting', 'catastrophe bond',
            'parametric insurance', 'microinsurance', 'insurance-linked securities',
            'policyholder surplus', 'policy lapse', 'self-insurance',
            'fiduciary duty', 'proxy voting', 'stewardship', 'shareholder activism',
            'poison pill', 'golden parachute', 'white knight', 'greenmail',
            'dual-class shares', 'corporate governance', 'ESG scoring',
            'carbon neutrality', 'emissions trading', 'green bonds',
            'social impact bonds', 'blue economy', 'circular finance',
            'impact measurement', 'stakeholder capitalism', 'benefit corporation',
            'public-private partnership', 'crowdfunding', 'peer-to-peer lending',
            'alternative investments', 'artificial intelligence in finance',
            'machine learning in trading', 'neural networks in investing',
            'alternative data', 'satellite imagery in finance',
            'sentiment analysis in trading', 'predictive analytics',
            'automated wealth management', 'fintech disruption',
            'regtech', 'insurtech', 'suptech', 'financial inclusion',
            'mobile banking', 'cashless society', 'open banking',
            'digital wallet interoperability', 'banking as a service',
            'embedded finance', 'central bank autonomy'
        ])
        self.sample_usage_counter = {}  # Track how many times each sample is used
        self.max_sample_usage = 5  # Maximum times a sample can be used

        # Add individual cache files
        self.cache_paths = {
            'initial_load': self.cache_dir / "initial_data.joblib",
            'score_filtered': self.cache_dir / "score_filtered.joblib",
            'cleaned_text': self.cache_dir / "cleaned_text.joblib",
            'text_quality': self.cache_dir / "text_quality.joblib",
            'complexity': self.cache_dir / "complexity.joblib",
            'financial_relevance': self.cache_dir / "financial_relevance.joblib",
            'final_filtered': self.cache_dir / "final_filtered.joblib"
        }

    def load_conv_starters(self) -> List[Dict[str, str]]:
        """Load conversational starters from a text file."""
        starters = []
        with open(self.conv_starters_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                user_line = lines[i].strip()
                assistant_line = lines[i + 1].strip()
                if user_line.startswith("User:") and assistant_line.startswith("Assistant:"):
                    starters.append({
                        "user": user_line.replace("User:", "").strip(),
                        "assistant": assistant_line.replace("Assistant:", "").strip()
                    })
        return starters

    def contains_profanity(self, text: str) -> bool:
        """Check if text contains profanity using regex patterns."""
        if not isinstance(text, str):
            return True
        
        text = text.lower()
        return any(bool(re.search(pattern, text, re.IGNORECASE)) 
                  for pattern in self.profanity_patterns)

    def load_data(self) -> pd.DataFrame:
        """Load and prepare the dataset."""
        all_data = []
        try:
            with open(self.input_file, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    try:
                        all_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON at line {line_num}")
            logger.info(f"Successfully loaded {len(all_data)} records from {self.input_file}")
            return pd.DataFrame(all_data)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """Enhanced text cleaning for Reddit-style content"""
        if not isinstance(text, str):
            return ""
        
        # Remove markdown-style links with their text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', '', text)
        
        # Remove plain URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-style formatting
        replacements = [
            # Remove markdown formatting
            (r'\*\*(.+?)\*\*', r'\1'),  # Bold
            (r'\*(.+?)\*', r'\1'),      # Italic
            (r'\_(.+?)\_', r'\1'),      # Underscore emphasis
            (r'\~\~(.+?)\~\~', r'\1'),  # Strikethrough
            (r'\^(.+?)(?:\s|$)', r'\1'), # Superscript
            
            # Remove Reddit quote blocks and lists
            (r'^\s*>\s*(.+?)$', r'\1'),  # Quote blocks
            # (r'^\s*\*\s+(.+?)$', r'\1'), # Unordered lists
            # (r'^\s*\d+\.\s+(.+?)$', r'\1'), # Ordered lists
            
            # Remove special characters and emojis
            (r'&amp;', '&'),
            (r'&lt;', '<'),
            (r'&gt;', '>'),
            (r'[━┃┏┓┗┛│└┘╭╮╯╰▀▄█▌▐░▒▓]', ''),
            
            # Remove hashtags and their text completely
            (r'#\w+', ''),
            
            # Remove Reddit-style headers
            (r'^#+\s*(.+?)$', r'\1'),
            
            # Remove common Reddit artifacts
            (r'(?i)edit\s*\d*\s*:', ''),
            (r'(?i)update\s*\d*\s*:', ''),
            (r'(?i)tldr[:,]*', ''),
            (r'(?i)thanks? for (?:the)? (?:gold|silver|platinum|award).*', ''),
        ]
        
        # Apply all replacements
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        
        # Clean up whitespace
        text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with single space
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()
        
        # Ensure proper sentence structure
        sentences = []
        for sent in text.split('.'):
            sent = sent.strip()
            if sent:
                # Capitalize first letter if it's not already
                if sent[0].islower():
                    sent = sent[0].upper() + sent[1:]
                sentences.append(sent)
        
        text = '. '.join(sentences)
        
        # Ensure proper ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'
            
        return text

    def is_quality_response(self, text: str) -> bool:
        """Additional quality checks for responses with enhanced filtering"""
        if not text:
            return False
            
        # Basic length checks
        words = text.split()
        if len(words) < self.min_words or len(words) > self.max_words:
            return False
            
        # Check for coherent sentence structure
        sentences = sent_tokenize(text)
        if len(sentences) < 1:
            return False
            
        # Additional quality checks
        content_flags = [
            # URL and markdown checks
            len(re.findall(r'http[s]?://', text)) > 0,
            any(marker in text for marker in ['[', ']', '*', '#', '~~', '>', '`']),
            
            # Repetitive phrases and low-effort content
            any(text.lower().count(phrase) > 1 for phrase in [
            "is there anything",
            "yes thank you",
            "please specify",
            "let me know",
            "hope this helps",
            "not financial advice",
            "just my opinion",
            "do your own research",
            "this is not advice"
            ]),
            
            # Style and formatting checks
            text.count('?') > 3,  # Excessive questions
            text.count('!') > 3,  # Excessive exclamations
            text.count('...') > 2,  # Too many ellipses
            len(re.findall(r'[A-Z]{3,}', text)) > 3,  # Too many all-caps words
            
            # Low-quality indicators
            any(ending in text.lower() for ending in [
            "edit:",
            "update:",
            "source:",
            "tldr",
            "tl;dr",
            "thanks for reading",
            "thanks for coming to my ted talk",
            "obligatory",
            "disclaimer:",
            "not financial advice",
            "this is the way",
            "to the moon"
            ]),
            
            # Structure checks
            len(text.split()) < 10,  # Too short
            text.count('\n') > 5,    # Too many line breaks
            sum(1 for c in text if c.isupper()) / len(text) > 0.3,  # Too many caps
            len(re.findall(r'(\w)\1{2,}', text)) > 0,  # Repeated characters (e.g., "yesss")
            
            # Boilerplate checks
            any(phrase in text.lower() for phrase in [
            "not investment advice",
            "this is just my opinion",
            "don't sue me",
            "do your dd",
            "for entertainment only"
            ])
        ]
        
        return not any(content_flags)

    def assess_text_quality(self, text: str) -> float:
        """Assess the quality of text based on multiple factors."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0

        try:
            doc = nlp(text[:10000])  # Limit text length for processing
            
            avg_word_length = np.mean([len(token.text) for token in doc])
            sentence_count = len(list(doc.sents))
            has_punctuation = any(token.is_punct for token in doc)
            proper_capitalization = text[0].isupper() if text else False
            
            sentences = sent_tokenize(text[:10000])
            min_sent_length = 3
            coherent_sentences = sum(1 for sent in sentences if len(sent.split()) >= min_sent_length)
            
            scores = [
                min(avg_word_length / 10, 1.0),
                min(sentence_count / 3, 1.0),
                float(has_punctuation),
                float(proper_capitalization),
                coherent_sentences / max(len(sentences), 1)
            ]
            
            return np.mean(scores)
        except Exception as e:
            logger.warning(f"Error assessing text quality: {e}")
            return 0.0

    def is_conversational(self, text: str) -> bool:
        """Check if text appears to be conversational."""
        if not isinstance(text, str):
            return False
            
        indicators = [
            # Greetings and courtesy
            r'\b(hi|hello|hey|thanks|thank you|please|welcome)\b',
            
            # Questions
            r'\?',
            
            # Personal pronouns
            r'\b(you|your|yours)\b',
            r'\b(I|me|my|mine)\b',
            r'\b(we|us|our|ours)\b',
            
            # Conversational phrases
            r'\b(understand|hope|suggest|recommend|advice|help)\b',
            r'\b(let me|here\'s|take a look|consider)\b',
            
            # Engagement markers
            r'\b(actually|basically|essentially|specifically)\b',
            r'\b(sure|definitely|absolutely|certainly)\b',
            
            # Opinion indicators
            r'\b(think|believe|feel|assume|suppose)\b'
        ]

        
        score = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in indicators)
        return score >= 2

    def filter_lengthy_prompts(self, text: str) -> bool:
        """Filter out very lengthy prompts."""
        return len(text) <= self.max_prompt_length

    def can_use_sample(self, idx: int) -> bool:
        """Check if a sample can still be used"""
        return self.sample_usage_counter.get(idx, 0) < self.max_sample_usage
        
    def mark_sample_used(self, idx: int):
        """Mark a sample as used"""
        self.sample_usage_counter[idx] = self.sample_usage_counter.get(idx, 0) + 1

    def convert_to_sft_format(self, df: pd.DataFrame) -> List[Dict]:
        """Convert records to chat format for fine-tuning with conversational pairs"""
        formatted_data = []
        total_samples = len(df)
        num_conversational_samples = int(total_samples * 0.8)
        
        # Shuffle the DataFrame to randomize selection
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Process conversational samples
        for i in range(num_conversational_samples):
            try:
                row = df.iloc[i]
                conversation_id = hashlib.sha256(row["id"].encode()).hexdigest()
                
                # Clean texts
                cleaned_body = self.clean_text(row["body"])
                cleaned_title = self.clean_text(row["title"])
                cleaned_selftext = self.clean_text(row["selftext"] if pd.notna(row["selftext"]) else "")
                
                # Skip if response quality check fails
                if not self.is_quality_response(cleaned_body):
                    continue
                
                # Format the question/prompt - only include title if it ends with "?"
                if cleaned_title.strip().endswith('?'):
                    question = cleaned_title
                    if cleaned_selftext:
                        question += f"\n\n{cleaned_selftext}"
                else:
                    question = cleaned_selftext if cleaned_selftext else cleaned_title

                # Select a random conversational starter
                starter = random.choice(self.conv_starters)
                
                # Create multi-turn conversation
                messages = [
                    {
                        "role": "system",
                        "content": "You are FinSight, an AI financial advisor. Provide accurate and helpful financial guidance."
                    },
                    {
                        "role": "user",
                        "content": starter["user"]
                    },
                    {
                        "role": "assistant",
                        "content": starter["assistant"]
                    },
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": cleaned_body
                    }
                ]
                
                # Add additional turns
                num_turns = random.randint(3, 7)
                available_indices = [idx for idx in df.index if self.can_use_sample(idx)]
                selected_indices = random.sample(available_indices, min(num_turns, len(available_indices)))
                
                for idx in selected_indices:
                    turn_row = df.loc[idx]
                    turn_question = self.clean_text(turn_row["title"])
                    turn_answer = self.clean_text(turn_row["body"])
                    
                    if turn_question and turn_answer:
                        messages.append({"role": "user", "content": turn_question})
                        messages.append({"role": "assistant", "content": turn_answer})
                        self.mark_sample_used(idx)
                
                formatted_data.append({
                    "messages": messages,
                    "metadata": {
                        "source": f"reddit: {row['subreddit']}",
                        "conversation_id": conversation_id
                    }
                })
            except Exception as e:
                if not self.silent_warnings:
                    logger.warning(f"Failed to convert row {row.get('id', 'UNKNOWN')}: {e}")
        
        # Process remaining samples without conversational starters
        for i in range(num_conversational_samples, total_samples):
            try:
                row = df.iloc[i]
                conversation_id = hashlib.sha256(row["id"].encode()).hexdigest()
                
                # Clean texts
                cleaned_body = self.clean_text(row["body"])
                cleaned_title = self.clean_text(row["title"])
                cleaned_selftext = self.clean_text(row["selftext"] if pd.notna(row["selftext"]) else "")
                
                # Skip if response quality check fails
                if not self.is_quality_response(cleaned_body):
                    continue
                
                # Format the question/prompt
                question = cleaned_title
                if cleaned_selftext:
                    question += f"\n\n{cleaned_selftext}"
                
                # Create multi-turn conversation without starter
                messages = [
                    {
                        "role": "system",
                        "content": "You are FinSight, an AI financial advisor. Provide accurate and helpful financial guidance."
                    },
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": cleaned_body
                    }
                ]
                
                # Add additional turns
                num_turns = random.randint(3, 7)
                available_indices = [idx for idx in df.index if self.can_use_sample(idx)]
                selected_indices = random.sample(available_indices, min(num_turns, len(available_indices)))
                
                for idx in selected_indices:
                    turn_row = df.loc[idx]
                    turn_question = self.clean_text(turn_row["title"])
                    turn_answer = self.clean_text(turn_row["body"])
                    
                    if turn_question and turn_answer:
                        messages.append({"role": "user", "content": turn_question})
                        messages.append({"role": "assistant", "content": turn_answer})
                        self.mark_sample_used(idx)
                
                formatted_data.append({
                    "messages": messages,
                    "metadata": {
                        "source": f"reddit: {row['subreddit']}",
                        "conversation_id": conversation_id
                    }
                })
            except Exception as e:
                if not self.silent_warnings:
                    logger.warning(f"Failed to convert row {row.get('id', 'UNKNOWN')}: {e}")
        
        return formatted_data

    def calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity using multiple metrics"""
        if not isinstance(text, str) or len(text.strip()) < 10:
            return 0.0
            
        try:
            # Get TextBlob metrics
            blob = TextBlob(text)
            
            # Calculate various complexity metrics
            avg_word_length = np.mean([len(word) for word in blob.words])
            avg_sentence_length = np.mean([len(sent.words) for sent in blob.sentences])
            
            # Calculate lexical diversity
            words = blob.words
            unique_words = set(words)
            lexical_diversity = len(unique_words) / len(words) if words else 0
            
            # Get percentage of complex words (3+ syllables)
            complex_words = sum(1 for word in unique_words if len(TextBlob(word).words[0]) >= 8)
            complex_ratio = complex_words / len(unique_words) if unique_words else 0
            
            # Calculate final complexity score
            complexity_score = np.mean([
                min(avg_word_length / 10, 1.0),
                min(avg_sentence_length / 20, 1.0),
                lexical_diversity,
                complex_ratio
            ])
            
            return complexity_score
        except Exception as e:
            logger.warning(f"Error calculating complexity: {e}")
            return 0.0

    def assess_financial_relevance(self, text: str) -> float:
        """Assess how relevant the text is to financial topics using multiple metrics"""
        if not isinstance(text, str):
            return 0.0
            
        try:
            start_time = time()
            logger.debug(f"Starting financial relevance assessment for text of length {len(text)}")

            # Use both general and financial models
            logger.debug("Processing with spaCy models...")
            doc_general = nlp(text[:10000]) # Limit text length for processing
            doc_financial = nlp_financial(text[:10000])
            
            # Enhanced entity recogni tion with weighted categories
            logger.debug("Calculating entity scores...")
            entity_weights = {
                'ORG': 1.0,      # Organizations
                'MONEY': 1.2,    # Monetary values
                'PERCENT': 1.1,  # Percentages
                'QUANTITY': 0.8, # Quantities
                'DATE': 0.5,     # Dates (some relevance for financial context)
                'GPE': 0.3       # Geopolitical entities
            }
            
            # Calculate weighted entity score
            entity_score = sum(
                entity_weights.get(ent.label_, 0)
                for ent in doc_financial.ents
                if ent.label_ in entity_weights
            ) / max(len(doc_financial.ents), 1)
            
            logger.debug(f"Entity score: {entity_score:.3f}")
            
            # Keyword analysis with context
            logger.debug("Performing keyword analysis...")
            words = text.lower().split()
            word_set = set(words)
            
            # Calculate keyword density
            keyword_matches = len(word_set.intersection(self.financial_keywords))
            keyword_density = keyword_matches / max(len(words), 1)
            
            logger.debug(f"Keyword density: {keyword_density:.3f}")
            
            # Check for financial bigrams and phrases
            logger.debug("Checking financial phrases...")
            financial_phrases = [
                'market analysis', 'risk management', 'asset allocation',
                'interest rates', 'stock market', 'financial planning'
            ]
            phrase_matches = sum(1 for phrase in financial_phrases if phrase in text.lower())
            phrase_score = phrase_matches / max(len(financial_phrases), 1)
            
            logger.debug(f"Phrase score: {phrase_score:.3f}")
            
            # Semantic analysis using spaCy's word vectors
            logger.debug("Performing semantic analysis...")
            financial_topics = [
                'finance', 'investment', 'banking', 'trade', 'stock',
                'market', 'economy', 'money', 'crypto', 'currency',
                'fund', 'asset', 'portfolio', 'risk', 'return', 'dividend', 
                'interest', 'inflation', 'tax', 'loan', 'mortgage', 'savings', 
                'wealth', 'insurance', 'audit', 'accounting', 'budget', 'retirement',
                'pension', 'regulation', 'compliance', 'fraud', 'scam', 'payment',
                'transaction', 'exchange', 'blockchain', 'token', 'coin', 'wallet',
                'mining', 'staking', 'defi', 'nft', 'yield farming', 'liquidity pool',
                'impermanent loss', 'rug pull', 'pump and dump', 'bear market',
                'bull market', 'short selling', 'long position', 'margin trading',
                'leverage', 'volatility', 'correlation', 'beta', 'alpha', 'sharpe ratio',
                'sortino ratio', 'treynor ratio', 'jensen alpha', 'efficient market hypothesis',
                'random walk', 'technical analysis', 'fundamental analysis', 'quantitative analysis',
                'quantitative easing', 'monetary policy', 'fiscal policy', 'central bank', 'interest rate',
                'inflation rate', 'deflation', 'stagflation', 'recession', 'depression', 'recovery',
                'growth', 'expansion', 'peak', 'trough', 'cycle', 'bubble', 'crash', 'black swan',
                'tail risk', 'systemic risk', 'counterparty risk', 'credit risk', 'liquidity risk',
                'market risk', 'operational risk', 'regulatory risk', 'political risk', 'economic risk',
                'geopolitical risk', 'environmental risk', 'social risk', 'ESG', 'sustainable investing',
                'impact investing', 'green finance', 'carbon footprint', 'carbon offset', 'carbon credit',
                'sustainability', 'climate change', 'global warming', 'renewable energy', 'clean energy',
                'green energy', 'solar power', 'arbitrage', 'bid-ask spread', 'buyback', 'capital gains',
                'private equity', 'venture capital', 'angel investing', 'seed funding', 'series A funding',
                'series B funding', 'series C funding', 'leveraged buyout', 'mergers and acquisitions',
                'hostile takeover', 'share buyback', 'initial public offering (IPO)', 'special purpose acquisition company (SPAC)',
                'direct listing', 'secondary offering', 'private placement', 'hedge fund', 'mutual fund',
                'exchange-traded fund (ETF)', 'index fund', 'target-date fund', 'sovereign wealth fund',
                'family office', 'custodial account', 'margin call', 'stop-loss order', 'limit order',
                'market order', 'bid-ask spread', 'liquidity premium', 'discount rate', 'yield curve',
                'bond rating', 'credit default swap (CDS)', 'collateralized debt obligation (CDO)',
                'mortgage-backed security (MBS)', 'asset-backed security (ABS)', 'structured finance',
                'fixed-income securities', 'convertible bond', 'municipal bond', 'corporate bond',
                'sovereign bond', 'treasury bond', 'treasury bill', 'zero-coupon bond', 'junk bond',
                'high-yield bond', 'floating rate bond', 'perpetual bond', 'coupon rate',
                'principal payment', 'balloon payment', 'callable bond', 'puttable bond',
                'debt restructuring', 'forbearance', 'loan default', 'creditworthiness',
                'credit bureau', 'FICO score', 'debt-to-equity ratio', 'current ratio',
                'quick ratio', 'working capital', 'capital structure', 'capital budgeting',
                'return on assets (ROA)', 'return on equity (ROE)', 'return on investment (ROI)',
                'internal rate of return (IRR)', 'net present value (NPV)', 'discounted cash flow (DCF)',
                'earnings before interest and taxes (EBIT)', 'earnings before interest, taxes, depreciation, and amortization (EBITDA)',
                'price-to-earnings (P/E) ratio', 'price-to-book (P/B) ratio', 'enterprise value',
                'market capitalization', 'free cash flow (FCF)', 'dividend yield', 'dividend payout ratio',
                'stock split', 'reverse stock split', 'stock dilution', 'shareholder equity',
                'preferred stock', 'common stock', 'restricted stock unit (RSU)', 'stock option',
                'phantom stock', 'vesting schedule', 'exercise price', 'strike price',
                'in-the-money option', 'out-of-the-money option', 'covered call', 'naked put',
                'iron condor', 'butterfly spread', 'collar strategy', 'delta hedging',
                'gamma scalping', 'theta decay', 'VIX (volatility index)', 'greeks (delta, gamma, theta, vega, rho)',
                'carry trade', 'currency peg', 'foreign exchange reserves', 'balance of payments',
                'current account deficit', 'trade surplus', 'trade deficit', 'export-import balance',
                'tariffs', 'sanctions', 'quantitative tightening', 'yield spread', 'credit crunch',
                'bank run', 'shadow banking system', 'fractional reserve banking', 'Basel III regulations',
                'Dodd-Frank Act', 'Glass-Steagall Act', 'Volcker Rule', 'stress testing',
                'bailout', 'bail-in', 'too big to fail', 'systemically important financial institution (SIFI)',
                'financial contagion', 'moral hazard', 'asymmetric information', 'adverse selection',
                'lemon problem', 'principal-agent problem', 'agency cost', 'corporate governance',
                'proxy fight', 'dual-class shares', 'golden parachute', 'poison pill',
                'white knight', 'shareholder activism', 'socially responsible investing (SRI)',
                'triple bottom line (TBL)', 'green bonds', 'impact bonds', 'blue economy',
                'carbon trading', 'greenwashing', 'corporate social responsibility (CSR)',
                'behavioral finance', 'market efficiency', 'noise trader', 'sentiment analysis',
                'algorithmic trading', 'high-frequency trading (HFT)', 'flash crash',
                'mean reversion', 'momentum investing', 'contrarian investing',
                'factor investing', 'smart beta', 'value investing', 'growth investing',
                'income investing', 'small-cap stocks', 'mid-cap stocks', 'large-cap stocks',
                'emerging markets', 'developed markets', 'sovereign risk', 'country risk',
                'BRICS economies', 'frontier markets', 'macroeconomic indicators',
                'leading indicators', 'lagging indicators', 'coincident indicators',
                'consumer confidence index (CCI)', 'purchasing managers index (PMI)',
                'gross domestic product (GDP)', 'gross national product (GNP)',
                'human development index (HDI)', 'misery index', 'Gini coefficient',
                'Lorenz curve', 'Phillips curve', 'Laffer curve', 'Okuns law',
                'stagflation', 'hyperinflation', 'deleveraging', 'helicopter money',
                'debt monetization', 'sovereign default', 'capital flight',
                'hot money flows', 'brain drain', 'demographic dividend',
                'pension fund crisis', 'social security insolvency', 'universal basic income (UBI)',
                'negative income tax', 'wealth tax', 'estate tax', 'inheritance tax',
                'sin tax', 'flat tax', 'progressive tax', 'regressive tax',
                'tax evasion', 'tax avoidance', 'offshore banking', 'shell company',
                'money laundering', 'Ponzi scheme', 'pyramid scheme', 'insider trading',
                'whistleblower', 'white-collar crime', 'corporate espionage',
                'dark pool trading', 'front running', 'pump and dump scheme',
                'wash trading', 'naked short selling', 'dark web markets'

            ]
            semantic_scores = []
            for token in doc_general:
                if token.has_vector:
                    topic_similarity = max(
                        token.similarity(doc_general.vocab[topic])
                        for topic in financial_topics
                    )
                    semantic_scores.append(topic_similarity)
            
            semantic_score = np.mean(semantic_scores) if semantic_scores else 0.0
            logger.debug(f"Semantic score: {semantic_score:.3f}")
            
            # Combine scores with weights
            final_score = (
                0.35 * entity_score +
                0.30 * keyword_density +
                0.20 * semantic_score +
                0.15 * phrase_score
            )
            
            processing_time = time() - start_time
            logger.debug(f"Financial relevance assessment completed in {processing_time:.2f}s with score {final_score:.3f}")
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error assessing financial relevance: {e}")
            return 0.0

    def parallel_process_text(self, texts: List[str], func) -> List[float]:
        """Process texts in parallel using ProcessPoolExecutor"""
        total = len(texts)
        logger.info(f"Starting parallel processing of {total:,} texts...")
        start_time = time()
        
        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(func, text) for text in texts]
            for i, future in enumerate(tqdm(as_completed(futures), total=total, desc=f"Processing with {func.__name__}")):
                results.append(future.result())
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1:,}/{total:,} texts...")
        
        processing_time = time() - start_time
        logger.info(f"Parallel processing completed in {processing_time:.2f}s")
        return results

    def find_similar_texts(self, texts: List[str], threshold: float = 0.85) -> List[int]:
        """Find and return indices of similar texts using TF-IDF and cosine similarity"""
        try:
            # Convert texts to TF-IDF vectors
            vectors = self.vectorizer.fit_transform(texts)
            
            # Calculate pairwise similarities using sklearn's cosine_similarity
            duplicate_indices = set()
            batch_size = 1000  # Process in batches to save memory
            
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                current_vectors = vectors[i:batch_end]
                
                # Calculate similarities for current batch
                similarities = sklearn_cosine_similarity(current_vectors, vectors)
                
                # Find similar texts
                for j in range(similarities.shape[0]):
                    # Get indices of similar texts (excluding self-similarity)
                    similar = np.where(similarities[j] > threshold)[0]
                    similar = similar[similar > (i + j)]  # Only keep indices after current text
                    duplicate_indices.update(similar)
            
            return list(duplicate_indices)
            
        except Exception as e:
            logger.warning(f"Error finding similar texts: {e}")
            return []

    def load_or_compute(self, cache_key: str, compute_func, df=None, message="Processing"):
        """Generic function to handle loading from cache or computing"""
        cache_path = self.cache_paths[cache_key]
        
        if cache_path.exists():
            logger.info(f"Loading {cache_key} from cache...")
            return joblib.load(cache_path)
        
        logger.info(f"Computing {cache_key}...")
        result = compute_func(df) if df is not None else compute_func()
        
        logger.info(f"Caching {cache_key}...")
        joblib.dump(result, cache_path)
        return result

    def process_dataset(self):
        """Enhanced main processing pipeline with granular caching"""
        start_time = time()
        logger.info("Starting dataset processing pipeline...")
        
        try:
            # Load initial data
            df = self.load_or_compute('initial_load', self.load_data)
            initial_size = len(df)
            logger.info(f"Initial dataset size: {initial_size:,} records")

            # Score-based filtering
            def apply_score_filtering(data):
                score_cols = ['z_score', 'combined_score', 'comment_normalized_score']
                return data[np.all([data[col] > data[col].quantile(0.2) for col in score_cols], axis=0)]
            
            df = self.load_or_compute('score_filtered', apply_score_filtering, df)
            score_filtered_size = len(df)
            logger.info(f"After score filtering: {score_filtered_size:,} records")

            # Text cleaning
            def apply_text_cleaning(data):
                with ProcessPoolExecutor() as executor:
                    data['cleaned_body'] = list(tqdm(
                        executor.map(self.clean_text, data['body']),
                        total=len(data),
                        desc="Cleaning text"
                    ))
                return data[data['cleaned_body'].str.len() > 0]
            
            df = self.load_or_compute('cleaned_text', apply_text_cleaning, df)
            
            # Quality metrics calculation (separate caching for each metric)
            def compute_quality(data):
                return self.parallel_process_text(data['cleaned_body'], self.assess_text_quality)
            
            def compute_complexity(data):
                return self.parallel_process_text(data['cleaned_body'], self.calculate_text_complexity)
            
            def compute_financial_relevance(data):
                return self.parallel_process_text(data['cleaned_body'], self.assess_financial_relevance)
            
            df['body_quality'] = self.load_or_compute('text_quality', compute_quality, df)
            df['complexity'] = self.load_or_compute('complexity', compute_complexity, df)
            df['financial_relevance'] = self.load_or_compute('financial_relevance', compute_financial_relevance, df)

            # Apply all filters
            def apply_final_filtering(data):
                quality_filters = [
                    ('Quality threshold', data['body_quality'] > self.quality_threshold),
                    ('Complexity threshold', data['complexity'] > self.complexity_threshold),
                    ('Financial relevance', data['financial_relevance'] > self.min_financial_relevance),
                    ('Conversational style', data['cleaned_body'].apply(self.is_conversational)),
                    ('No profanity', ~data['cleaned_body'].apply(self.contains_profanity)),
                    ('Length constraints', data['cleaned_body'].str.len().between(50, 2000))
                ]
                
                # Log individual filter impacts
                for filter_name, condition in quality_filters:
                    passing = len(data[condition])
                    logger.info(f"{filter_name}: {passing:,} records pass ({(passing/len(data))*100:.1f}% pass rate)")
                
                filtered_data = data.copy()
                for _, condition in quality_filters:
                    filtered_data = filtered_data[condition]
                
                # Deduplication
                duplicate_indices = self.find_similar_texts(filtered_data['cleaned_body'].tolist())
                filtered_data = filtered_data.drop(index=filtered_data.index[duplicate_indices])
                
                return filtered_data
            
            df = self.load_or_compute('final_filtered', apply_final_filtering, df)
            
            # Convert to SFT format and save
            training_data = self.convert_to_sft_format(df)
            with open(self.output_file, 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Successfully saved {len(training_data):,} examples to {self.output_file}")
            logger.info(f"Total processing time: {(time() - start_time):.2f}s")
            
        except Exception as e:
            logger.error(f"Error in dataset processing: {e}")
            raise

if __name__ == "__main__":
    try:
        cleaner = DatasetCleaner(
            input_file='/home/zahemen/datasets/reddit-finance-250k/Data.jsonl',
            output_file='/home/zahemen/datasets/reddit-finance-250k/sft_cleaned_data.jsonl',
            conv_starters_file='/home/zahemen/datasets/reddit-finance-250k/conv_starter_pairs.txt',
            silent_warnings=True  # Set to True to silence warnings
        )
        cleaner.process_dataset()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)