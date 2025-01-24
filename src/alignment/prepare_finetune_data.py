import json
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
from torch import cosine_similarity
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')

# Download required NLTK data
nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet'])
from nltk.corpus import wordnet

# Load spaCy model for text quality analysis
nlp = spacy.load('en_core_web_sm')
# Load financial domain specific model if available
try:
    nlp_financial = spacy.load('en_core_financial_web_sm')
except:
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
        self.min_financial_relevance = 0.6  # Lowered from 0.7
        self.max_similarity_threshold = 0.85  # For deduplication
        self.complexity_threshold = 0.3  # Lowered from 0.4
        self.min_words = 5
        self.max_words = 150  # Limit response length
        
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
            'inflation', 'economy', 'security', 'hedge', 'option', 'future'
        ])

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
        """Enhanced text cleaning"""
        if not isinstance(text, str):
            return ""
            
        # Remove chat artifacts
        text = re.sub(r'###\s*(Human|Assistant):', '', text)
        text = re.sub(r'(AI|User):', '', text)
        
        # Remove multiple line breaks and extra spaces
        text = re.sub(r'\n+', ' ', text)
        text = ' '.join(text.split())
        
        # Ensure proper sentence structure
        text = '. '.join(s.strip().capitalize() for s in text.split('.') if s.strip())
        if text and text[-1] not in '.!?':
            text += '.'
            
        return text.strip()

    def is_quality_response(self, text: str) -> bool:
        """Additional quality checks for responses"""
        if not text:
            return False
            
        words = text.split()
        if len(words) < self.min_words or len(words) > self.max_words:
            return False
            
        # Check for coherent sentence structure
        sentences = sent_tokenize(text)
        if len(sentences) < 1:
            return False
            
        # Check for repeated phrases
        text_lower = text.lower()
        for phrase in ["is there anything", "yes thank you", "please specify"]:
            if text_lower.count(phrase) > 1:
                return False
                
        return True

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
            r'\b(hi|hello|hey|thanks|thank you)\b',
            r'\?',
            r'\b(you|your|yours)\b',
            r'\b(I|me|my|mine)\b'
        ]
        
        score = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in indicators)
        return score >= 2

    def filter_lengthy_prompts(self, text: str) -> bool:
        """Filter out very lengthy prompts."""
        return len(text) <= self.max_prompt_length

    def convert_to_sft_format(self, row: pd.Series) -> Dict:
        """Convert a single record to chat format for fine-tuning"""
        try:
            # Generate unique conversation ID
            conversation_id = hashlib.sha256(row["id"].encode()).hexdigest()
            
            # Clean texts
            cleaned_body = self.clean_text(row["body"])
            cleaned_title = self.clean_text(row["title"])
            cleaned_selftext = self.clean_text(row["selftext"] if pd.notna(row["selftext"]) else "")
            
            # Skip if response quality check fails
            if not self.is_quality_response(cleaned_body):
                raise ValueError("Response quality check failed")
            
            # Format the question/prompt
            question = cleaned_title
            if cleaned_selftext:
                question += f"\n\n{cleaned_selftext}"
            
            # Create the messages list directly (no prompt field)
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
            
            # Only return messages and metadata
            return {
                "messages": messages,
                "metadata": {
                    "source": f"reddit: {row['subreddit']}",
                    "conversation_id": conversation_id
                }
            }
            
        except Exception as e:
            if not self.silent_warnings:
                logger.warning(f"Failed to convert row {row.get('id', 'UNKNOWN')}: {e}")
            raise

    @lru_cache(maxsize=10000)
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
        """Assess how relevant the text is to financial topics"""
        if not isinstance(text, str):
            return 0.0
            
        try:
            # Use spaCy's financial model for entity recognition
            doc = nlp_financial(text)
            
            # Count financial entities
            financial_entities = sum(1 for ent in doc.ents if ent.label_ in {
                'ORG', 'MONEY', 'PERCENT', 'QUANTITY'
            })
            
            # Count financial keywords
            words = set(text.lower().split())
            keyword_matches = len(words.intersection(self.financial_keywords))
            
            # Calculate combined score
            relevance_score = (
                0.6 * (financial_entities / max(len(doc.ents), 1)) +
                0.4 * (keyword_matches / max(len(words), 1))
            )
            
            return min(relevance_score, 1.0)
        except Exception as e:
            logger.warning(f"Error assessing financial relevance: {e}")
            return 0.0

    def parallel_process_text(self, texts: List[str], func) -> List[float]:
        """Process texts in parallel using ProcessPoolExecutor"""
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(func, text) for text in texts]
            return [future.result() for future in as_completed(futures)]

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

    def process_dataset(self):
        """Enhanced main processing pipeline with safe filtering"""
        try:
            # Try to load from cache first
            cache_file = self.cache_dir / "processed_data.joblib"
            if cache_file.exists():
                df = joblib.load(cache_file)
                logger.info("Loaded processed data from cache")
            else:
                df = self.load_data()
                initial_size = len(df)
                logger.info(f"Initial dataset size: {initial_size:,} records")
                
                # Initial filtering based on scores (Group 1: Score-based filtering)
                # logger.info("\n=== Score-based Filtering ===")
                score_cols = ['z_score', 'combined_score', 'comment_normalized_score']
                df = df[np.all([df[col] > df[col].quantile(0.2) for col in score_cols], axis=0)]
                score_filtered_size = len(df)
                logger.info(f"After score filtering: {score_filtered_size:,} records ({(score_filtered_size/initial_size)*100:.1f}% retained)")
                
                # Text cleaning and quality metrics (Group 2: Text processing)
                # logger.info("\n=== Text Processing and Quality Assessment ===")
                with ProcessPoolExecutor() as executor:
                    df['cleaned_body'] = list(executor.map(self.clean_text, df['body']))
                df = df[df['cleaned_body'].str.len() > 0]  # Remove empty texts
                text_cleaned_size = len(df)
                logger.info(f"After text cleaning: {text_cleaned_size:,} records ({(text_cleaned_size/score_filtered_size)*100:.1f}% retained)")
                
                # Calculate all quality metrics in parallel (Group 3: Quality metrics)
                # logger.info("\n=== Quality Metrics Calculation ===")
                df['body_quality'] = self.parallel_process_text(df['cleaned_body'], self.assess_text_quality)
                df['complexity'] = self.parallel_process_text(df['cleaned_body'], self.calculate_text_complexity)
                df['financial_relevance'] = self.parallel_process_text(df['cleaned_body'], self.assess_financial_relevance)
                
                # Apply all quality filters together (Group 4: Quality filtering)
                # logger.info("\n=== Quality Filtering ===")
                quality_filters = [
                    ('Quality threshold', df['body_quality'] > self.quality_threshold),
                    ('Complexity threshold', df['complexity'] > self.complexity_threshold),
                    ('Financial relevance', df['financial_relevance'] > self.min_financial_relevance),
                    ('Conversational style', df['cleaned_body'].apply(self.is_conversational)),
                    ('No profanity', ~df['cleaned_body'].apply(self.contains_profanity)),
                    ('Length constraints', df['cleaned_body'].str.len().between(50, 2000))
                ]
                
                # Log individual filter impacts
                for filter_name, condition in quality_filters:
                    passing = len(df[condition])
                    logger.info(f"{filter_name}: {passing:,} records pass ({(passing/len(df))*100:.1f}% pass rate)")
                
                # Apply all filters
                for _, condition in quality_filters:
                    df = df[condition]
                quality_filtered_size = len(df)
                logger.info(f"After all quality filters: {quality_filtered_size:,} records ({(quality_filtered_size/text_cleaned_size)*100:.1f}% retained)")
                
                # Deduplication (Group 5: Similarity filtering)
                logger.info("\n=== Deduplication ===")
                duplicate_indices = self.find_similar_texts(df['cleaned_body'].tolist())
                df = df.drop(index=df.index[duplicate_indices])
                final_size = len(df)
                logger.info(f"After deduplication: {final_size:,} records ({(final_size/quality_filtered_size)*100:.1f}% retained)")
                
                # Overall statistics
                logger.info("\n=== Final Statistics ===")
                logger.info(f"Initial dataset size: {initial_size:,}")
                logger.info(f"Final dataset size: {final_size:,}")
                logger.info(f"Overall retention rate: {(final_size/initial_size)*100:.1f}%")
                
                # Cache the processed DataFrame
                joblib.dump(df, cache_file)
                logger.info("\nProcessed data cached successfully")
            
            # Convert to SFT format
            logger.info("\n=== Converting to SFT Format ===")
            training_data = []
            for _, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    formatted_data = self.convert_to_sft_format(row)
                    training_data.append(formatted_data)
                except Exception as e:
                    if not self.silent_warnings:
                        logger.warning(f"Failed to convert row {row.get('id', 'UNKNOWN')}: {e}")
            
            # Save final output
            with open(self.output_file, 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Successfully saved {len(training_data):,} examples to {self.output_file}")
            
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