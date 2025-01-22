import json
import pandas as pd
import re
import numpy as np
import logging
import hashlib
from typing import Dict, List
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')



# Download required NLTK data
nltk.download('punkt')
# Load spaCy model for text quality analysis
nlp = spacy.load('en_core_web_sm')

class DatasetCleaner:
    def __init__(self, input_file: str, output_file: str, conv_starters_file: str, silent_warnings: bool = True):
        self.input_file = input_file
        self.output_file = output_file
        self.conv_starters_file = conv_starters_file
        self.silent_warnings = silent_warnings
        self.quality_threshold = 0.7
        self.max_prompt_length = 1200  # Set a maximum length for prompts
        
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
        """Clean text by removing Reddit-specific formatting and noise."""
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit markdown and formatting
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'[━┃┏┓┗┛│└┘╭╮╯╰▀▄█▌▐░▒▓]', '', text)
        text = re.sub(r'\*{1,3}', '', text)
        text = re.sub(r'~{2}.*?~{2}', '', text)
        text = re.sub(r'_{1,2}.*?_{1,2}', '', text)
        
        # Remove edit notices and award speech
        text = re.sub(r'edit\s*\d*\s*:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'update\s*\d*\s*:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'thanks? for (?:the)? (?:gold|silver|platinum|award).*', '', text, flags=re.IGNORECASE)
        
        return ' '.join(text.split()).strip()

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
        """Convert a single record to SFT chat format with proper ID hashing."""
        try:
            # Create 64-char hash from the original ID
            prompt_id = hashlib.sha256(row["id"].encode()).hexdigest()
            
            cleaned_title = self.clean_text(row["title"])
            cleaned_selftext = self.clean_text(row["selftext"] if pd.notna(row["selftext"]) else "")
            cleaned_body = self.clean_text(row["body"])
            
            prompt = f"{cleaned_title}\n\n{cleaned_selftext}".strip()
            
            # Check if the prompt is too lengthy
            if not self.filter_lengthy_prompts(prompt):
                raise ValueError("Prompt is too lengthy")
            
            # Create multi-turn conversation starters for selected prompts
            if np.random.rand() < 0.5:  # 50% chance to add multi-turn starters
                starter = np.random.choice(self.conv_starters)
                messages = [
                    {
                        "role": "system",
                        "content": "You are FinSight, an AI financial advisor skilled in multiple domains. Provide helpful, accurate financial guidance while being clear that you're not a licensed professional."
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
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": cleaned_body
                    }
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": "You are FinSight, an AI financial advisor skilled in multiple domains, not limited to finance. Provide helpful, accurate financial guidance while being clear that you're not a licensed professional."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": cleaned_body
                    }
                ]
            
            return {
                "prompt": prompt,
                "messages": messages,
                "prompt_id": prompt_id
            }
        except Exception as e:
            if not self.silent_warnings:
                logger.warning(f"Failed to convert row {row.get('id', 'UNKNOWN')}: {e}")
            raise

    def process_dataset(self):
        """Main processing pipeline."""
        try:
            # Load data
            df = self.load_data()
            logger.info(f"Initial dataset size: {len(df)}")
            
            # Apply score-based filtering
            for col in ['z_score', 'combined_score', 'comment_normalized_score']:
                threshold = df[col].quantile(0.5)
                df = df[df[col] > threshold]
            logger.info(f"After score filtering: {len(df)}")
            
            # Clean and assess text quality
            df['cleaned_body'] = df['body'].apply(self.clean_text)
            df['body_quality'] = df['cleaned_body'].apply(self.assess_text_quality)
            
            # Apply quality filters
            df = df[
                (df['body_quality'] > self.quality_threshold) &
                (df['cleaned_body'].apply(self.is_conversational)) &
                (~df['cleaned_body'].apply(self.contains_profanity)) &
                (df['cleaned_body'].str.len() > 50) &
                (df['cleaned_body'].str.len() < 2000)
            ]
            logger.info(f"After quality filtering: {len(df)}")
            
            # Convert to SFT format
            training_data = []
            for _, row in df.iterrows():
                try:
                    formatted_data = self.convert_to_sft_format(row)
                    training_data.append(formatted_data)
                except Exception as e:
                    if not self.silent_warnings:
                        logger.warning(f"Failed to convert row {row.get('id', 'UNKNOWN')}: {e}")
            
            # Save processed data
            with open(self.output_file, 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Successfully saved {len(training_data)} examples to {self.output_file}")
            
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