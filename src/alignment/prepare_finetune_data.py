import json
import logging
import hashlib
from typing import Dict, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def convert_reddit_to_sft_format(reddit_post: Dict) -> Dict:
    """Convert a single Reddit post to SFT chat format"""
    try:
        # Create 64-char hash from the original ID
        prompt_id = hashlib.sha256(reddit_post["id"].encode()).hexdigest()
        
        return {
            "prompt": reddit_post["title"] + "\n\n" + reddit_post["selftext"],
            "messages": [
                {
                    "role": "user",
                    "content": reddit_post["title"] + "\n\n" + reddit_post["selftext"]
                },
                {
                    "role": "assistant", 
                    "content": reddit_post["body"]
                }
            ],
            "prompt_id": prompt_id  # Using the 64-char hash instead of original ID
        }
    except KeyError as e:
        logger.error(f"Missing required field in post {reddit_post.get('id', 'UNKNOWN')}: {e}")
        raise

def prepare_dataset():
    input_path = '/home/zahemen/datasets/reddit-finance-250k/filtered_data.jsonl'
    output_path = '/home/zahemen/datasets/reddit-finance-250k/sft_format_data.jsonl'

    # Ensure input file exists
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info("Starting dataset preparation...")
    
    # Load Reddit data
    reddit_data = []
    try:
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    reddit_data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON at line {line_num}")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return

    logger.info(f"Loaded {len(reddit_data)} posts from JSONL file")
    
    # Convert to SFT format
    sft_data = []
    for post in reddit_data:
        try:
            sft_data.append(convert_reddit_to_sft_format(post))
        except Exception as e:
            logger.warning(f"Failed to convert post {post.get('id', 'UNKNOWN')}: {e}")
    
    logger.info(f"Successfully converted {len(sft_data)} posts to SFT format")
    
    # Save converted data
    try:
        with open(output_path, 'w') as f:
            for item in sft_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved converted data to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving output file: {e}")

if __name__ == "__main__":
    try:
        prepare_dataset()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
