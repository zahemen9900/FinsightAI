import json
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm import tqdm
from rich.logging import RichHandler
from rich.progress import track

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')

class ModelEvaluator:
    def __init__(
        self, 
        model_name: str = 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        test_data_path: str = "/home/zahemen/datasets/reddit-finance-250k/sft_format_data.jsonl",
        device: str = None,
        max_length: int = 512
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        # Load model and tokenizer
        logger.info(f"Loading model and tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Load test dataset
        logger.info(f"Loading test dataset from {test_data_path}")
        dataset = load_dataset('json', data_files=test_data_path)['train']
        self.test_data = dataset.train_test_split(test_size=0.2)['test']
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

    def generate_response(self, prompt: str) -> str:
        """Generate response from model for given prompt"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def compute_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores between prediction and reference"""
        scores = self.rouge_scorer.score(prediction, reference)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def compute_bleu_score(self, prediction: str, reference: str) -> float:
        """Compute BLEU score between prediction and reference"""
        prediction_tokens = nltk.word_tokenize(prediction.lower())
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        return sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=self.smooth)

    def evaluate(self, num_samples: int = None) -> Dict[str, float]:
        """Evaluate model on test dataset"""
        if num_samples:
            eval_data = self.test_data.select(range(min(num_samples, len(self.test_data))))
        else:
            eval_data = self.test_data

        rouge_scores = []
        bleu_scores = []
        
        logger.info(f"Evaluating model on {len(eval_data)} samples")
        for item in track(eval_data, description="Evaluating"):
            try:
                # Get prompt and reference
                prompt = item['messages'][0]['content']
                reference = item['messages'][1]['content']
                
                # Generate prediction
                prediction = self.generate_response(prompt)
                
                # Compute metrics
                rouge = self.compute_rouge_scores(prediction, reference)
                bleu = self.compute_bleu_score(prediction, reference)
                
                rouge_scores.append(rouge)
                bleu_scores.append(bleu)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate sample: {e}")
                continue
        
        # Calculate average scores
        avg_scores = {
            'rouge1': np.mean([s['rouge1'] for s in rouge_scores]),
            'rouge2': np.mean([s['rouge2'] for s in rouge_scores]),
            'rougeL': np.mean([s['rougeL'] for s in rouge_scores]),
            'bleu': np.mean(bleu_scores)
        }
        
        logger.info("Evaluation Results:")
        for metric, score in avg_scores.items():
            logger.info(f"{metric}: {score:.4f}")
            
        return avg_scores

    def save_results(self, results: Dict[str, float], output_path: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        os.makedirs('metrics', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of model to evaluate")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json", help="Path to save results")
    args = parser.parse_args()

    # Run evaluation
    evaluator = ModelEvaluator(args.model_name)
    results = evaluator.evaluate(num_samples=args.num_samples)
    evaluator.save_results(results, args.output_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
