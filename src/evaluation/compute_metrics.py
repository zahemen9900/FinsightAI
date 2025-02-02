import argparse
import json
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Any, List, Dict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm import tqdm
from rich.logging import RichHandler
from rich.progress import track
from datetime import datetime

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
        dataset_paths: List[Dict[str, str]] = None,  # New parameter for multiple datasets
        device: str = None,
        max_length: int = 512
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Load model and tokenizer
        logger.info(f"Loading model and tokenizer from {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load test datasets
        self.datasets = {}
        if dataset_paths:
            for dataset_info in dataset_paths:
                path = dataset_info['path']
                name = dataset_info['name']
                logger.info(f"Loading dataset from {path}")
                try:
                    dataset = load_dataset('json', data_files=path)['train']
                    self.datasets[name] = dataset.train_test_split(test_size=0.2)['test']
                except Exception as e:
                    logger.warning(f"Failed to load dataset {name}: {e}")
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1
        
        # Add financial terms set
        self.financial_terms = set([
            'investment', 'stock', 'bond', 'market', 'fund', 'dividend',
            # ...rest of financial terms...
        ])

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

    def evaluate_single_response(self, prompt: str, reference: str) -> Dict[str, float]:
        """Evaluate a single response"""
        prediction = self.generate_response(prompt)
        rouge = self.compute_rouge_scores(prediction, reference)
        bleu = self.compute_bleu_score(prediction, reference)
        return {**rouge, 'bleu': bleu}

    def evaluate_dataset(self, dataset, name: str, num_samples: int = 100) -> Dict[str, Dict[str, float]]:
        """Evaluate model on a specific dataset"""
        if num_samples:
            eval_data = dataset.select(range(min(num_samples, len(dataset))))
        else:
            eval_data = dataset

        all_metrics = []
        
        logger.info(f"\nEvaluating base model on {len(eval_data)} samples from {name}")
        for item in track(eval_data, description=f"Evaluating {name}"):
            try:
                messages = item['messages']
                last_exchange = None
                
                for i in range(len(messages)-1):
                    if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                        last_exchange = (messages[i]['content'], messages[i+1]['content'])
                
                if not last_exchange:
                    continue
                
                prompt, reference = last_exchange
                metrics = self.evaluate_single_response(prompt, reference)
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate sample from {name}: {e}")
                continue

        # Calculate statistics for this dataset
        dataset_scores = {}
        for metric in all_metrics[0].keys():
            values = [m[metric] for m in all_metrics if metric in m]
            dataset_scores[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return dataset_scores

    def evaluate(self, num_samples: int = 100) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate model on all datasets"""
        all_results = {}
        
        # Evaluate each dataset separately
        for name, dataset in self.datasets.items():
            dataset_scores = self.evaluate_dataset(dataset, name, num_samples)
            all_results[name] = dataset_scores
        
        # Calculate overall averages
        overall_averages = {}
        for metric in next(iter(all_results.values())).keys():
            means = [scores[metric]['mean'] for scores in all_results.values()]
            overall_averages[metric] = {
                'mean': np.mean(means),
                'std': np.std(means),
                'min': np.min(means),
                'max': np.max(means)
            }
        
        all_results['overall'] = overall_averages
        
        # Log detailed results
        logger.info("\n=== Base Model Evaluation Results ===")
        # ...logging code...

        return all_results

    def convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        return obj

    def save_results(self, results: Dict[str, Dict[str, Dict[str, float]]], output_path: str = "metrics/base_model_evaluation_results.json"):
        """Save evaluation results with additional metadata"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert all numpy types to native Python types
        serializable_results = self.convert_to_serializable(results)
        
        # Add metadata to results
        final_results = {
            "metrics_by_dataset": serializable_results,
            "metadata": {
                "evaluation_time": datetime.now().isoformat(),
                "model_name": self.model.config._name_or_path,
                "datasets": list(self.datasets.keys()),
                "device": self.device,
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

def main():
    # Parse arguments
    os.makedirs('metrics', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="metrics/base_model_evaluation_results.json")
    args = parser.parse_args()

    # Define datasets to evaluate (same as in compute_qlora_metrics.py)
    dataset_paths = [
        {
            "path": "/home/zahemen/datasets/reddit-finance-250k/sft_cleaned_data.jsonl",
            "name": "reddit_finance"
        },
        {
            "path": "/home/zahemen/datasets/finance_qa_conversations.jsonl",
            "name": "finance_qa"
        },
        {
            "path": "/home/zahemen/datasets/intro_conversations.jsonl",
            "name": "intro_conversations"
        }
    ]

    # Run evaluation
    evaluator = ModelEvaluator(
        model_name=args.model_name,
        dataset_paths=dataset_paths
    )
    results = evaluator.evaluate(num_samples=args.num_samples)
    evaluator.save_results(results, args.output_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
