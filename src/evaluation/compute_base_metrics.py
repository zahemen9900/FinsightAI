import argparse
import json
import os
import re
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
import uuid
import pandas as pd

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
        
        # Add financial terms set (expanded version from QLora script)
        self.financial_terms = set([
            'investment', 'stock', 'bond', 'market', 'fund', 'dividend',
            'portfolio', 'asset', 'equity', 'risk', 'return', 'trading',
            'broker', 'hedge', 'option', 'futures', 'commodity', 'leverage',
            'volatility', 'yield', 'capital', 'margin', 'inflation', 'liquidity',
            'diversification', 'securities', 'share', 'etf', 'reit', 'interest',
            'appreciation', 'depreciation', 'bear', 'bull', 'derivative', 'forex',
            # ... copy rest of financial terms from QLora script ...
        ])
        
        # Add metrics directory path
        self.metrics_dir = Path("metrics")
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Add run identifier
        self.run_id = str(uuid.uuid4())[:8]
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add metrics tracking
        self.all_sample_metrics = []

        # Add system prompt
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are FinSight, a professional financial advisor. "
                "Keep responses clear, focused, and concise."
            )
        }

    def generate_response(self, prompt: str) -> str:
        """Generate response using proper chat template formatting"""
        # Format messages properly with system prompt
        messages = [
            self.system_prompt,
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template using tokenizer's built-in method
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare inputs with proper handling
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.5,
                no_repeat_ngram_size=5,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode and clean up response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split(prompt)[-1].strip()
        
        # Ensure proper formatting
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        if response and response[-1] not in '.!?':
            response += '.'
            
        return response

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

    def analyze_response(self, response: str) -> Dict[str, float]:
        """Analyze response quality based on multiple metrics"""
        metrics = {
            'length': len(response.split()),
            'financial_terms': len([w for w in response.lower().split() if w in self.financial_terms]),
            'has_numbers': bool(re.search(r'\d', response)),
            'sentence_count': len(nltk.sent_tokenize(response)),
            'avg_word_length': np.mean([len(w) for w in response.split()]),
            'capitalization': response[0].isupper() if response else False,
            'ends_properly': response[-1] in '.!?' if response else False,
        }
        return metrics

    def evaluate_single_response(self, prompt: str, reference: str) -> Dict[str, float]:
        """Evaluate a single response with additional metrics"""
        # Generate response first
        prediction = self.generate_response(prompt)
        
        # Calculate existing metrics
        rouge_scores = self.compute_rouge_scores(prediction, reference)
        bleu_score = self.compute_bleu_score(prediction, reference)
        
        # Add new response metrics
        response_metrics = self.analyze_response(prediction)
        
        # Combine all metrics
        metrics = {
            **rouge_scores,
            'bleu': bleu_score,
            **{f'response_{k}': v for k, v in response_metrics.items()}
        }
        
        # Track individual sample metrics with full information
        self.all_sample_metrics.append({
            'prompt': prompt,
            'reference': reference,
            'prediction': prediction,
            **metrics
        })
        
        return metrics

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
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (str, int, float)):
            return obj
        elif obj is None:
            return None
        else:
            return str(obj)

    def save_results(self, results: Dict[str, Dict[str, Dict[str, float]]], output_path: str = None):
        """Enhanced save_results to match QLoRA format"""
        if output_path is None:
            output_path = self.metrics_dir / f"base_eval_{self.run_timestamp}_{self.run_id}"
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results in same format as QLoRA
        detailed_results = {
            "run_metadata": {
                "run_id": self.run_id,
                "timestamp": self.run_timestamp,
                "model_type": "base",
                "model_name": self.model.config._name_or_path,
                "device": self.device,
                "torch_dtype": str(self.dtype),
            },
            "metrics_summary": results,
            "datasets_info": {
                name: {"size": len(dataset)} 
                for name, dataset in self.datasets.items()
            }
        }

        # Convert results to serializable format before saving
        detailed_results = self.convert_to_serializable(detailed_results)

        # Save main results JSON
        with open(output_path / "results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # Convert DataFrame values to native types before saving
        df_samples = pd.DataFrame(self.convert_to_serializable(self.all_sample_metrics))
        df_samples.to_csv(output_path / "sample_metrics.csv", index=False)

        aggregated_metrics = {}
        for dataset_name, metrics in results.items():
            if dataset_name != 'overall':
                dataset_df = pd.DataFrame([{
                    'dataset': dataset_name,
                    'metric': metric,
                    'mean': float(stats['mean']),  # Explicit conversion
                    'std': float(stats['std']),    # Explicit conversion
                    'min': float(stats['min']),    # Explicit conversion
                    'max': float(stats['max'])     # Explicit conversion
                } for metric, stats in metrics.items()])
                aggregated_metrics[dataset_name] = dataset_df

        df_all = pd.concat(aggregated_metrics.values(), ignore_index=True)
        df_all.to_csv(output_path / "aggregated_metrics.csv", index=False)

        logger.info(f"Results saved to {output_path}")
        return output_path

def main():
    # Parse arguments
    os.makedirs('metrics', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="metrics/base_model_evaluation_results")
    args = parser.parse_args()

    # Define datasets to evaluate (same as in compute_qlora_metrics.py)
    dataset_paths = [
        {
            "path": "/home/zahemen/datasets/sft_datasets/intro_conversations.jsonl",
            "name": "finsight_intro",
        },
        {
            "path": "/home/zahemen/datasets/sft_datasets/reddit_finance_conversations.jsonl",
            "name": "reddit_finance",
        },
        {
            "path": "/home/zahemen/datasets/sft_datasets/company_conversations.jsonl",
            "name": "company_specific_qa",
        },
        {
            "path": "/home/zahemen/datasets/sft_datasets/financial_definitions_dataset.jsonl",
            "name": "financial_definitions",
        },
        {
            "path": "/home/zahemen/datasets/sft_datasets/finance_conversations.jsonl",
            "name": "finance_conversations",
        },
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
