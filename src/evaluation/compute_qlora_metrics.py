from datetime import datetime  # Change this line
import json
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm import tqdm
from rich.logging import RichHandler
from rich.progress import track
import re
import argparse


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

class QLoRAEvaluator:
    def __init__(
        self, 
        base_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        adapter_path: str = "qlora_output",
        dataset_paths: List[Dict[str, str]] = None,  # New parameter for multiple datasets
        device: str = None,
        max_length: int = 512
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Load base model and LoRA adapter
        logger.info(f"Loading base model from {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Load LoRA adapter if available
        if Path(adapter_path).exists():
            logger.info(f"Loading LoRA adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(
                base,
                adapter_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
        else:
            logger.warning(f"No LoRA adapter found at {adapter_path}, using base model only")
            self.model = base.to(self.device)
            
        self.model.eval()
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
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

        # Add patterns for response analysis
        self.financial_terms = set([
            'investment', 'stock', 'bond', 'market', 'fund', 'dividend',
            'portfolio', 'asset', 'equity', 'risk', 'return', 'trading',
            'broker', 'hedge', 'option', 'futures', 'commodity', 'leverage',
            'volatility', 'yield', 'capital', 'margin', 'inflation', 'liquidity',
            'diversification', 'securities', 'share', 'etf', 'reit', 'interest',
            'appreciation', 'depreciation', 'bear', 'bull', 'derivative', 'forex',
            'valuation', 'arbitrage', 'balance', 'bankruptcy', 'cash', 'credit',
            'debt', 'deposit', 'expense', 'growth', 'income', 'liability',
            'loan', 'loss', 'profit', 'revenue', 'sector', 'spread', 'tax',
            'wealth', 'index', 'nasdaq', 'dow', 'sp500', 'mortgage', 'mutual',
            'penny', 'stake', 'warrant', 'asset', 'beta', 'bid', 'ask',
            'blockchain', 'crypto', 'bitcoin', 'ethereum', 'correlation',
            'dividend', 'earnings', 'fiduciary', 'hedge', 'ipo', 'leverage',
            'margin', 'portfolio', 'quant', 'rally', 'recession', 'security',
            'stock', 'trade', 'volume', 'yield'
        ])

    def generate_response(self, prompt: str) -> str:
        """Generate response from model for given prompt"""
        # Format prompt with chat template
        formatted_prompt = f"### Human: {prompt}\n### Assistant:"
        
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response
        try:
            response = response.split("### Assistant:")[-1]
            if "### Human:" in response:
                response = response.split("### Human:")[0]
            response = response.strip()
        except:
            response = "I apologize, but I couldn't generate a proper response."
            
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
        """Evaluate a single response with comprehensive metrics"""
        # Format messages properly with system prompt and chat history
        messages = [
            {
                "role": "system",
                "content": (
                    "You are FinSight, a professional financial advisor. "
                    "Keep responses clear, focused, and concise."
                )
            }
        ]

        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
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

        with torch.inference_mode(), torch.amp.autocast(enabled=True, dtype=self.dtype, device_type=self.device, cache_enabled=True):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.3,  # Lower temperature for evaluation
                top_p=0.9,
                repetition_penalty=1.5,
                no_repeat_ngram_size=5,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )

        # Decode and clean up response
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = prediction.split(prompt)[-1].strip()
        
        # Ensure proper formatting
        if prediction and prediction[0].islower():
            prediction = prediction[0].upper() + prediction[1:]
        
        if prediction and prediction[-1] not in '.!?':
            prediction += '.'

        # Calculate metrics
        rouge_scores = self.compute_rouge_scores(prediction, reference)
        bleu_score = self.compute_bleu_score(prediction, reference)
        response_metrics = self.analyze_response(prediction)

        return {
            **rouge_scores,
            'bleu': bleu_score,
            **{f'response_{k}': v for k, v in response_metrics.items()}
        }

    def evaluate_dataset(self, dataset, name: str, num_samples: int = 100) -> Dict[str, Dict[str, float]]:
        """Evaluate model on a specific dataset"""
        if num_samples:
            eval_data = dataset.select(range(min(num_samples, len(dataset))))
        else:
            eval_data = dataset

        all_metrics = []
        
        logger.info(f"\nEvaluating QLoRA model on {len(eval_data)} samples from {name}")
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
        logger.info("\n=== QLoRA Model Evaluation Results ===")
        
        for dataset_name, scores in all_results.items():
            logger.info(f"\n--- {dataset_name.upper()} ---")
            for metric, stats in scores.items():
                if not metric.endswith(('_std', '_min', '_max')):
                    logger.info(f"\n{metric}:")
                    logger.info(f"  Mean: {stats['mean']:.4f}")
                    logger.info(f"  Std:  {stats['std']:.4f}")
                    logger.info(f"  Min:  {stats['min']:.4f}")
                    logger.info(f"  Max:  {stats['max']:.4f}")

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

    def save_results(self, results: Dict[str, Dict[str, Dict[str, float]]], output_path: str = "metrics/qlora_evaluation_results.json"):
        """Save evaluation results with additional metadata"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert all numpy types to native Python types
        serializable_results = self.convert_to_serializable(results)
        
        # Add metadata to results
        final_results = {
            "metrics_by_dataset": serializable_results,
            "metadata": {
                "evaluation_time": datetime.now().isoformat(),  # Fixed datetime usage
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
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="qlora_output")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="metrics/qlora_evaluation_results.json")
    args = parser.parse_args()

    # Define datasets to evaluate
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
    evaluator = QLoRAEvaluator(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        dataset_paths=dataset_paths
    )
    results = evaluator.evaluate(num_samples=args.num_samples)
    evaluator.save_results(results, args.output_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
