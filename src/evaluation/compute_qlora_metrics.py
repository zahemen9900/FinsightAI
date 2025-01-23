import json
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
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

class QLoRAEvaluator:
    def __init__(
        self, 
        base_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        adapter_path: str = "qlora_output",
        test_data_path: str = "/home/zahemen/datasets/reddit-finance-250k/sft_format_data.jsonl",
        device: str = None,
        max_length: int = 512
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
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
        
        # Load test dataset
        logger.info(f"Loading test dataset from {test_data_path}")
        dataset = load_dataset('json', data_files=test_data_path)['train']
        self.test_data = dataset.train_test_split(test_size=0.2)['test']
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

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

    def evaluate(self, num_samples: int = None) -> Dict[str, float]:
        """Evaluate model on test dataset"""
        if num_samples:
            eval_data = self.test_data.select(range(min(num_samples, len(self.test_data))))
        else:
            eval_data = self.test_data

        rouge_scores = []
        bleu_scores = []
        
        logger.info(f"Evaluating QLoRA model on {len(eval_data)} samples")
        for item in track(eval_data, description="Evaluating"):
            try:
                # Get prompt and reference from messages
                messages = item['messages']
                prompt = ""
                reference = ""
                
                # Find the last user-assistant pair
                for i in range(len(messages)-1):
                    if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                        prompt = messages[i]['content']
                        reference = messages[i+1]['content']
                
                if not prompt or not reference:
                    continue
                
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
        
        logger.info("QLoRA Model Evaluation Results:")
        for metric, score in avg_scores.items():
            logger.info(f"{metric}: {score:.4f}")
            
        return avg_scores

    def save_results(self, results: Dict[str, float], output_path: str = "qlora_evaluation_results.json"):
        """Save evaluation results to file"""
        os.makedirs('metrics', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="qlora_output")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="qlora_evaluation_results.json")
    args = parser.parse_args()

    # Run evaluation
    evaluator = QLoRAEvaluator(
        base_model=args.base_model,
        adapter_path=args.adapter_path
    )
    results = evaluator.evaluate(num_samples=args.num_samples)
    evaluator.save_results(results, args.output_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
