import sys
import torch
import logging
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel 
from rich.logging import RichHandler
from rich.console import Console
from rich.markdown import Markdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(show_time=False)],
)
logger = logging.getLogger('rich')
console = Console()

class FinanceAdvisor:
    def __init__(
        self,
        base_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        adapter_path: str = "qlora_output/checkpoint-300",
        device: str = None,
        max_length: int = 512,
        num_beams: int = 1
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.num_beams = num_beams
        
        # Load tokenizer with specific settings and move to device
        logger.info(f"Loading tokenizer from {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            padding_side="left",
            trust_remote_code=True
        )
        
        # Load base model and explicitly move to device
        logger.info(f"Loading base model from {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to(self.device)  # Explicitly move to device
        
        # Load LoRA adapter
        if Path(adapter_path).exists():
            logger.info(f"Loading LoRA adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(
                base,
                adapter_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)  # Explicitly move to device
        else:
            logger.warning(f"No LoRA adapter found at {adapter_path}, using base model only")
            self.model = base

        self.model.eval()
        self.conversation_history: List[Dict] = []
        self.system_prompt = (
            "You are a financial advisor chatbot trained to provide helpful advice about finance and investing. "
            "Please keep your responses concise, relevant to the user's questions, and avoid adding unrelated information."
        )

    def format_prompt(self, message: str) -> str:
        """Format a message into the expected chat format"""
        # Add the new message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Keep only last 3 messages for context window
        recent_history = self.conversation_history[-3:]
        
        # Format conversation history with specific prefix
        formatted = f"{self.system_prompt}\n\n"
        for msg in recent_history:
            if msg["role"] == "user":
                formatted += f"### Human: {msg['content']}\n"
            else:
                formatted += f"### Assistant: {msg['content']}\n"
                
        return formatted.strip() + "\n### Assistant:"

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_new_tokens: int = 150,  # Reduce max_new_tokens for more concise responses
    ) -> str:
        """Generate a response for the given prompt"""
        formatted_prompt = self.format_prompt(prompt)
        
        # Prepare inputs and explicitly move to device
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move all tensors to device

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                num_beams=self.num_beams,  # Add num_beams parameter
                early_stopping=False  # Disable early stopping since num_beams=1
            )
            
            # Move outputs to CPU for decoding
            outputs = outputs.cpu()
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response
        try:
            response = response.split("### Assistant:")[-1]
            if "### Human:" in response:
                response = response.split("### Human:")[0]
            response = response.strip()
        except:
            response = "I apologize, but I couldn't generate a proper response."
        
        # Add response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="qlora_output/checkpoint-300")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    
    advisor = FinanceAdvisor(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        max_length=args.max_length
    )
    
    # Print welcome message
    console.print("\n[bold cyan]Welcome to FinsightAI - Your Financial Advisory Assistant![/bold cyan]")
    console.print("[dim]Type 'quit' to exit, 'clear' to start fresh[/dim]\n")
    
    while True:
        try:
            # Get user input
            user_input = console.input("[bold green]You:[/bold green] ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                advisor.conversation_history = []
                console.print("[dim]Conversation history cleared[/dim]")
                continue
                
            # Generate and print response
            response = advisor.generate_response(
                user_input,
                temperature=args.temperature
            )
            
            console.print("\n[bold blue]Assistant:[/bold blue]", style="bold")
            console.print(Markdown(response))
            console.print()  # Empty line for readability
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            continue
    
    console.print("\n[bold cyan]Thanks for chatting! Goodbye![/bold cyan]")

if __name__ == "__main__":
    main()
