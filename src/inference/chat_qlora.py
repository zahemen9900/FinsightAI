import re
import sys
import torch
import logging
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
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
        adapter_path: str = "qlora_output",
        device: str = None,
        max_length: int = 512,
        num_beams: int = 1,
        stream: bool = True  # Add streaming parameter
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.num_beams = num_beams
        self.stream = stream
        
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
        self.conversation_history = []
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are FinSight, a professional financial advisor. "
                "Keep responses clear, focused, and concise. "
                # "Provide accurate guidance while maintaining transparency about being an AI."
            )
        }

        # Enhanced question type patterns
        self.question_patterns = {
            # Basic interactions (very concise)
            'greeting': (
                r'\b(hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening)|yo|sup)\b|^[^a-zA-Z]*$',
                36
            ),
            'farewell': (
                r'\b(bye|goodbye|thanks|thank you|exit|quit|stop)\b',
                24
            ),
            'acknowledgment': (
                r'^(ok|okay|sure|alright|i see|got it|understood)\b',
                24
            ),
            
            # Simple queries (concise)
            'basic_question': (
                r'^(what( is|\'s)|how|why|can you|could you|would you|do you|is|are)\b.{0,50}\?*$',
                 80
            ),
            'definition': (
                r'\b(what (is|are|does)|define|meaning of|definition)\b.{0,50}\?*$',
                80
            ),
            
            # Comparisons (medium length)
            'comparison': (
                r'\b(vs|versus|compare|difference|better|which|or|between|prefer)\b',
                96
            ),
            'pros_cons': (
                r'\b(advantages|disadvantages|benefits|drawbacks|pros|cons|upsides|downsides)\b',
                112
            ),
            
            # Explanations (detailed)
            'explanation': (
                r'\b(explain|describe|elaborate|tell me about|how does|in what way|why does)\b',
                128
            ),
            'process': (
                r'\b(how to|steps|process|procedure|method|approach|way to|guide)\b',
                144
            ),
            
            # Analysis (comprehensive)
            'analysis': (
                r'\b(analyze|evaluate|assess|review|examine|consider|thoughts on|opinion|strategy|plan)\b',
                192
            ),
            'recommendation': (
                r'\b(recommend|suggest|advise|should i|what would you|best way|optimal|ideal)\b',
                160
            ),
            
            # Lists and enumerations
            'list': (
                r'\b(list|what are|give me|show me|tips|steps|ways)\b.*\b(points|steps|ways|things|tips|examples|factors|reasons)\b',
                160
            ),
            
            # Financial specifics (detailed)
            'investment': (
                r'\b(invest|stock|bond|etf|fund|portfolio|diversify|asset|allocation)\b',
                176
            ),
            'risk_related': (
                r'\b(risk|safe|secure|volatile|stability|protect|hedge|insurance)\b',
                144
            ),
            'numbers_heavy': (
                r'\b(\d+%|\$\d+|ratio|rate|return|yield|profit|loss)\b',
                128
            ),
            
            # Edge cases
            'complex_query': (
                r'(.*?\b(and|or)\b.*?\?)|(\?.*?\band\b.*?\?)',  # Multiple questions
                200
            ),
            'scenario': (
                r'\b(imagine|suppose|what if|scenario|case|situation)\b',
                176
            ),
            'urgent': (
                r'\b(urgent|asap|emergency|quick|fast|help|crisis)\b',
                96  # Keep it focused for urgent queries
            ),
            'clarification': (
                r'\b(clarify|understand|mean|rephrase|explain again|still don\'t get)\b',
                80
            ),
        }

    def analyze_question(self, question: str) -> int:
        """Enhanced question analysis with better fallbacks"""
        question = question.lower().strip()
        
        # Default token length for unmatched patterns
        default_length = 96
        
        # Check for specific patterns
        matched_tokens = []
        for _, (pattern, tokens) in self.question_patterns.items():
            if re.search(pattern, question):
                matched_tokens.append(tokens)
        
        if matched_tokens:
            # If multiple patterns match, use the larger token length
            return max(matched_tokens)
        
        # Enhanced fallback logic
        words = len(question.split())
        
        # Super short inputs
        if words <= 3:
            return 48
        
        # Very long or complex queries
        if words >= 20:
            return 200
        
        # Questions with multiple parts
        if question.count('?') > 1:
            return 176
        
        # Questions with numbers might need more detail
        if re.search(r'\d+', question):
            return 144
        
        # Questions with specific financial terms
        financial_terms = r'\b(stock|bond|invest|market|fund|portfolio|risk|return|dividend)\b'
        if re.search(financial_terms, question):
            return 144
        
        # Default based on question length
        if words < 8:
            return 64
        elif words < 15:
            return 96
        else:
            return 128

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.3,  # Lower temperature for more focused responses
        top_p: float = 0.9,
        analyze_response = False
    ) -> str:
        """Generate a response using proper chat template"""
        # Determine appropriate response length
        max_new_tokens = 128 if not analyze_response else self.analyze_question(prompt)

        # Keep only last two exchanges before adding new message
        if len(self.conversation_history) > 4:  # 4 messages = 2 exchanges (user + assistant)
            self.conversation_history = self.conversation_history[-4:]

        # Format messages with limited history
        messages = [self.system_prompt]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template using tokenizer's built-in method
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

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

        # Set up streamer if streaming is enabled
        timeout = max(20.0, max_new_tokens // 5.0)
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=timeout
        ) if self.stream else None

        # Add temperature control for more focused responses
        temperature = min(temperature, 0.7 if max_new_tokens > 128 else 0.4)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.5,  # Increased
                no_repeat_ngram_size=5,  # Increased
                num_beams=self.num_beams,
                streamer=streamer
            )
            
        # Update conversation history consistently for both streaming and non-streaming
        self.conversation_history.append({"role": "user", "content": prompt})
        
        if not self.stream:
            outputs = outputs.cpu()
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split(prompt)[-1].strip()
            if not response:
                response = "I apologize, but I couldn't generate a proper response."
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Ensure only last two exchanges are kept
            if len(self.conversation_history) > 4:
                self.conversation_history = self.conversation_history[-4:]
            
            return response
        
        return ""  # Empty string for streaming mode

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="qlora_output/checkpoint-1000")
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
                
            # Print assistant prefix for streaming
            console.print("\n[bold blue]Assistant:[/bold blue] ", style="bold", end="")
            
            # Generate response (will stream automatically)
            advisor.generate_response(
                user_input,
                temperature=args.temperature
            )
            
            console.print()  # Add newline after streamed response
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            continue
    
    console.print("\n[bold cyan]Thanks for chatting! Goodbye![/bold cyan]")

if __name__ == "__main__":
    main()
