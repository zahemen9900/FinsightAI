import re
import sys
from tkinter import FALSE
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
        stream: bool = True,
        should_analyze_question: bool = True  # Renamed from analyze_question
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.num_beams = num_beams
        self.stream = stream
        self.should_analyze_question = should_analyze_question  # Renamed variable
        
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
        ).to(self.device).eval()  # Explicitly move to device
        
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
                "You are FinSight, a professional financial advisor chatbot specialized in "
                "company analysis and financial insights. Provide accurate, factual responses "
                "and use lists when appropriate to organize information clearly."
            )
        }
        sys_prompt = "You are FinSight, a specialized AI designed to help users understand complex financial concepts and make informed decisions.",

        # Enhanced question patterns with more granular control
        self.question_patterns = {
            # Short responses (24-48 tokens)
            'confirmation': (r'^(yes|no|maybe|correct|right|wrong|true|false)\b\??$', 24),
            'greeting': (r'^\b(hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))\b\s*\??$', 32),
            'farewell': (r'^\b(bye|goodbye|thanks|thank you|exit|quit)\b\s*\??$', 24),
            'acknowledgment': (r'^(ok|okay|sure|alright|i see|got it|understood)\b', 24),
            
            # Brief responses (48-96 tokens)
            'quick_clarification': (r'\b(what does|what is|who is|where is|when is)\b.{0,30}\?$', 48),
            'simple_query': (r'^(can|could|would|should|is|are|do|does)\b.{0,40}\?$', 64),
            'status_check': (r'\b(how (?:is|are|does)|what\'s the status)\b', 72),
            'verification': (r'\b(verify|confirm|check|validate|right|correct)\b', 64),
            
            # Standard responses (96-144 tokens)
            'definition': (r'\b(what (?:is|are|does)|define|explain|meaning of)\b.{0,50}\?', 96),
            'process_query': (r'\b(how (?:to|do|can|should)|what steps|process for)\b', 128),
            'comparison': (r'\b(vs|versus|compare|difference|better|which|or|between)\b', 144),
            'market_status': (r'\b(market|stock|price|rate|trend)\b.*\b(now|today|current|latest)\b', 128),
            
            # Detailed responses (144-192 tokens)
            'strategy': (r'\b(strategy|plan|approach|method|system|framework)\b', 176),
            'explanation': (r'\b(explain|describe|elaborate|tell me about|how does)\b', 160),
            'methodology': (r'\b(methodology|procedure|technique|practice|protocol)\b', 144),
            'risk_analysis': (r'\b(risk|safety|security|protection|danger|threat)\b', 176),
            
            # Comprehensive responses (192-256 tokens)
            'analysis': (r'\b(analyze|evaluate|assess|review|examine|investigate)\b', 224),
            'recommendation': (r'\b(recommend|suggest|advise|propose|guidance|opinion)\b', 208),
            'scenario': (r'\b(scenario|situation|case|example|instance|suppose|imagine)\b', 192),
            'complex_query': (r'.*\b(and|or)\b.*\?.*\b(and|or)\b.*\?', 256),  # Multiple questions
            
            # Financial specifics (192-256 tokens)
            'investment_strategy': (r'\b(portfolio|diversification|allocation|rebalancing)\b', 224),
            'market_analysis': (r'\b(technical analysis|fundamental analysis|indicators)\b', 240),
            'risk_management': (r'\b(hedge|insurance|protection|stop loss|risk management)\b', 208),
            'financial_planning': (r'\b(retirement|estate|tax|planning|long[- ]term)\b', 224),
            
            # Technical topics (224-288 tokens)
            'derivatives': (r'\b(options|futures|swaps|derivatives|contracts)\b', 256),
            'crypto': (r'\b(crypto|blockchain|defi|nft|token|mining)\b', 240),
            'trading_systems': (r'\b(algorithm|automated|system|bot|trading)\b', 224),
            'regulations': (r'\b(regulation|compliance|law|rule|requirement)\b', 288),
        }

    def analyze_question(self, question: str) -> int:
        """Enhanced question analysis with context-aware token control for longer responses"""
        if not isinstance(question, str) or not question.strip():
            return 192  # Increased default length
            
        question = question.lower().strip()
        
        # Initial pattern matching
        matched_tokens = []
        primary_pattern = None
        for pattern_name, (pattern, tokens) in self.question_patterns.items():
            if re.search(pattern, question):
                # Increase token count by 40% for each match to ensure longer responses
                matched_tokens.append(int(tokens * 1.4))
                if not primary_pattern:
                    primary_pattern = pattern_name
        
        # Context-based adjustments with increased multipliers
        context_multiplier = 1.3  # Start with higher base multiplier
        
        # Complexity factors with enhanced multipliers
        if len(question.split()) >= 20:
            context_multiplier *= 1.9  # Increased from 1.7
        elif len(question.split()) >= 10:
            context_multiplier *= 1.6  # Added mid-length question handling
        if question.count('?') > 1:
            context_multiplier *= 1.8  # Increased from 1.6
        if re.search(r'\d+', question):
            context_multiplier *= 1.5  # Increased from 1.4
            
        # Topic-based adjustments
        financial_terms = {
            'complex': {
                'derivatives', 'options', 'futures', 'hedge', 'swap', 'volatility',
                'correlation', 'beta', 'alpha', 'sharpe', 'portfolio', 'risk',
                'technical analysis', 'fundamental analysis', 'arbitrage', 'liquidity'
            },
            'moderate': {
                'stock', 'bond', 'fund', 'etf', 'dividend', 'interest', 'market',
                'investment', 'trading', 'price', 'trend', 'chart', 'valuation', 'sector'
            },
            'basic': {
                'money', 'save', 'spend', 'buy', 'sell', 'profit', 'loss',
                'account', 'bank', 'credit', 'debit', 'invest', 'plan', 'budget'
            }
        }
        
        words = set(question.split())
        if any(term in question for term in financial_terms['complex']):
            context_multiplier *= 2.0  # Increased from 1.8
        elif any(term in question for term in financial_terms['moderate']):
            context_multiplier *= 1.7  # Increased from 1.4
        elif any(term in question for term in financial_terms['basic']):
            context_multiplier *= 1.4  # Increased from 1.25
            
        # Depth indicators
        depth_markers = {
            'detailed': {'explain', 'detail', 'elaborate', 'analyze', 'describe', 'comprehensive', 'thorough'},
            'brief': {'quick', 'brief', 'shortly', 'summary', 'tldr'}
        }
        
        if any(marker in words for marker in depth_markers['detailed']):
            context_multiplier *= 1.9  # Increased from 1.7
        elif any(marker in words for marker in depth_markers['brief']):
            context_multiplier *= 1.0  # Changed from 0.9 to ensure even brief responses aren't too short
        
        # Conversation history context adjustment - provide longer responses as conversation deepens
        if len(self.conversation_history) >= 2:
            context_multiplier *= 1.2
        
        # Calculate final token count
        if matched_tokens:
            base_tokens = max(matched_tokens)
        else:
            # Smart fallback based on question length and structure with increased baselines
            words = len(question.split())
            if words <= 5:
                base_tokens = 128  # Increased from 96
            elif words <= 10:
                base_tokens = 160  # Increased from 128
            elif words <= 15:
                base_tokens = 192  # Increased from 160
            elif words <= 20:
                base_tokens = 224  # Increased from 192
            else:
                base_tokens = 256  # Increased from 216
        
        final_tokens = int(base_tokens * context_multiplier)
        
        # Ensure tokens stay within reasonable bounds but with higher minimum
        return max(96, min(final_tokens, 512))

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response using enhanced prompt handling"""
        max_new_tokens = 512 if not self.should_analyze_question else self.analyze_question(prompt)

        # Keep conversation history manageable
        if len(self.conversation_history) > 4:
            self.conversation_history = self.conversation_history[-2:]

        # Format messages with system prompt reinforcement
        messages = [self.system_prompt]
        
        # Add a brief system reminder before each user message
        if self.conversation_history:
            messages.extend(self.conversation_history)
        
        messages.append({"role": "user", "content": prompt})

        # Enhanced prompt formatting
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # Keep as string for better control
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

        # Adjust generation parameters for more controlled responses
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": min(temperature, 0.7),
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 4,
            "num_beams": self.num_beams,
            "early_stopping": False,
            "length_penalty": 1.0,
        }

        timeout = max(20.0, max_new_tokens // 5.0)
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=timeout
        ) if self.stream else None

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                streamer=streamer,
                **generation_config
            )

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})

        if not self.stream:
            outputs = outputs.cpu()
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split(prompt)[-1].strip()
            
            if not response:
                response = "I apologize, but I couldn't generate a proper response."
            
            self.conversation_history.append({"role": "assistant", "content": response})
            
            if len(self.conversation_history) > 4:
                self.conversation_history = self.conversation_history[-4:]
            
            return response

        return ""

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="qlora_output")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_length", type=int, default=8192)
    # parser.add_argument("--analyze_question", action=False)
    args = parser.parse_args()
    
    advisor = FinanceAdvisor(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        max_length=args.max_length,
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
                temperature=args.temperature,
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