import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from pathlib import Path
from threading import Thread
import logging
from rich.logging import RichHandler
from typing import Generator
import shutil
from pathlib import Path
import torch.amp
import re  # Add this import at the top

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(show_time=False)],
)
logger = logging.getLogger('rich')

class FinanceAdvisorBot:
    def __init__(
        self,
        base_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        adapter_path: str = "qlora_output",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Set precision based on device
        self.precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        logger.info(f"Precision set to {self.precision}")
        
        # Load tokenizer and model with precision
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            padding_side="left",
            trust_remote_code=True
        )
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=self.precision,
            trust_remote_code=True,
        ).to(self.device)
        
        if Path(adapter_path).exists():
            logger.info(f"Loading adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(
                base,
                adapter_path,
                torch_dtype=self.precision,
            ).to(self.device)
            logger.info("Adapter loaded successfully")
        else:
            logger.info("No adapter found, using base model")
            self.model = base
        
        self.model.eval()
        
        # Enable memory efficient optimizations
        if hasattr(self.model, "set_gradient_checkpointing"):
            self.model.set_gradient_checkpointing(False)
        
        # Add new system prompt for better responses
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are FinSight, a professional financial advisor. "
                "Keep responses clear, focused, and concise. "
            )
        }
        self.conversation_history = []

        # Add question type patterns
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
            # ...add other patterns as needed...
        }

    def analyze_question(self, question: str) -> int:
        """Determine appropriate response length based on question type"""
        question = question.lower().strip()
        
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

    def chat(self, message: str, history: list) -> Generator[str, None, None]:
        """Main chat function with improved streaming and generation"""
        # Determine appropriate response length
        max_new_tokens = self.analyze_question(message)
        
        # Format messages with history
        messages = [self.system_prompt]
        for user_msg, assistant_msg in history[-3:]:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})
        
        # Apply template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Set up generation
        with torch.inference_mode(), torch.amp.autocast(enabled=True, dtype=self.precision, device_type=self.device, cache_enabled=True):
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                return_attention_mask=True
            ).to(self.device)
            
            # Setup streaming with dynamic timeout
            timeout = max(20.0, max_new_tokens / 5.0)
            streamer = TextIteratorStreamer(
                self.tokenizer,
                timeout=timeout,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            def generate_with_mixed_precision():
                with torch.inference_mode(), torch.amp.autocast(enabled=True, dtype=self.precision, device_type=self.device, cache_enabled=True):
                    # Dynamic temperature based on response length
                    temperature = min(0.7, 0.5 if max_new_tokens < 84 else 0.7)
                    
                    generation_kwargs = dict(
                        **inputs,
                        streamer=streamer,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=5,
                        num_beams=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        use_cache=True
                    )
                    
                    self.model.generate(**generation_kwargs)
            
            thread = Thread(target=generate_with_mixed_precision)
            thread.start()
            
            # Stream response
            partial_text = ""
            for new_text in streamer:
                partial_text += new_text
                cleaned = partial_text.strip()
                if cleaned:
                    if "***" in cleaned or "Your Query" in cleaned:
                        cleaned = cleaned.split("***")[0].strip()
                    if cleaned and cleaned[0].islower():
                        cleaned = cleaned[0].upper() + cleaned[1:]
                    yield cleaned

def create_demo():
    # Initialize bot
    bot = FinanceAdvisorBot()
    
    with gr.Blocks(theme=gr.themes.Ocean(),
                  css="""
        .gradio-container {max-width: 900px !important}
        .chat-bubble {border-radius: 12px !important}
        .chat-bubble.user {background-color: #e3f2fd !important}
        .chat-bubble.bot {background-color: #f5f5f5 !important}
        
        /* Custom Header Styling */
        .custom-header {
            text-align: center;
            margin-bottom: 2rem;
            padding: 2.5rem;
            background: linear-gradient(135deg, #4286f4 0%, #373B44 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(255, 75, 31, 0.2);  # Matching shadow tint
        }
        .custom-header h1 {
            font-family: 'Space Mono', monospace !important;
            font-weight: 700 !important;
            color: white !important;
            letter-spacing: 1px !important;
            font-size: 3.5em !important;
            margin-bottom: 0.5rem !important;
        }
        
        .custom-header p {
            font-size: 1.2em !important;
            opacity: 0.9;
            margin-top: 1rem !important;
        }
        
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    """) as demo:
        
        # Add the custom header as HTML
        gr.HTML("""
            <div class="custom-header">
                <h1>⚡FinSight AI</h1>
                <p>Your intelligent financial companion, powered by advanced AI.<br>
                Ask me anything about personal finance, investments, or financial planning.</p>
            </div>
        """)
        
        # Create the chat interface
        chat_interface = gr.ChatInterface(
            fn=bot.chat,
            examples=[
                ["What's the best way to start investing with $1000?"],
                ["Explain dollar-cost averaging in simple terms."],
                ["What's the difference between stocks and bonds?"],
            ],
        )
    
    return demo

def setup_assets():
    """Set up assets directory and files"""
    current_dir = Path(__file__).parent
    assets_dir = current_dir / "static" / "assets"
    favicon_path = assets_dir / "favicon.ico"
    
    # Create assets directory
    # If favicon doesn't exist, create a default one or copy from another location
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy default favicon if it doesn't exist
    if not favicon_path.exists():
        default_favicon = current_dir.parent.parent / "assets" / "favicon.ico"
        if default_favicon.exists():
            shutil.copy(default_favicon, favicon_path)
        else:
            # Create empty favicon if no default exists
            favicon_path.touch()
    
    return favicon_path

if __name__ == "__main__":
    # favicon_path = setup_assets()
    
    demo = create_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        pwa=True,
        share=True,
        # favicon_path=str(favicon_path)
    )

    # if not favicon_path.exists():
    #     # Either create a symlink to an existing favicon
    #     # os.symlink("/path/to/existing/favicon.ico", favicon_path)
    #     # Or touch an empty file
    #     favicon_path.touch()
    
    # # demo = create_demo()
    # # demo.queue().launch(
    # #     server_name="0.0.0.0",
    # #     pwa=True,
    # #     share=True,
    # #     favicon_path=str(favicon_path)
    # # )
