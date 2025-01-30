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

    def chat(self, message: str, history: list) -> Generator[str, None, None]:
        """Main chat function with improved streaming and generation"""
        messages = []
        
        # Format messages with system prompt first
        messages.append(self.system_prompt)
        
        # Add history (last few turns only)
        for user_msg, assistant_msg in history[-3:]:  # Keep last 3 turns for context
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Apply chat template using tokenizer's built-in method
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare inputs with memory management
        with torch.inference_mode(), torch.amp.autocast(enabled=True, dtype=self.precision, device_type=self.device, cache_enabled=True):
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                return_attention_mask=True
            ).to(self.device)
            
            # Setup streaming with improved parameters
            streamer = TextIteratorStreamer(
                self.tokenizer,
                timeout=20.0,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            def generate_with_mixed_precision():
                with torch.inference_mode(), torch.amp.autocast(enabled=True, dtype=self.precision, device_type=self.device, cache_enabled=True):
                    # Aligned generation parameters with chat_qlora.py
                    generation_kwargs = dict(
                        **inputs,
                        streamer=streamer,
                        max_new_tokens=128,
                        temperature=0.7,
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
            
            # Stream response with improved handling
            partial_text = ""
            for new_text in streamer:
                partial_text += new_text
                cleaned = partial_text.strip()
                if cleaned:
                    # Remove any unwanted markers and format response
                    if "***" in cleaned or "Your Query" in cleaned:
                        cleaned = cleaned.split("***")[0].strip()
                    # Ensure response starts with proper capitalization
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
