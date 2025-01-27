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
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set precision based on device
        self.precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        logger.info(f"Using precision: {self.precision}")
        
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

    def chat(self, message: str, history: list) -> Generator[str, None, None]:
        """Main chat function with mixed precision inference"""
        messages = []
        
        # Add history
        for user_msg, assistant_msg in history[-1:]: # Only last message
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current message
        messages.append(
            {
                "role": "system",
                "content": "You are Finsight, an AI chatbot designed to provide financial advice. Keep responses short and straightforward. Avoid verbose long-winded explanations."
            }
        )
        messages.append({"role": "user", "content": message})
        
        # Apply template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare inputs with automatic memory management
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=self.precision):
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                return_attention_mask=True
            ).to(self.device)
            
            # Setup streaming
            streamer = TextIteratorStreamer(
                self.tokenizer,
                timeout=10.0,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Optimized generation parameters
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=128,
                temperature=0.3,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                use_cache=True  # Enable KV-cache for faster inference
            )
            
            # Generate in separate thread with mixed precision
            def generate_with_mixed_precision():
                with torch.inference_mode(), torch.amp.autocast(enabled=True, dtype=self.precision):
                    self.model.generate(**generation_kwargs)
            
            thread = Thread(target=generate_with_mixed_precision)
            thread.start()
            
            # Stream response with better memory management
            partial_text = ""
            for new_text in streamer:
                partial_text += new_text
                cleaned = partial_text.strip()
                if cleaned:
                    if "***" in cleaned or "Your Query" in cleaned:
                        cleaned = cleaned.split("***")[0].strip()
                    yield cleaned

def create_demo():
    # Initialize bot
    bot = FinanceAdvisorBot()
    
    # Create demo with modern styling
    demo = gr.ChatInterface(
        fn=bot.chat,
        title="⚡ FinSight AI Advisor",
        description=(
            "Your intelligent financial companion, powered by advanced AI. "
            "Ask me anything about personal finance, investments, or financial planning."
        ),
        examples=[
            ["What's the best way to start investing with $1000?"],
            # ["How can I build an emergency fund?"],
            ["Explain dollar-cost averaging in simple terms."],
            ["What's the difference between stocks and bonds?"],
        ],
        theme=gr.themes.Monochrome(),
        css="""
            .gradio-container {max-width: 900px !important}
            .chat-bubble {border-radius: 12px !important}
            .chat-bubble.user {background-color: #e3f2fd !important}
            .chat-bubble.bot {background-color: #f5f5f5 !important}
        """
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
    favicon_path = setup_assets()
    
    demo = create_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        pwa=True,
        share=True,
        favicon_path=str(favicon_path)
    )

    if not favicon_path.exists():
        # Either create a symlink to an existing favicon
        # os.symlink("/path/to/existing/favicon.ico", favicon_path)
        # Or touch an empty file
        favicon_path.touch()
    
    # demo = create_demo()
    # demo.queue().launch(
    #     server_name="0.0.0.0",
    #     pwa=True,
    #     share=True,
    #     favicon_path=str(favicon_path)
    # )
