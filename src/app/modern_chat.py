import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from pathlib import Path
from threading import Thread
import logging
from rich.logging import RichHandler

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
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            padding_side="left",
            trust_remote_code=True
        )
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        
        if Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(
                base,
                adapter_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)
        else:
            self.model = base
        
        self.model.eval()

    def chat(self, message: str, history: list) -> str:
        """Main chat function for Gradio interface"""
        # Format conversation
        chat_history = []
        for user_msg, assistant_msg in history:
            chat_history.append({"role": "user", "content": user_msg})
            if assistant_msg:
                chat_history.append({"role": "assistant", "content": assistant_msg})
        chat_history.append({"role": "user", "content": message})
        
        # Create prompt
        system_prompt = (
            "You are FinSight, a professional financial advisor chatbot. Follow these rules strictly:\n"
            "1. Always use proper punctuation and grammar\n"
            "2. Use standard sentence case (not Title Case or ALL CAPS)\n"
            "3. End all sentences with appropriate punctuation marks\n"
            "4. Keep responses focused and well-structured\n"
            "5. Use commas, periods, and other punctuation marks correctly\n"
            "6. Never use hashtags, emojis, or @ mentions\n"
            "7. Format responses in clear, complete paragraphs\n"
            "8. Maintain formal, professional language"
        )

        
        formatted_prompt = f"{system_prompt}\n\n"
        for msg in chat_history:
            if msg["role"] == "user":
                formatted_prompt += f"### Human: {msg['content']}\n"
            else:
                formatted_prompt += f"### Assistant: {msg['content']}\n"
        formatted_prompt += "### Assistant:"

        # Tokenize and prepare inputs
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
            timeout=20.0,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generation config
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        
        # Generate in separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream response
        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            yield partial_text

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
            ["How can I build an emergency fund?"],
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

if __name__ == "__main__":
    demo = create_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        pwa=True,
        share=True,
        favicon_path="assets/favicon.ico"
    )
