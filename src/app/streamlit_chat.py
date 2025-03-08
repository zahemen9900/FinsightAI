import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from pathlib import Path
import logging
from rich.logging import RichHandler
from threading import Thread
import torch.amp
import time
from typing import Generator
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(show_time=False)],
)
logger = logging.getLogger('rich')

# Custom CSS for better UI
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-container {
        padding: 2rem;
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .header-emoji {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    /* Chat container styling */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border: 1px solid #90CAF9;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #F5F5F5 0%, #EEEEEE 100%);
        border: 1px solid #E0E0E0;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Input container styling */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Accent boxes styling */
    .accent-box {
        display: inline-block;
        padding: 0.3em 0.8em;
        border-radius: 8px;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        margin: 0.5em;
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
    }
    
    /* Add smooth scrolling */
    * {
        scroll-behavior: smooth;
    }
</style>
"""

class StreamlitChat:
    def __init__(
        self,
        base_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        adapter_path: str = "qlora_output",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Initialize model and tokenizer
        self.setup_model(base_model, adapter_path)
        
        # Question patterns and system prompt from FinanceAdvisorBot
        self.question_patterns = {
            # ... Copy question_patterns from FinanceAdvisorBot ...
        }
        
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are FinSight, a professional financial advisor chatbot. "
                "Follow these guidelines strictly:\n"
                "1. Provide clear, concise, and accurate financial guidance\n"
                "2. Focus on factual, practical advice without speculation\n"
                "3. Use professional but accessible language\n"
                "4. Break down complex concepts into understandable terms\n"
                "5. Always consider risk management in advice\n"
                "6. Be transparent about limitations of AI advice\n"
                "7. Cite reliable sources when appropriate\n"
                "8. Encourage due diligence and research\n"
                "9. Give bullet points and numbered lists when necessary\n"
                "Remember: You are an AI assistant focused on financial education and guidance."
            )
        }

    def setup_model(self, base_model: str, adapter_path: str):
        """Initialize model and tokenizer with proper settings"""
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
            self.model = PeftModel.from_pretrained(
                base,
                adapter_path,
                torch_dtype=self.precision,
            ).to(self.device)
        else:
            self.model = base
        
        self.model.eval()

    def analyze_question(self, question: str) -> int:
        """Copy analyze_question method from FinanceAdvisorBot"""
        # ... Copy analyze_question implementation ...
        pass

    def generate_response(self, message: str, history: list) -> str:
        """Generate response with proper formatting and streaming"""
        messages = [self.system_prompt]
        
        # Add conversation history
        if history:
            for msg in history[-3:]:  # Keep last 3 turns
                messages.append({"role": "user", "content": msg["user"]})
                if msg.get("assistant"):
                    messages.append({"role": "assistant", "content": msg["assistant"]})
        
        messages.append({"role": "user", "content": message})
        
        # Format prompt
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare inputs
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        ).to(self.device)
        
        # Set up streaming
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=30.0,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generation configuration
        max_new_tokens = 512  #self.analyze_question(message)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate in separate thread
        thread = Thread(target=lambda: self.model.generate(**generation_kwargs))
        thread.start()
        
        # Stream response with placeholder
        response_placeholder = st.empty()
        collected_chunks = []
        for new_text in streamer:
            collected_chunks.append(new_text)
            partial_text = "".join(collected_chunks)
            response_placeholder.markdown(partial_text + "▌")
            time.sleep(0.01)
        
        full_response = "".join(collected_chunks)
        response_placeholder.markdown(full_response)
        return full_response

def main():
    st.set_page_config(
        page_title="FinSight AI - Financial Advisory Assistant",
        page_icon="💡",
        layout="wide"
    )
    
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="header-container">
            <div class="header-emoji">💡</div>
            <div class="header-title">FinSight AI</div>
            <div class="header-subtitle">Your intelligent financial companion, powered by advanced AI</div>
            <div>
                <span class="accent-box">💼 Personal Finance</span>
                <span class="accent-box">📈 Investments</span>
                <span class="accent-box">🎯 Financial Planning</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat interface
    if "chat_interface" not in st.session_state:
        st.session_state.chat_interface = StreamlitChat()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(message["user"])
        if message.get("assistant"):
            with st.chat_message("assistant"):
                st.markdown(message["assistant"])
    
    # Chat input
    if prompt := st.chat_input("How can I help you with your financial questions?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to history
        st.session_state.messages.append({"user": prompt})
        
        # Generate and display response
        with st.chat_message("assistant"):
            response = st.session_state.chat_interface.generate_response(
                prompt, 
                st.session_state.messages[:-1]
            )
            st.session_state.messages[-1]["assistant"] = response
    
    # Example questions
    with st.sidebar:
        st.markdown("### Example Questions")
        examples = [
            "How should I start investing with $5000?",
            "What's the best way to pay off credit card debt?",
            "Should I buy or rent a house?",
            "How do I create a retirement plan?",
            "Explain dollar-cost averaging",
        ]
        for example in examples:
            if st.button(example):
                # Insert example into chat input
                st.chat_input(example)

if __name__ == "__main__":
    main()
