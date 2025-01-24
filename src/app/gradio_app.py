import gradio as gr
import torch
import sys
from pathlib import Path
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import logging
from rich.logging import RichHandler

# Configure logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(show_time=False)],
)
logger = logging.getLogger('rich')

class FinanceAdvisorChat:
    def __init__(
        self,
        base_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        adapter_path: str = "qlora_output",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        
    def format_prompt(self, message: str, history: list) -> str:
        system_prompt = (
            "You are FinSight, a professional financial advisor chatbot. Follow these rules strictly:\n"
            "1. Always use proper punctuation and grammar\n"
            "2. Use standard sentence case\n"
            "3. Keep responses focused and well-structured\n"
            # "4. Maintain formal, professional language"
        )
        
        formatted = f"{system_prompt}\n\n"
        for user_msg, assistant_msg in history:
            formatted += f"### Human: {user_msg}\n"
            if assistant_msg:
                formatted += f"### Assistant: {assistant_msg}\n"
        formatted += f"### Human: {message}\n### Assistant:"
        return formatted

    def clean_response(self, text: str) -> str:
        """Clean and format the response text"""
        # Fix capitalization issues
        text = '. '.join(s.strip().capitalize() for s in text.split('.') if s.strip())
        text = text.replace('.', '. ').replace('?', '? ').replace('!', '! ')
        text = ' '.join(text.split())
        # text = text.replace('#', '').replace('@', '')
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    def generate_response(self, message: str, history: list, temperature: float = 0.3, max_tokens: int = 256):
        """Generate response with proper chat history handling"""
        # Add the current message to history
        new_history = history + [[message, None]]
        formatted_prompt = self.format_prompt(message, history)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        ).to(self.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=20.0,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=min(temperature, 0.5),
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.5,
            no_repeat_ngram_size=5,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            cleaned_response = self.clean_response(partial_text)
            new_history[-1][1] = cleaned_response
            yield new_history

# Updated CSS for modern, centered design
custom_css = """
.container {
    max-width: 1000px !important;
    margin: auto !important;
    padding: 2rem;
}

#main-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2rem;
    max-width: 900px;
    margin: 0 auto;
    padding: 1rem;
}

#title {
    text-align: center;
    background: linear-gradient(90deg, #1a237e, #0d47a1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em !important;
    font-weight: 800;
    margin: 1rem 0 0.5rem 0;
    width: 100%;
}

#subtitle {
    text-align: center;
    color: #666;
    font-size: 1.2em;
    margin-bottom: 1rem;
    width: 100%;
}

.feature-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    width: 100%;
    margin: 1rem auto;
    padding: 0 1rem;
}

.feature-card {
    background: transparent !important;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
    text-align: center;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.feature-icon {
    font-size: 2em;
    margin-bottom: 0.75em;
    color: #1a237e;
}

.feature-card h3 {
    color: #1a237e;
    margin-bottom: 0.5em;
    font-size: 1.2em;
    font-weight: 600;
}

.feature-card p {
    color: #666;
    font-size: 0.95em;
    line-height: 1.5;
}

#chatbot {
    width: 100%;
    height: 600px;
    overflow: auto;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin: 0rem auto;
    padding: 0.5rem;
}

.message-actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
    justify-content: flex-end;
}

.input-row {
    display: flex;
    gap: 0.5rem;
    width: 100%;
    margin: 1rem auto;
}

.input-container {
    background: transparent !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 12px !important;
    padding: 0.5rem 1rem !important;
}

.submit-btn {
    background: linear-gradient(90deg, #1a237e, #0d47a1) !important;
    border: none !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 0.5rem 1.5rem !important;
}

.control-buttons {
    display: flex;
    gap: 1rem;
    width: 100%;
    justify-content: center;
    margin: 0.5rem 0;
}

.examples-container {
    width: 100%;
    margin: 1rem auto;
}
"""

def create_demo():
    advisor = FinanceAdvisorChat()
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_id="main-container"):
            gr.HTML("""
                <h1 id="title">💼 FinSight AI Advisor</h1>
                <p id="subtitle">Your intelligent financial companion, powered by advanced AI</p>
            """)
            
            # Feature cards
            gr.HTML("""
                <div class="feature-cards">
                    <div class="feature-card">
                        <div class="feature-icon">💡</div>
                        <h3>Smart Insights</h3>
                        <p>Get intelligent financial advice tailored to your needs</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🔒</div>
                        <h3>Secure & Private</h3>
                        <p>Your conversations are private and never stored</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">⚡</div>
                        <h3>Real-time Analysis</h3>
                        <p>Instant responses to your financial queries</p>
                    </div>
                </div>
            """)
            
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    height=600,
                    avatar_images=("👤", "🤖"),
                    show_label=False,
                )
                
                with gr.Row(elem_classes="input-row"):
                    msg = gr.Textbox(
                        show_label=False,
                        placeholder="Ask me anything about finance...",
                        scale=9,
                        elem_classes="input-container"
                    )
                    submit = gr.Button("Send", variant="primary", scale=1, elem_classes="submit-btn")

                with gr.Row(elem_classes="control-buttons"):
                    clear = gr.Button("Clear Chat", size="sm")
                    retry = gr.Button("Retry Last", size="sm")
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=0.5,
                            value=0.3,
                            step=0.1,
                            label="Response Creativity"
                        )
                        max_tokens = gr.Slider(
                            minimum=64,
                            maximum=512,
                            value=256,
                            step=32,
                            label="Response Length"
                        )

                with gr.Column(elem_classes="examples-container"):
                    gr.Examples(
                        examples=[
                            "What's the best way to start investing with $1000?",
                            "How can I build an emergency fund?",
                            "Explain dollar-cost averaging in simple terms.",
                            "What's the difference between stocks and bonds?",
                        ],
                        inputs=msg,
                    )

        # Event handlers
        last_response = gr.State("")
        
        def retry_last(history, last_msg, temp, tokens):
            if history:
                history.pop()  # Remove last response
                return advisor.generate_response(last_msg, history[:-1], temp, tokens)
            return history

        msg.submit(
            advisor.generate_response,
            [msg, chatbot, temperature, max_tokens],
            [chatbot]
        ).then(lambda x: "", None, msg)

        submit.click(
            advisor.generate_response,
            [msg, chatbot, temperature, max_tokens],
            [chatbot]
        ).then(lambda x: "", None, msg)

        clear.click(lambda: [], outputs=[chatbot])
        retry.click(
            retry_last,
            [chatbot, msg, temperature, max_tokens],
            [chatbot]
        )

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        share=True,
        favicon_path="assets/favicon.ico"
    )
