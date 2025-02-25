import re
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
        adapter_path: str = "qlora_output/checkpoint-1500",
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
            logger.info(f"No adapter found at `{adapter_path}`, using base model")
            self.model = base
        
        self.model.eval()
        
        # Enable memory efficient optimizations
        if hasattr(self.model, "set_gradient_checkpointing"):
            self.model.set_gradient_checkpointing(False)
        
        # Update system prompt to match chat_qlora.py
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are FinSight, a professional financial advisor chatbot. "
                "Follow these guidelines strictly:\n"
                "1. Provide clear, concise, and accurate financial guidance\n"
                "2. Focus on factual, practical advice without speculation\n"
                "3. Use professional but accessible language\n"
                "4. Break down complex concepts into understandable terms\n"
                # "5. Maintain objectivity and avoid personal opinions\n"
                "6. Always consider risk management in advice\n"
                "7. Be transparent about limitations of AI advice\n"
                "8. Cite reliable sources when appropriate\n"
                "9. Encourage due diligence and research\n"
                "10. Give bullet points and numbered lists when necessary\n"
                "Remember: You are an AI assistant focused on financial education and guidance."
            )
        }
        self.conversation_history = []

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
        """Enhanced question analysis with context-aware token control"""
        if not isinstance(question, str) or not question.strip():
            return 128  # Default moderate length
            
        question = question.lower().strip()
        
        # Initial pattern matching
        matched_tokens = []
        primary_pattern = None
        for pattern_name, (pattern, tokens) in self.question_patterns.items():
            if re.search(pattern, question):
                matched_tokens.append(tokens)
                if not primary_pattern:
                    primary_pattern = pattern_name
        
        # Context-based adjustments
        context_multiplier = 1.0
        
        # Complexity factors
        if len(question.split()) >= 20:
            context_multiplier *= 1.4  # Longer questions need more detailed responses
        if question.count('?') > 1:
            context_multiplier *= 1.3  # Multiple questions need longer responses
        if re.search(r'\d+', question):
            context_multiplier *= 1.2  # Questions with numbers often need more explanation
            
        # Topic-based adjustments
        financial_terms = {
            'complex': {
                'derivatives', 'options', 'futures', 'hedge', 'swap', 'volatility',
                'correlation', 'beta', 'alpha', 'sharpe', 'portfolio', 'risk',
                'technical analysis', 'fundamental analysis'
            },
            'moderate': {
                'stock', 'bond', 'fund', 'etf', 'dividend', 'interest', 'market',
                'investment', 'trading', 'price', 'trend', 'chart'
            },
            'basic': {
                'money', 'save', 'spend', 'buy', 'sell', 'profit', 'loss',
                'account', 'bank', 'credit', 'debit'
            }
        }
        
        words = set(question.split())
        if any(term in question for term in financial_terms['complex']):
            context_multiplier *= 1.5
        elif any(term in question for term in financial_terms['moderate']):
            context_multiplier *= 1.25
        elif any(term in question for term in financial_terms['basic']):
            context_multiplier *= 1.0
            
        # Depth indicators
        depth_markers = {
            'detailed': {'explain', 'detail', 'elaborate', 'analyze', 'describe'},
            'brief': {'quick', 'brief', 'shortly', 'summary', 'tldr'}
        }
        
        if any(marker in words for marker in depth_markers['detailed']):
            context_multiplier *= 1.3
        elif any(marker in words for marker in depth_markers['brief']):
            context_multiplier *= 0.7
            
        # Calculate final token count
        if matched_tokens:
            base_tokens = max(matched_tokens)
        else:
            # Smart fallback based on question length and structure
            words = len(question.split())
            if words <= 5:
                base_tokens = 64
            elif words <= 10:
                base_tokens = 96
            elif words <= 15:
                base_tokens = 128
            elif words <= 20:
                base_tokens = 160
            else:
                base_tokens = 192
        
        final_tokens = int(base_tokens * context_multiplier)
        
        # Ensure tokens stay within reasonable bounds
        return max(48, min(final_tokens, 512))

    def chat(self, message: str, history: list) -> Generator[str, None, None]:
        """Main chat function with improved streaming and generation"""
        messages = []
        
        # Format messages with system prompt first
        messages.append(self.system_prompt)
        
        # Add conversation history from Gradio's history format
        if history:
            # History comes as a list of [user_message, bot_message] pairs
            for human, assistant in history[-3:]:  # Keep last 3 turns
                if human:
                    messages.append({"role": "user", "content": human})
                if assistant:
                    messages.append({"role": "assistant", "content": assistant})
                    
            # Add system reminder after history
            messages.append({
                "role": "system",
                "content": "Remember to provide professional financial guidance. Also give bullet points and numbered lists when necessary."
            })
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Apply chat template using tokenizer's built-in method
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        with torch.inference_mode(), torch.amp.autocast(enabled=True, dtype=self.precision, device_type=self.device, cache_enabled=True):
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                return_attention_mask=True
            ).to(self.device)
            
            # Align streaming configuration with chat_qlora.py
            streamer = TextIteratorStreamer(
                self.tokenizer,
                timeout=20.0,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=self.analyze_question(message),
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=False,
                length_penalty=1.0,
                use_cache=True
            )
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Improved response streaming with better formatting
            partial_text = ""
            for new_text in streamer:
                partial_text += new_text
                cleaned = partial_text.strip()
                if cleaned:
                    # Clean up response formatting
                    if "***" in cleaned or "Your Query" in cleaned:
                        cleaned = cleaned.split("***")[0].strip()
                    # Ensure proper capitalization
                    if cleaned and cleaned[0].lower():
                        cleaned = cleaned[0].upper() + cleaned[1:]
                    # Ensure proper ending punctuation
                    if cleaned and cleaned[-1] not in '.!?':
                        cleaned += '.'
                    yield cleaned

def create_demo():
    # Initialize bot
    bot = FinanceAdvisorBot()
    
    with gr.Blocks(theme=gr.themes.Ocean(),
                  css="""
        /* Base container styling */
        .gradio-container {
            max-width: 750px !important;
            margin: 0 auto !important;
        }

        /* Chat bubble styling */
        .chat-bubble {
            border-radius: 15px !important;
            padding: 15px 20px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        }
        .chat-bubble.user {
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%) !important;
            border: 1px solid #90CAF9 !important;
        }
        .chat-bubble.bot {
            background: linear-gradient(135deg, #F5F5F5 0%, #EEEEEE 100%) !important;
            border: 1px solid #E0E0E0 !important;
        }
        
        /* Modern Header Styling */
        .custom-header {
            text-align: center;
            margin: -10px -10px -20px -10px;
            padding: 30px 20px;
            background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
            position: relative;
            overflow: hidden;
            max-width: 900px;
        }

        /* Animated gradient overlay */
        .custom-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, 
                rgba(33, 150, 243, 0.15) 0%,
                rgba(3, 169, 244, 0.15) 25%,
                rgba(0, 188, 212, 0.15) 50%,
                rgba(3, 169, 244, 0.15) 75%,
                rgba(33, 150, 243, 0.15) 100%);
            animation: gradient 15s ease infinite;
            background-size: 400% 400%;
        }

        /* Header content styling */
        .header-content {
            position: relative;
            z-index: 2;
        }

        .custom-header h1 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 800 !important;
            color: white !important;
            font-size: 3.8em !important;
            margin-bottom: 0.3rem !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
            letter-spacing: -0.5px !important;
        }

        .custom-header .logo {
            font-size: 2.5em;
            margin-right: 0.3em;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .custom-header p {
            font-family: 'Inter', sans-serif !important;
            font-size: 1.25em !important;
            line-height: 1.6 !important;
            color: rgba(255,255,255,0.95) !important;
            margin: 1.2rem auto 0 !important;
            max-width: 700px !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        }

        /* Accent elements */
        .accent-box {
            display: inline-block;
            padding: 0.3em 0.8em;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            margin: 0.5em;
            border: 1px solid rgba(255,255,255,0.2);
        }

        /* Animation keyframes */
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    """) as demo:
        
        # Enhanced header HTML with accent elements
        gr.HTML("""
            <div class="custom-header">
                <div class="header-content">
                    <h1>💡FinSight AI</h1>
                    <p>Your intelligent financial companion, powered by advanced AI</p>
                    <div>
                        <span class="accent-box">💼 Personal Finance</span>
                        <span class="accent-box">📈 Investments</span>
                        <span class="accent-box">🎯 Financial Planning</span>
                    </div>
                </div>
            </div>
        """)
        
        # Create the chat interface with proper history handling
        chat_interface = gr.ChatInterface(
            bot.chat,
            chatbot=gr.Chatbot(
                height=600,
                show_copy_button=True,
                show_share_button=True,
                bubble_full_width=False
            ),
            examples=[
                ["I have $5000 saved up. What's the best way to start investing it?"],
                ["My monthly income is $4000. How should I split my expenses and savings?"],
                ["I have $10,000 in credit card debt with 20% APR. What's my best strategy to pay it off?"],
                ["Should I consider buying or renting a house?"],
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
