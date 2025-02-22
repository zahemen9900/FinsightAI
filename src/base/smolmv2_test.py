import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from rich.console import Console
from rich.markdown import Markdown

console = Console()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
    trust_remote_code=True
)

# Model initialization with dynamic dtype
precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # Default to bfloat16 on GPU

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=precision,
    trust_remote_code=True,
    device_map="auto"
).eval()

class Conversation:
    def __init__(self):
        self.history = []
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

def generate_response(conversation, prompt: str, stream: bool = True, precision=precision) -> str:
    """Generate response with better handling and mixed precision."""
    # Format messages with system prompt
    messages = conversation.history + [
        {"role": "system", "content": (
            "You are FinSight AI, a financial advisor. "
            "Keep responses concise and to the point. Avoid long-winded explanations."
        )},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Prepare inputs
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)
    
    # Setup streaming
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=precision, cache_enabled=True):
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            streamer=streamer
        )
    
    if not stream:
        # Get response without streaming
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split(prompt)[-1].strip()
        print(f"\nAssistant: {response}")
        return response

def chat():
    """Interactive chat loop with better formatting."""
    conversation = Conversation()
    console.print("\n[bold cyan]Welcome to FinSight AI - Your Financial Advisor![/bold cyan]")
    console.print("[dim]Type 'quit' to exit, 'clear' to start fresh[/dim]\n")
    
    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                conversation = Conversation()
                console.print("[dim]Conversation history cleared[/dim]")
                continue
            
            # Add user message to history
            conversation.add_message("user", user_input)
            
            # Generate and display response
            console.print("\n[bold blue]Assistant:[/bold blue] ", end="")
            response = generate_response(conversation, user_input, precision=precision)
            
            # Add assistant response to history
            if response:
                conversation.add_message("assistant", response)
            
            console.print()  # Add newline after response
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            continue
    
    console.print("\n[bold cyan]Thanks for chatting! Goodbye![/bold cyan]")

if __name__ == "__main__":
    chat()
