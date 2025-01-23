import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import sys

# Initialize tokenizer and model with device handling and optimizations
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Optimize tokenizer settings
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
    trust_remote_code=True
)

# Load model with optimizations
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    device_map="auto"  # Automatically handle model placement
).eval()  # Set to eval mode immediately

class Conversation:
    def __init__(self):
        self.history = []
    
    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
    
    def get_formatted_context(self):
        formatted = ""
        for msg in self.history[-3:]:  # Only keep last 3 messages for context window
            if msg["role"] == "human":
                formatted += f"### Human: {msg['content']}\n"
            else:
                formatted += f"### Assistant: {msg['content']}\n"
        return formatted.strip() + "\n### Assistant:"  # Add final assistant prompt

def generate_response(conversation, prompt, max_length=256, temperature=0.7, stream=True):
    conversation.add_message("human", prompt)
    context = conversation.get_formatted_context()
    
    inputs = tokenizer(
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_attention_mask=True
    ).to(device)
    
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    ) if stream else None
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            num_beams=1,  # Set to 1 since we're not doing beam search
            early_stopping=False,  # Disable early stopping since num_beams=1
            streamer=streamer
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up response
    try:
        assistant_response = response.split("### Assistant:")[-1]
        if "### Human:" in assistant_response:
            assistant_response = assistant_response.split("### Human:")[0]
        assistant_response = assistant_response.strip()
    except:
        assistant_response = "I apologize, but I couldn't generate a proper response."
    
    conversation.add_message("assistant", assistant_response)
    return assistant_response

def chat():
    conversation = Conversation()
    print("Chat with SMOLLM-V2! (type 'quit' to exit)")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
            
        print("\nAssistant: ", end="")  # Start response line
        response = generate_response(conversation, user_input, stream=True)
        print("\n")  # Add newline after streamed response

if __name__ == "__main__":
    chat()
