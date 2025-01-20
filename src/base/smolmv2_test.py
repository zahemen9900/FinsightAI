import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize tokenizer and model with device handling
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct").to(device)

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

def generate_response(conversation, prompt, max_length=256, temperature=0.7):
    # Add the new prompt to conversation
    conversation.add_message("human", prompt)
    
    # Get full context including history
    context = conversation.get_formatted_context()
    
    # Tokenize with padding and create attention mask
    inputs = tokenizer(
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Add parameters to improve response quality
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response
    try:
        # Get everything after the last "### Assistant:" but before any new "### Human:"
        assistant_response = response.split("### Assistant:")[-1]
        if "### Human:" in assistant_response:
            assistant_response = assistant_response.split("### Human:")[0]
        assistant_response = assistant_response.strip()
    except:
        assistant_response = "I apologize, but I couldn't generate a proper response."
    
    # Add assistant's response to conversation
    conversation.add_message("assistant", assistant_response)
    return assistant_response

def chat():
    conversation = Conversation()
    print("Chat with SMOLLM-V2! (type 'quit' to exit)")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
            
        response = generate_response(conversation, user_input)
        print("\nAssistant:", response)

if __name__ == "__main__":
    chat()
