from transformers import AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def get_special_tokens(tokenizer=tokenizer):
    """
    Get and print all special tokens from a transformer model's tokenizer.
    
    Args:
        model_name (str): Name of the pre-trained model
    """
    try:        
        # Get all special tokens
        special_tokens = {
            "All Special Tokens": tokenizer.all_special_tokens,
            "Special Tokens Map": tokenizer.special_tokens_map,
            "Pad Token": tokenizer.pad_token,
            "UNK Token": tokenizer.unk_token,
            "SEP Token": tokenizer.sep_token,
            "CLS Token": tokenizer.cls_token,
            "Mask Token": tokenizer.mask_token
        }
        
        # Print all special tokens
        print(f"\nSpecial tokens for model: {model_name}")
        print("-" * 50)
        for token_type, tokens in special_tokens.items():
            print(f"{token_type}: {tokens}")
            
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")

def test_chat_template(tokenizer=tokenizer):
    """
    Test the chat template functionality of a transformer model's tokenizer.
    
    Args:
        model_name (str): Name of the pre-trained model
    """
    try:        
        # Example conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you?"}
        ]
        
        # Apply chat template
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        print("\nChat Template Example:")
        print("-" * 50)
        print(f"Formatted conversation:\n{formatted}")
        
    except Exception as e:
        print(f"Error testing chat template: {str(e)}")

def test_metadata_support(tokenizer=tokenizer):
    """
    Test if the model's tokenizer supports metadata in chat templates.
    
    Args:
        model_name (str): Name of the pre-trained model
    """
    try:       
        # Test message with metadata
        messages = [
            {"role": "user", "content": "Hello!", "metadata": {"timestamp": "2024-01-01"}}
        ]
        
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            print("\nMetadata Support Test:")
            print("-" * 50)
            print("Model supports metadata in chat template")
            print(f"Formatted result:\n{formatted}")
        except ValueError as ve:
            if "metadata" in str(ve).lower():
                print("\nMetadata Support Test:")
                print("-" * 50)
                print("Model does not support metadata in chat template")
            else:
                raise ve
                
    except Exception as e:
        print(f"Error testing metadata support: {str(e)}")

def check_special_tokens_after_tokenizing(tokenizer=tokenizer):
    """
    Check if special tokens are available after tokenizing a sequence.
    """
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"}
    ]

    formatted_text = tokenizer.apply_chat_template(test_messages, tokenize=False)
    print(formatted_text)

    tokens = tokenizer(formatted_text, return_tensors="pt")
    decoded = tokenizer.decode(tokens["input_ids"][0])
    print("Decoded Output:\n", decoded)  # Should end with the EOS token

if __name__ == "__main__":
    # Example usage
    get_special_tokens()
    test_chat_template()
    test_metadata_support()
    check_special_tokens_after_tokenizing()