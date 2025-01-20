import os
import shutil
import logging
from pathlib import Path
from huggingface_hub import (
    HfFolder,
    create_repo,
    upload_file,
    Repository
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')

def setup_repo_structure(base_path: Path) -> None:
    """Create standard repo structure if it doesn't exist"""
    dirs = ['src/alignment', 'src/main', 'notebooks', 'models', 'data']
    
    for dir_path in dirs:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
    # Create empty __init__.py files
    for dir_path in dirs:
        if 'src' in dir_path:
            init_file = base_path / dir_path / '__init__.py'
            init_file.touch()

def create_metadata_files(base_path: Path, model_info: dict) -> None:
    """Create metadata files like README, requirements.txt etc"""
    
    # Save requirements.txt
    requirements = [
        "torch>=2.0.0",
        "transformers",
        "datasets",
        "peft",
        "trl",
        "bitsandbytes",
        "wandb",
        "rich"
    ]
    
    with open(base_path / "requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    # Save model card as README.md
    model_card = f"""
# {model_info['name']} - Financial Advisory Chatbot

{model_info['description']}

## Model Details

- **Base Model**: {model_info['base_model']}
- **Task**: {model_info['task']}
- **Training Data**: {model_info['data_desc']}
- **Training Method**: {model_info['train_method']}
- **Language**: {model_info['language']}
- **License**: {model_info['license']}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_info['repo_id']}")
tokenizer = AutoTokenizer.from_pretrained("{model_info['repo_id']}")

# Example usage
prompt = "What's a good strategy for long-term investing?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```
"""
    
    with open(base_path / "README.md", "w") as f:
        f.write(model_card)

def push_to_hub(
    model_path: str,
    repo_name: str,
    commit_message: str = "Initial commit with fine-tuned model", 
    private: bool = False,
    token: str = None,
    create_local: bool = True
) -> None:
    """
    Create and push full model repository to HuggingFace Hub
    
    Args:
        model_path: Path to saved model directory
        repo_name: Name for HuggingFace repo (e.g. 'username/model-name')
        commit_message: Message for the commit
        private: Whether to make the repo private
        token: HuggingFace API token. If None, will look for HUGGING_FACE_HUB_TOKEN env var
        create_local: Whether to create local repository structure
    """
    if not Path(model_path).exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Set HF token
    if token:
        HfFolder.save_token(token)
    elif not os.getenv("HUGGING_FACE_HUB_TOKEN"):
        raise ValueError("No HuggingFace token provided")

    try:
        logger.info("Setting up repository...")
        
        # Create repo on HF Hub
        repo_url = create_repo(
            repo_id=repo_name,
            private=private,
            token=token,
            exist_ok=True
        )
        
        # Setup local repo if requested
        if create_local:
            base_path = Path.cwd().parent.parent  # Get FinsightAI root dir
            setup_repo_structure(base_path)
            
            # Define model info for metadata
            model_info = {
                "name": "FinsightAI",
                "description": "A fine-tuned version of SmolLM2-1.7B optimized for financial advice and discussion",
                "base_model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "task": "Financial Advisory and Discussion",
                "data_desc": "250k+ high quality Reddit financial discussions",
                "train_method": "QLoRA (4-bit quantization with LoRA)",
                "language": "English",
                "license": "MIT",
                "repo_id": repo_name
            }
            
            create_metadata_files(base_path, model_info)
            
            # Clone the empty repo
            repo = Repository(
                local_dir=base_path / "models" / repo_name.split("/")[-1],
                clone_from=repo_url,
                token=token
            )
            
            # Copy model files to repo directory
            logger.info(f"Copying model files from {model_path}")
            for item in Path(model_path).glob("*"):
                if item.is_file():
                    shutil.copy2(item, repo.local_dir)
            
            # Push everything
            logger.info("Pushing repository...")
            repo.push_to_hub(
                commit_message=commit_message
            )
            
        else:
            # Just push the model directly
            logger.info(f"Loading model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            logger.info(f"Pushing to {repo_name}")
            model.push_to_hub(repo_name, token=token)
            tokenizer.push_to_hub(repo_name, token=token)

        logger.info(f"Successfully pushed to https://huggingface.co/{repo_name}")

    except Exception as e:
        logger.error(f"Error pushing to hub: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--repo_name", type=str, required=True, help="HuggingFace repo name (e.g. 'username/model-name')")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--token", type=str, help="HuggingFace token (optional if env var set)")
    parser.add_argument("--local", action="store_true", help="Create local repository structure")
    
    args = parser.parse_args()
    
    push_to_hub(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private,
        token=args.token,
        create_local=args.local
    )
