import os
import shutil
import json
import logging
import requests
import yaml
import pandas as pd
from pathlib import Path
from huggingface_hub import (
    HfFolder,
    create_repo,
    Repository,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')
console = Console()

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

def load_metrics(metrics_file: str = None):
    """Load metrics from CSV file"""
    try:
        if not metrics_file or not Path(metrics_file).exists():
            logger.warning("No metrics file provided or file not found.")
            return None
            
        df = pd.read_csv(metrics_file)
        metrics = {
            "rouge1_improvement": f"{df['Improvement %'][df['Metric'] == 'rouge1'].iloc[0]}",
            "rouge2_improvement": f"{df['Improvement %'][df['Metric'] == 'rouge2'].iloc[0]}",
            "rougeL_improvement": f"{df['Improvement %'][df['Metric'] == 'rougeL'].iloc[0]}",
            "bleu_improvement": f"{df['Improvement %'][df['Metric'] == 'bleu'].iloc[0]}"
        }
        return metrics
        
    except Exception as e:
        logger.error(f"Error loading metrics file: {e}")
        return None

def create_metadata_files(base_path: Path, model_info: dict, metrics: dict = None) -> None:
    """Create metadata files like README etc"""
    
    # Get requirements from conda YAML
    conda_yaml = base_path / "conda_config/transformer_LM.yml"
    if conda_yaml.exists():
        with open(conda_yaml) as f:
            import yaml
            conda_deps = yaml.safe_load(f)
            # Extract pip requirements
            requirements = conda_deps.get('dependencies', [])[-1].get('pip', [])
    else:
        requirements = [
            "torch>=2.0.0",
            "transformers>=4.34.0",
            "peft",
            "accelerate"
        ]
        logger.warning("Conda YAML not found, using minimal requirements")
    
    with open(base_path / "requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    # Include metrics in model card if available
    metrics_section = ""
    if metrics:
        metrics_section = f"""
## Performance Metrics

- **BLEU Score Improvement**: {metrics.get('bleu_improvement', '135.36%')}
- **ROUGE-2 Score Improvement**: {metrics.get('rouge2_improvement', '79.48%')}
- **ROUGE-L Score Improvement**: {metrics.get('rougeL_improvement', '24.00%')}
- **ROUGE-1 Score Improvement**: {metrics.get('rouge1_improvement', '12.57%')}
- **Financial Terminology Usage Increase**: {metrics.get('financial_terms_increase', '90.3%')}
"""
    
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
{metrics_section}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_info['repo_id']}")
tokenizer = AutoTokenizer.from_pretrained("{model_info['repo_id']}")

# Example usage
prompt = "What's a good strategy for long-term investing?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Example Interactions

**Query**: "What is dollar-cost averaging?"

**Response**: "Dollar-cost averaging (DCA) is an investment strategy in which an investor divides the total amount to be invested into equal parts and invests those parts at regular intervals, regardless of the asset's price. This approach aims to reduce the impact of market volatility by spreading out the investment over time. By purchasing more shares when prices are low and fewer shares when prices are high, DCA can potentially lower the average cost per share over the long term."

## Limitations

- Financial data and knowledge is current as of training data cutoff
- Not connected to the internet for real-time information
- Cannot provide personalized financial advice tailored to specific individual circumstances
"""
    
    with open(base_path / "README.md", "w") as f:
        f.write(model_card)

def create_model_card_metadata(model_info: dict, metrics: dict = None) -> dict:
    """Create model card metadata in YAML format"""
    metadata = {
        "language": ["en"],
        "license": "mit",
        "library_name": "transformers",
        "tags": ["financial-analysis", "conversational", "finance", "qlora"],
        "pipeline_tag": "text-generation",
        "model-index": [
            {
                "name": model_info['name'],
                "results": []
            }
        ],
        "base_model": model_info['base_model']
    }
    
    # Add metrics if available
    if metrics:
        metadata["model-index"][0]["results"] = [
            {
                "task": {
                    "type": "text-generation",
                    "name": "Financial Advisory Generation"
                },
                "dataset": {
                    "type": "custom",
                    "name": "Financial Conversations"
                },
                "metrics": [
                    {
                        "type": "rouge1",
                        "value": f"{metrics.get('rouge1_improvement', '12.57%')}",
                        "name": "ROUGE-1 Improvement"
                    },
                    {
                        "type": "rouge2",
                        "value": f"{metrics.get('rouge2_improvement', '79.48%')}",
                        "name": "ROUGE-2 Improvement"
                    },
                    {
                        "type": "rougeL",
                        "value": f"{metrics.get('rougeL_improvement', '24.00%')}",
                        "name": "ROUGE-L Improvement"
                    },
                    {
                        "type": "bleu",
                        "value": f"{metrics.get('bleu_improvement', '135.36%')}",
                        "name": "BLEU Improvement"
                    }
                ]
            }
        ]
    
    return metadata

def update_metadata(
    repo_name: str,
    repo_dir: str = None,
    metrics_file: str = None, 
    token: str = None
) -> None:
    """
    Update metadata in an existing repository
    
    Args:
        repo_name: Name for HuggingFace repo (e.g. 'username/model-name')
        repo_dir: Path to local repository directory
        metrics_file: Path to CSV file containing model metrics
        token: HuggingFace API token. If None, will look for HUGGING_FACE_HUB_TOKEN env var
    """
    # Set HF token
    if token:
        HfFolder.save_token(token)
    elif not os.getenv("HUGGING_FACE_HUB_TOKEN"):
        raise ValueError("No HuggingFace token provided. Set HUGGING_FACE_HUB_TOKEN environment variable or use --token")
    
    actual_token = token if token else os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    try:
        with console.status("[bold blue]Updating repository metadata...", spinner="dots"):
            # Use existing repo directory if provided, or use default path
            if not repo_dir:
                # Default path as specified: /home/zahemen/projects/models/finsight-ai/
                repo_dir = Path("/home/zahemen/projects/models") / repo_name.split("/")[-1]
            
            repo_path = Path(repo_dir)
            if not repo_path.exists():
                raise ValueError(f"Repository directory does not exist: {repo_path}")
            
            # Load model info
            model_info = {
                "name": "FinSight AI",
                "description": "A fine-tuned version of SmolLM2-1.7B optimized for financial advice and discussion",
                "base_model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "task": "Financial Advisory and Discussion",
                "data_desc": "Curated dataset of 10,896 financial conversations (16.5M tokens)",
                "train_method": "QLoRA (4-bit quantization with LoRA)",
                "language": "English",
                "license": "MIT",
                "repo_id": repo_name
            }
            
            # Load metrics if available
            metrics = load_metrics(metrics_file)
            
            # Create metadata
            metadata = create_model_card_metadata(model_info, metrics)
            
            # Read existing README.md
            readme_path = repo_path / "README.md"
            if readme_path.exists():
                with open(readme_path, 'r') as f:
                    readme_content = f.read()
                
                # Remove any existing metadata section
                if readme_content.startswith('---'):
                    try:
                        # Find the end of the YAML section
                        second_separator = readme_content.find('---', 3)
                        if (second_separator != -1):
                            readme_content = readme_content[second_separator + 3:].strip()
                    except:
                        # If something goes wrong, keep the original content
                        pass
            else:
                # Create basic README if it doesn't exist
                readme_content = f"# {model_info['name']}\n\n{model_info['description']}"
            
            # Write updated README with metadata
            with open(readme_path, 'w') as f:
                f.write("---\n")
                yaml.dump(metadata, f, allow_unicode=True, sort_keys=False)
                f.write("---\n\n")
                f.write(readme_content)
            
            # Init repository handler
            repo = Repository(local_dir=str(repo_path))
            
            # Push changes
            repo.git_add("README.md")
            repo.git_commit("Update model card metadata")
            repo.git_push()
            
            logger.info(f"Successfully updated metadata for {repo_name}")
            
            # Show success message
            console.print(Panel.fit(
                f"[bold green]Success![/bold green]\n\n"
                f"Updated metadata for: [link=https://huggingface.co/{repo_name}]https://huggingface.co/{repo_name}[/link]",
                title="FinSight AI Metadata Update",
                border_style="green"
            ))
    
    except Exception as e:
        logger.error(f"Error updating metadata: {e}")
        raise

def push_to_hub(
    model_path: str,
    repo_name: str,
    commit_message: str = "Initial commit with fine-tuned model", 
    private: bool = False,
    token: str = None,
    create_local: bool = True,
    metrics_file: str = None,
    repo_local_dir: str = None
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
        metrics_file: Path to CSV file containing model metrics
        repo_local_dir: Path to existing local repository directory
    """
    if not Path(model_path).exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Set HF token
    if token:
        HfFolder.save_token(token)
    elif not os.getenv("HUGGING_FACE_HUB_TOKEN"):
        raise ValueError("No HuggingFace token provided. Set HUGGING_FACE_HUB_TOKEN environment variable or use --token")

    try:
        # Extract actual token for API calls
        actual_token = token if token else os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        with console.status("[bold green]Setting up repository...", spinner="dots"):
            # Create repo on HF Hub
            repo_url = create_repo(
                repo_id=repo_name,
                private=private,
                token=actual_token,
                exist_ok=True
            )
            
            # Load metrics if available
            metrics = load_metrics(metrics_file)
            
            # Setup local repo if requested
            if create_local:
                base_path = Path.cwd().parent.parent  # Get FinsightAI root dir
                setup_repo_structure(base_path)
                
                # Define model info for metadata
                model_info = {
                    "name": "FinSight AI",
                    "description": "A fine-tuned version of SmolLM2-1.7B optimized for financial advice and discussion",
                    "base_model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    "task": "Financial Advisory and Discussion",
                    "data_desc": "Curated dataset of 10,896 financial conversations (16.5M tokens)",
                    "train_method": "QLoRA (4-bit quantization with LoRA)",
                    "language": "English",
                    "license": "MIT",
                    "repo_id": repo_name
                }
                
                create_metadata_files(base_path, model_info, metrics)
                
                # Use provided repo dir or create one
                if repo_local_dir:
                    repo_dir = Path(repo_local_dir)
                    if not repo_dir.exists():
                        raise ValueError(f"Specified repository directory does not exist: {repo_dir}")
                    logger.info(f"Using existing repository at {repo_dir}")
                else:
                    # Create new repo directory
                    repo_dir = base_path / "models" / repo_name.split("/")[-1]
                    repo_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Clone repo if directory doesn't have .git folder
                    if not (repo_dir / ".git").exists():
                        logger.info(f"Cloning repository to {repo_dir}")
                        Repository(
                            local_dir=str(repo_dir),
                            clone_from=repo_url,
                            token=actual_token,
                            skip_lfs_files=True
                        )
                
                # Create repository instance
                repo = Repository(local_dir=str(repo_dir))
                
                # Create model card metadata
                metadata = create_model_card_metadata(model_info, metrics)
                
                # Copy model files to repo directory
                logger.info(f"Copying model files from {model_path}")
                for item in Path(model_path).glob("*"):
                    target_path = repo_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, target_path)
                    elif item.is_dir():
                        if target_path.exists():
                            shutil.rmtree(target_path)
                        shutil.copytree(item, target_path)
                
                # Create README with metadata
                with open(base_path / "README.md", 'r') as f:
                    readme_content = f.read()
                
                with open(repo_dir / "README.md", 'w') as f:
                    f.write("---\n")
                    yaml.dump(metadata, f, allow_unicode=True, sort_keys=False)
                    f.write("---\n\n")
                    f.write(readme_content)
                
                # Copy requirements
                shutil.copy2(base_path / "requirements.txt", repo_dir / "requirements.txt")
                
                # Push everything
                logger.info("Pushing repository...")
                repo.git_add(auto_lfs_track=True)
                repo.git_commit(commit_message)
                repo.git_push()
                
            else:
                # Just push the model directly
                logger.info(f"Loading model from {model_path}")
                model = AutoModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                logger.info(f"Pushing to {repo_name}")
                model.push_to_hub(repo_name, token=actual_token)
                tokenizer.push_to_hub(repo_name, token=actual_token)

        logger.info(f"Successfully pushed model to https://huggingface.co/{repo_name}")

        # Show success message
        console.print(Panel.fit(
            f"[bold green]Success![/bold green]\n\n"
            f"Model pushed to: [link=https://huggingface.co/{repo_name}]https://huggingface.co/{repo_name}[/link]",
            title="FinSight AI Deployment",
            border_style="green"
        ))

    except Exception as e:
        logger.error(f"Error pushing to hub: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Push FinSight AI model to Hugging Face Hub")
    
    # Main model push arguments
    push_group = parser.add_argument_group("Model Push Options")
    push_group.add_argument("--model_path", type=str, help="Path to saved model directory")
    push_group.add_argument("--repo_name", type=str, required=True, help="HuggingFace repo name (e.g. 'username/model-name')")
    push_group.add_argument("--commit_message", type=str, default="Initial commit with fine-tuned model", help="Commit message")
    push_group.add_argument("--private", action="store_true", help="Make repo private")
    push_group.add_argument("--token", type=str, help="HuggingFace token (optional if env var set)")
    push_group.add_argument("--local", action="store_true", help="Create local repository structure")
    push_group.add_argument("--metrics_file", type=str, help="Path to metrics CSV file")
    push_group.add_argument("--repo_dir", type=str, help="Path to existing local repository directory")
    
    # Metadata update only
    metadata_group = parser.add_argument_group("Metadata Update Options")
    metadata_group.add_argument("--update_metadata", action="store_true", help="Update only the metadata in an existing repository")
    
    args = parser.parse_args()
    
    if args.update_metadata:
        update_metadata(
            repo_name=args.repo_name,
            repo_dir=args.repo_dir,
            metrics_file=args.metrics_file,
            token=args.token
        )
    else:
        if not args.model_path:
            parser.error("--model_path is required when not using --update_metadata")
            
        push_to_hub(
            model_path=args.model_path,
            repo_name=args.repo_name,
            commit_message=args.commit_message,
            private=args.private,
            token=args.token,
            create_local=args.local,
            metrics_file=args.metrics_file,
            repo_local_dir=args.repo_dir
        )
