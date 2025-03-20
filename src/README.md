# FinSight AI - Source Code Overview

This directory contains the source code for the FinSight AI project, a specialized financial advisory assistant built by fine-tuning SmolLM2-1.7B-Instruct using QLoRA.

## Project Structure

The codebase is organized into the following main directories:

- `alignment/`: Data preparation and dataset generation
- `main/`: Core training and model management functionality
- `inference/`: Inference and user interaction features
- `evaluation/`: Metrics and model evaluation tools
- `app/`: User interfaces (Gradio and Streamlit)

## Data Preparation (`alignment/`)

The alignment directory contains scripts for dataset generation and processing:

### Financial Dataset Generation

- `run_dataset_pipeline.py`: Master script that orchestrates the entire dataset generation process
- `extract_financial_definitions.py`: Extracts financial definitions from various sources
- `prepare_definition_dataset.py`: Transforms definitions into conversation format
- `extract_finance_conversations.py`: Processes general financial Q&A conversations
- `extract_advanced_finance_conversations.py`: Handles complex financial conversations with proper formatting
- `create_intro_dataset.py`: Generates introduction/greeting conversational data
- `prepare_company_qa.py`: Processes company-specific financial Q&A data
- `prepare_reddit_data.py`: Cleans and formats Reddit financial discussions

### Data Processing Features

- **Currency Diversification**: Conversations can be modified to use diverse currencies (dollars, euros, pounds, cedis) while maintaining financial term integrity
- **Company-Specific Handling**: Special handling for company-related questions to ensure currency references remain in dollars
- **Error Introduction**: Deliberate introduction of typos and formatting errors to enhance model robustness
- **Dataset Merging**: Tools to combine multiple dataset sources while preserving metadata

## Training (`main/`)

- `train.py`: Core training script implementing QLoRA fine-tuning
- `push_model.py`: Utilities for pushing trained models to Hugging Face Hub
- Data loading and processing utilities for the training pipeline

### Training Features

- **QLoRA Implementation**: 4-bit quantization with Low-Rank Adaptation
- **LoRA Configuration**: 
  - Rank: 64
  - Alpha: 16
  - Target modules: Query, Key, Value projections, MLP layers
- **DeepSpeed Integration**: Optimization for memory efficiency and speed

## Inference (`inference/`)

- `chat_qlora.py`: Command-line inference for interacting with the model
- Helper utilities for loading and configuring the model for inference

### Inference Features

- **Response Length Control**: Dynamic control based on question complexity
- **Conversational Context**: Maintains conversation history for coherent multi-turn discussions
- **Generation Parameters**: Configurable temperature, top_p, etc.

## Evaluation (`evaluation/`)

- `compute_base_metrics.py`: Calculate baseline metrics on the base model
- `compute_qlora_metrics.py`: Calculate metrics on the fine-tuned model
- `extract_metrics.py`: Extract and analyze performance metrics
- `visualize_metrics.py`: Generate visualizations of performance improvements

### Evaluation Metrics

- **ROUGE scores**: For measuring text overlap quality
- **BLEU scores**: For precision-oriented evaluation
- **Financial terminology usage**: Domain-specific evaluation
- **Response structural analysis**: Metrics for formatting and organization

## User Interfaces (`app/`)

- `modern_chat.py`: Gradio-based chat interface
- `streamlit_chat.py`: Streamlit-based alternative interface

## Key Achievements

1. **Dataset Creation**: Generated 10,896 high-quality financial conversations (16.5M tokens)
2. **Domain Specialization**: Successfully adapted model for finance-specific knowledge
3. **Performance Improvements**:
   - ROUGE-1: 10.37% improvement
   - ROUGE-2: 42.93% improvement
   - ROUGE-L: 17.94% improvement
   - BLEU: 68.43% improvement
4. **Resource Efficiency**: Full training on consumer-grade hardware (NVIDIA RTX 3050)

## Usage Examples

### Dataset Generation

```bash
# Generate all dataset components and merge
python alignment/run_dataset_pipeline.py --all

# Generate specific components
python alignment/run_dataset_pipeline.py --intro --finance-convos --merge
```

### Model Training

```bash
# Run QLoRA fine-tuning
python main/train.py

# Push model to Hugging Face Hub
python main/push_model.py --model_path="qlora_output" --repo_name="username/model-name"
```

### Evaluation

```bash
# Run evaluation pipeline
python evaluation/compute_base_metrics.py
python evaluation/compute_qlora_metrics.py
python evaluation/visualize_metrics.py
```

### Interactive Usage

```bash
# Launch Gradio interface
python app/modern_chat.py

# Use command-line interface
python inference/chat_qlora.py --adapter_path="qlora_output"
```

## Technical Features

- **Dollar-Cost Averaging Fix**: Special handling to preserve financial terms like "dollar-cost averaging" during currency conversions
- **Multi-turn Conversation Generation**: Sophisticated algorithms to create coherent financial dialogues
- **Company-Specific Knowledge**: Enhanced handling of company-related financial discussions
- **Metadata Preservation**: Rich metadata for tracking conversation characteristics

## Future Work

- Expansion to multilingual financial contexts
- Temporal adaptation for evolving financial information
- Specialized sub-domain adaptations (insurance, real estate, cryptocurrency)
- Enhanced evaluation with expert human feedback

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{FinSightAI2025,
  author = {Zahemen, FinsightAI Team},
  title = {FinSight AI: Enhancing Financial Domain Performance of Small Language Models Through QLoRA Fine-tuning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zahemen9900/FinsightAI}}
}
```
