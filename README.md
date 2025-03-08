<div align="center">

# ⚡ FinSight AI

Your intelligent financial companion, powered by advanced AI.

<div align="center">
    <img src="https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/Transformers-4.34.0-409EFF?style=for-the-badge&logo=huggingface" alt="Transformers"/>
    <img src="https://img.shields.io/badge/Gradio-3.50.2-F37626?style=for-the-badge&logo=hexo" alt="Gradio"/>
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python" alt="Python"/>
</div>

</div>

## 📋 Overview

FinSight AI is a specialized financial advisory assistant built by fine-tuning SmolLM2-1.7B-Instruct using QLoRA (Quantized Low-Rank Adaptation). The model has been trained on a comprehensive dataset of financial conversations to provide accurate, concise, and helpful information across various financial domains including personal finance, investing, market analysis, and financial planning.

## 🏆 Key Results

- **Substantial Performance Improvements**:
  - 135.36% improvement in BLEU score
  - 79.48% improvement in ROUGE-2 score
  - 24.00% improvement in ROUGE-L score
  - 12.57% improvement in ROUGE-1 score

- **Enhanced Capabilities**:
  - 90.3% increase in financial terminology usage
  - More precise and structured responses
  - Better handling of numerical information
  - Improved technical accuracy in finance domains

## 🚀 Features

- **Domain Expert**: Trained specifically for financial advisory conversations
- **Efficient Architecture**: Small yet capable 1.7B parameter model
- **Resource-Friendly**: Designed to run on consumer hardware
- **Interactive UI**: Modern, responsive Gradio interface
- **Adaptive Responses**: Dynamically adjusts response length based on query complexity
- **Contextual Awareness**: Maintains conversation history for coherent multi-turn discussions
- **Financial Terminology**: Enhanced recognition and use of domain-specific financial terms

## 🧠 Technical Architecture

- **Base Model**: HuggingFaceTB/SmolLM2-1.7B-Instruct
- **Fine-tuning Method**: QLoRA (4-bit quantization)
- **LoRA Configuration**:
  - Rank: 64
  - Alpha: 16
  - Target modules: Query, Key, Value projections, MLP layers
- **Training Data**: 10,896 conversations (16.5M tokens)
- **Hardware Requirements**:
  - Training: Consumer GPU (tested on NVIDIA RTX 3050)
  - Inference: CPU or GPU with 6GB+ VRAM

## 📊 Training Dataset

The model was trained on a carefully curated dataset comprising:

| Dataset Type | Conversations | Tokens | Description |
|--------------|---------------|--------|-------------|
| Financial Introductions | 1,000 | 381K | Opening dialogues establishing financial advisory context |
| Reddit Finance | 4,542 | 7.0M | Filtered discussions from financial subreddits |
| Company-Specific Q&A | 1,354 | 1.0M | Financial information about specific companies |
| Financial Definitions | 2,000 | 1.2M | Explanations of financial terms and concepts |
| Finance Conversations | 2,000 | 7.0M | Detailed financial discussions on various topics |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/zahemen9900/FinsightAI.git
cd FinsightAI

# Create a conda environment (recommended)
conda env create -f conda_config/transformer_LM.yml

# Activate the environment
conda activate transformer_LM

# Install additional dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Running the Chat Interface

```bash
# Launch the Gradio web interface
python src/app/modern_chat.py

# For Streamlit interface (alternative)
python src/app/streamlit_chat.py
```

### Command-Line Chat

```bash
# For quick command-line interaction
python src/inference/chat_qlora.py --adapter_path="qlora_output"
```

### Training Your Own Model

```bash
# Generate and process dataset
python src/alignment/run_dataset_pipeline.py --all

# Run QLoRA fine-tuning
python src/main/train.py

# Evaluate model performance
python src/evaluation/compute_base_metrics.py
python src/evaluation/compute_qlora_metrics.py
python src/evaluation/visualize_metrics.py --base_dir="metrics/base_model_evaluation_results" --qlora_dir="metrics/qlora_evaluation_results"
```

## 📈 Performance Evaluation

The model was evaluated using standard NLP metrics across multiple financial datasets:

![Metrics Comparison](visualizations/radar_chart_20250306_121621.png)

For detailed metrics and visualizations, see the [research paper](metrics/research_paper.md) in this repository.

## 🔍 Example Interactions

**Query**: "What is dollar-cost averaging?"

**FinSight Response**:



## ⚙️ Advanced Configuration

The model offers several configuration options for inference:

- **Question Analysis**: Dynamically determines appropriate response length based on query complexity
- **Context Window**: Configurable history length for multi-turn conversations
- **Generation Parameters**: Adjustable temperature, top_p, and other parameters for response generation
- **Deploy Modes**: Options for CPU, GPU, or quantized inference for different hardware capabilities

## 🚧 Limitations

- Financial data and knowledge is current as of training data cutoff
- Not connected to the internet for real-time information
- Cannot provide personalized financial advice tailored to specific individual circumstances
- Model has not been extensively tested with non-English financial terminology

## 🤝 Contributing

Contributions are welcome! Areas that could benefit from community involvement:

- Data collection for additional financial domains
- Multilingual support for financial conversations
- Model optimizations for faster inference on mobile devices
- Extended evaluation on specialized financial use cases

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use FinSight AI in your research, please cite:

```
@misc{FinSightAI2025,
  author = {Zahemen, FinsightAI Team},
  title = {FinSight AI: Enhancing Financial Domain Performance of Small Language Models Through QLoRA Fine-tuning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zahemen9900/FinsightAI}}
}
```

## 🙏 Acknowledgments

- HuggingFace team for SmolLM2 and Transformers library
- QLoRA authors (Dettmers et al.) for the efficient fine-tuning technique
- Financial domain experts who validated response quality
- Open source community for tools and libraries used in this project
