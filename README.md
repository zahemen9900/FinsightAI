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

<div align="center">
<table>
<tr>
  <td align="center">
    <a href="https://github.com/zahemen9900/FinsightAI.git">
      <img src="https://img.shields.io/badge/Training_Repo-181717?style=for-the-badge&logo=github&logoColor=white" alt="Training Repository"/>
      <br/>
      <strong>Project Repository</strong>
    </a>
  </td>
  <td align="center">
    <a href="https://colab.research.google.com/drive/1vpgADgpnlZ4wIAxOL79HDzAuQHG9RB-X?usp=sharing">
      <img src="https://img.shields.io/badge/Demo_on_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab Demo"/>
      <br/>
      <strong>Interactive Demo</strong>
    </a>
  </td>
</tr>
</table>
</div>

<div align="center">
  <h4><a href="https://github.com/zahemen9900/Datasets-for-Finsight/blob/97d7cacfff62e7b6099ef3bb0af9cf3d044a5b35/metrics/model_paper.md" target="_blank">Read Model Paper 📄</a></h4>
</div>


## 📋 Overview

FinSight AI is a specialized financial advisory assistant built by fine-tuning SmolLM2-1.7B-Instruct using QLoRA (Quantized Low-Rank Adaptation). The model has been trained on a comprehensive dataset of financial conversations to provide accurate, concise, and helpful information across various financial domains including personal finance, investing, market analysis, and financial planning.

Our evaluation demonstrates significant performance improvements across all standard NLP metrics **(ROUGE-1 , ROUGE-2, ROUGE-L & BLEU)**, showcasing the effectiveness of our domain-specific training approach. The model exhibits enhanced capabilities with richer financial terminology usage, more precise responses, improved handling of numerical data, and greater technical accuracy - all while maintaining a compact, resource-efficient architecture suitable for deployment on consumer hardware.

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

the details of the datasets used can be found [here](https://github.com/zahemen9900/Datasets-for-Finsight.git). The scripts for generating these datasets can be found at `src/alignment`.

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

The model was evaluated using standard NLP metrics across multiple financial datasets. More details [here](https://github.com/zahemen9900/Datasets-for-Finsight.git).

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
