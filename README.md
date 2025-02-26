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

## 🚀 Features

- **Fast and Efficient**: Optimized QLoRA training with mixed precision and DeepSpeed integration
- **Interactive Chat Interface**: Modern, responsive UI with Gradio
- **Smart Dataset Processing**: Advanced filtering and cleaning of financial conversations
- **Memory Efficient**: Optimized for both training and inference
- **Rich Conversation History**: Support for multi-turn financial discussions
- **Enhanced Response Formatting**: Improved readability with automatic paragraph breaks

## 🛠 Recent Updates

- **Improved Response Formatting**: Added intelligent paragraph grouping for enhanced readability
- **Advanced Data Cleaning**: Enhanced cleaning pipeline with sophisticated financial relevance scoring
- **Cross-Company Conversations**: Added support for comparative financial discussions across companies
- **List-Based Responses**: Implemented structured list formatting in responses for better organization
- **Expanded Financial Dictionary**: Enhanced keyword recognition with 600+ domain-specific terms
- **Comprehensive Documentation**: Updated data cleaning methodology with detailed technical specifications
- Added proportion-based dataset merging for better control over training data
- Improved model response conciseness with enhanced generation parameters
- Optimized training speed with ZeRO-2 and efficient batch sizes
- Enhanced UI with gradient header and modern styling
- Added flash attention support for faster inference
- Improved memory management during both training and inference

## 🏗️ Dataset Features

- **Rich Financial Content**: Expertly curated financial conversations from multiple sources
- **Multi-Turn Interactions**: Natural conversational flow with multi-turn context awareness
- **Enhanced Readability**: Automatic sentence grouping for improved text structure
- **Cross-Domain Expertise**: Coverage of stocks, crypto, personal finance, and market analysis
- **Structured Information**: List-based formatting for complex financial explanations
- **Company-Specific Knowledge**: Dedicated QA pairs for company financial insights
- **Natural Conversation Flow**: Professionally designed greeting and introduction templates

## 🔧 Installation

```bash
git clone https://github.com/zahemen9900/FinsightAI.git
cd FinsightAI
#create a custom conda environment for the project (recommended)
conda env create -f conda_config/transformer_LM.yml                                                                                                           
```

## 💻 Usage

### Data Preparation
```bash
python src/alignment/prepare_finetune_data.py
```

### Training
```bash
# Full precision training
python src/main/train.py

# QLoRA training (recommended)
python src/main/train_qlora.py
```

## 📊 Training Details

- **Base Model**: HuggingFaceTB/SmolLM2-1.7B-Instruct
- **Training Data**: Curated financial discussions from 43 subreddits
- **Optimization**: QLoRA with 4-bit quantization
- **Training Parameters**:
  - Learning Rate: 2e-4
  - Batch Size: 2 (effective 16 with gradient accumulation)
  - LoRA Rank: 64
  - LoRA Alpha: 16
  - Training Steps: 1000

## 📈 Performance

*Coming soon - Model evaluation metrics and comparisons*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace team for SmolLM2
- Reddit financial communities for training data
- QLoRA paper authors for efficient fine-tuning techniques

