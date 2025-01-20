# FinsightAI 


# [![Python 3.11](https://img.shields.io/badge/PYTHON-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PYTORCH-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-SmolLM2-orange)](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](LICENSE)

A financial advisory chatbot powered by SmolLM2-1.7B, fine-tuned on curated Reddit financial discussions using QLoRA for efficient training.

## 🌟 Features

- Fine-tuned on 250k+ high-quality financial discussions from Reddit
- Efficient training using QLoRA (4-bit quantization with LoRA)
- Optimized for financial domain knowledge and advice
- Memory efficient implementation for consumer hardware

## 🛠️ Tech Stack

- 🤗 SmolLM2-1.7B Base Model
- 🔧 PyTorch
- 📚 TRL (Transformer Reinforcement Learning)
- 🎯 PEFT & BitsAndBytes for efficient fine-tuning
- 📊 WandB for experiment tracking

## 🚀 Getting Started

### Prerequisites

```bash
python 3.11
pytorch 2.0+
transformers
peft
trl
bitsandbytes
```

### Installation

```bash
git clone https://github.com/yourusername/FinsightAI.git
cd FinsightAI
pip install -r requirements.txt
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

