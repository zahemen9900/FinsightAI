# QLoRA Training Methodology Documentation

This document outlines the methodology and design decisions used in implementing the QLoRA (Quantized Low-Rank Adaptation) training pipeline for FinSight AI.

## 1. Model Configuration and Architecture

### Base Model Selection
- Using Large Language Models optimized for instruction following
- Support for Flash Attention 2 for improved performance
- Models tested: Llama-2, Mistral, etc.

### QLoRA Parameters
- **LoRA Rank (r)**: 64
  - Higher rank for better fine-tuning capacity
  - Balanced against memory constraints
- **LoRA Alpha**: 16
  - Scaling factor for adapters
  - Optimized for financial domain adaptation
- **Dropout**: 0.1
  - Prevents overfitting while maintaining learning capacity

### Target Modules
```python
target_modules = [
    "q_proj", "k_proj", "v_proj",
    "o_proj", "gate_proj", 
    "up_proj", "down_proj"
]
```
- Comprehensive coverage of transformer components
- Balanced parameter efficiency and adaptation capability

## 2. Memory Optimizations

### Quantization Strategy
- 4-bit NormalFloat quantization
- Double quantization enabled
- BitsAndBytes configuration for efficient memory usage

### Memory Management
1. **Gradient Checkpointing**
   - Enabled by default
   - Optimized for memory efficiency
   - Configured with `use_reentrant=False`

2. **Flash Attention**
   - Utilized when available
   - Significant speed improvements
   - Reduced memory footprint

3. **Memory Efficient Optimizations**
   - Empty CUDA cache before training
   - Proper pin memory configuration
   - Optimized batch sizes

## 3. Training Configuration

### Hyperparameters
- **Learning Rate**: 6e-5
  - Higher than typical for faster convergence, and suitable for relatively smaller datasets (_~20k samples combined_)
  - Balanced against stability
- **Batch Size**: 4
  - Optimized for 24GB GPU
  - Gradient accumulation steps: 2
- **Training Epochs**: 1
  - Single pass through dataset
  - Early stopping enabled

### Learning Rate Schedule
- Linear scheduler
- Warmup ratio: 0.03
- Optimized for quick adaptation

### Evaluation Strategy
```python
eval_strategy = {
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 25
}
```

## 4. Dataset Integration

### Data Loading
1. **Multiple Sources**
   - Reddit finance (67%)
   - Company Q&A (100%)
   - Intro conversations (100%)

2. **Processing Pipeline**
   - Parallel processing
   - Efficient tokenization
   - Dynamic batching

### Dataset Merging Strategy
1. **Proportional Sampling**
   - Maintains dataset balance
   - Preserves domain expertise
   - Ensures conversation diversity

2. **Quality Controls**
   - Sample validation
   - Format verification
   - Content checking

## 5. Training Process

### Initialization
1. **Model Setup**
   - Load base model
   - Apply quantization
   - Configure LoRA adapters

2. **Optimizer Configuration**
   - AdamW optimizer
   - Weight decay: 0.0
   - Gradient clipping: 1.0

### Training Loop
1. **Performance Monitoring**
   - Loss tracking
   - Gradient norms
   - Memory usage
   - Training speed

2. **Checkpointing**
   - Regular model saves
   - Best model tracking
   - Resume capability

### Early Stopping
```python
early_stopping = {
    "patience": 3,
    "threshold": 0.05
}
```

## 6. DeepSpeed Integration

### Configuration
```python
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    }
}
```

### Optimization Features
- ZeRO Stage-2 optimization
- CPU offloading
- Gradient accumulation
- Automatic batch size scaling

## 7. Validation and Monitoring

### Training Metrics
- Loss tracking
- Perplexity
- Learning rate
- Gradient statistics

### Quality Checks
1. **Model Outputs**
   - Response quality
   - Financial accuracy
   - Conversation coherence

2. **Performance Metrics**
   - Training speed
   - Memory usage
   - GPU utilization

## 8. Error Handling and Logging

### Logging Strategy
- Rich logging integration
- Detailed progress tracking
- Error traceability
- Performance monitoring

### Recovery Mechanisms
- Checkpoint resumption
- Error state handling
- Graceful degradation

## Future Improvements

1. **Performance Optimizations**
   - Flash Attention 2 integration
   - Better memory management
   - Faster tokenization

2. **Training Enhancements**
   - Dynamic batch sizing
   - Adaptive learning rates
   - Improved early stopping

3. **Quality Improvements**
   - Better validation metrics
   - Enhanced monitoring
   - Automated quality checks
