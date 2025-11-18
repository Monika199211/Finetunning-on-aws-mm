
# Fine-tuning LLM on AWS SageMaker

This project demonstrates how to fine-tune a TinyLlama model for pharmaceutical instruction-following tasks using AWS SageMaker.

## Project Structure
```
Finetunning-on-aws-mm/
├── script/
│   └── train.py                 # Training script for SageMaker
├── finetuning_experiments/
│   └── experiment.ipynb         # Local fine-tuning experiments
├── estimator_launcher.ipynb     # SageMaker training job launcher
├── inference_app.py             # Streamlit app for model inference
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Features
- Fine-tune TinyLlama-1.1B for pharmaceutical Q&A
- Parameter Efficient Fine-Tuning (LoRA) implementation
- AWS SageMaker integration for scalable training
- S3 integration for dataset and model storage
- Streamlit web app for model inference

## Installation
```bash
# Install dependencies
pip install -r requirements.txt

# For UV users (if uv is installed)
uv init --no-workspace
uv add -r requirements.txt
```

## Usage

### Run Inference App
```bash
streamlit run inference_app.py
```

### Local Training
1. Open `finetuning_experiments/experiment.ipynb`
2. Run cells sequentially to train locally

### SageMaker Training
1. Configure your AWS credentials
2. Update the IAM role in `estimator_launcher.ipynb`
3. Run the estimator to start SageMaker training job

## Dataset
The project uses pharmaceutical instruction data stored in S3:
- Bucket: `s3://llm-finetune-dataset-monika/datasets/`
- Format: CSV with instruction, input, output columns

## Model Configuration
- Base Model: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- LoRA rank: 8
- Learning rate: 2e-5
- Training epochs: 2-3
- Batch size: 1-2

## AWS Setup
1. Create SageMaker execution role with appropriate permissions
2. Configure S3 buckets for data and model artifacts
3. Update role ARN in the launcher notebook
4. Ensure proper IAM permissions for SageMaker and S3

## Supported Platforms
- ✅ macOS (Apple Silicon)
- ✅ Linux (x86_64)
- ⚠️ Windows (limited support for some dependencies)

## Notes
- Uses CPU/MPS for local training on macOS
- GPU instances recommended for SageMaker training
- bitsandbytes not supported on macOS - removed from requirements

## Troubleshooting

### Common Issues
1. **UV init error**: Use `uv init --no-workspace` to avoid workspace detection issues
2. **bitsandbytes installation**: Skip on macOS, only needed for quantization
3. **Transformers version**: Use version 4.37.0 for SageMaker compatibility
4. **Git authentication**: Generate Personal Access Token for GitHub operations

### Version Compatibility
- transformers==4.37.0 (SageMaker compatible)
- torch>=1.13.0
- datasets>=2.19.1
- peft>=0.11.1

## Author
Monika Badadhe

## License
MIT License