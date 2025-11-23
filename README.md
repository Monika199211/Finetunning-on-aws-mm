# Fine-tuning LLM on AWS SageMaker with RAG System

This project demonstrates end-to-end fine-tuning of a TinyLlama model for pharmaceutical instruction-following tasks using AWS SageMaker, with complete deployment pipeline and RAG (Retrieval-Augmented Generation) integration.

## Project Structure
```
Finetunning-on-aws-mm/
├── scripts/
│   └── train.py                 # Training script for SageMaker
├── inference/
│   └── inference.py             # Custom inference script for SageMaker deployment
├── finetuning_experiments/
│   └── experiment.ipynb         # Local fine-tuning experiments
├── estimator_launcher.ipynb     # SageMaker training job launcher
├── deployment_of_model.ipynb    # Model deployment to SageMaker endpoint
├── inference_app.py             # Streamlit app for direct model inference
├── rag_app_ui.py               # RAG system Streamlit UI
├── rag_app_backend.py          # RAG system backend with vector search
├── lambda_function.py          # AWS Lambda for API Gateway integration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── README.md                   # Project documentation
```

## Features
- ✅ Fine-tune TinyLlama-1.1B for pharmaceutical Q&A
- ✅ Parameter Efficient Fine-Tuning (LoRA) implementation  
- ✅ AWS SageMaker training and deployment pipeline
- ✅ Custom inference script for optimized model serving
- ✅ API Gateway + Lambda integration for REST API access
- ✅ RAG system with vector search (FAISS/ChromaDB)
- ✅ Multiple Streamlit interfaces for testing
- ✅ OpenAI fallback for timeout handling
- ✅ S3 integration for datasets and model artifacts
- ✅ Comprehensive error handling and logging

## Quick Start

### 1. Installation
```bash
# Clone and install
git clone <repository-url>
cd Finetunning-on-aws-mm
pip install -r requirements.txt

# macOS fix for OpenMP conflicts
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 2. Environment Setup
Create `.env` file:
```bash
API_URL="https://your-api-gateway-url/prod/predict"
API_KEY="your-api-gateway-key"
OPENAI_API_KEY="sk-your-openai-key"
```

### 3. Run RAG System
```bash
streamlit run rag_app_ui.py --server.port 8502
```

### 4. Test Direct Model Inference  
```bash
streamlit run inference_app.py --server.port 8501
```

## Training & Deployment Pipeline

### Local Training
```bash
# Open and run experiment notebook
jupyter notebook finetuning_experiments/experiment.ipynb
```

### SageMaker Training
1. Configure AWS credentials
2. Update IAM role in `estimator_launcher.ipynb` 
3. Run training job (outputs to S3)

### Model Deployment
1. Open `deployment_of_model.ipynb`
2. Deploy to SageMaker endpoint with custom inference script
3. Test endpoint directly in notebook

### API Integration
- Lambda function: `lambda_function.py`
- API Gateway with 29s timeout
- DynamoDB logging (optional)

## Architecture

```
User Input → RAG UI → Vector Search → Context Retrieval
                ↓
        API Gateway → Lambda → SageMaker Endpoint → Fine-tuned Model
                ↓
        Response (with fallback to OpenAI if timeout)
```

## Dataset & Model
- **Dataset**: Pharmaceutical Q&A (S3: `s3://llm-finetune-dataset-monika/`)
- **Base Model**: TinyLlama-1.1B-intermediate-step-1431k-3T
- **Method**: LoRA fine-tuning (rank=8, lr=2e-5)
- **Output**: Deployed on `ml.g5.xlarge` (GPU) or `ml.m5.xlarge` (CPU)

## Key Components

### RAG System (`rag_app_backend.py`)
- Vector embeddings with OpenAI text-embedding-3-small
- FAISS/ChromaDB for similarity search
- Pharmaceutical knowledge base with 8 curated documents
- Context retrieval + fine-tuned model generation

### Custom Inference (`inference/inference.py`)  
- Optimized for SageMaker deployment
- Handles multiple input formats
- Fast text generation with configurable parameters
- Proper error handling and logging

### Lambda Integration (`lambda_function.py`)
- Processes API Gateway requests
- Invokes SageMaker endpoint
- Logs requests to DynamoDB
- Comprehensive error handling

## Troubleshooting

### Common Issues
```bash
# OpenMP conflicts (macOS)
export KMP_DUPLICATE_LIB_OK=TRUE

# API Gateway timeout (504)
# → Use GPU instance (ml.g5.xlarge) 
# → Or enable OpenAI fallback

# SageMaker endpoint errors (400)  
# → Ensure inference/inference.py exists
# → Check CloudWatch logs

# Environment variables
# → Verify .env file format
```

### Performance Tips
- **GPU instances** (ml.g5.xlarge) for <10s response times
- **CPU instances** (ml.m5.xlarge) may timeout (30s+) 
- **Reduce max_tokens** in inference script for faster responses
- **Enable OpenAI fallback** for production reliability

## AWS Services Used
- **SageMaker**: Training, inference endpoints
- **S3**: Dataset and model storage  
- **Lambda**: API backend processing
- **API Gateway**: REST API interface
- **DynamoDB**: Request logging
- **CloudWatch**: Monitoring and logs

## API Usage
```bash
# Direct API call
curl -X POST \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_KEY" \
  -d '{"inputs": "What are metformin side effects?"}' \
  https://your-api-gateway/prod/predict
```

## Requirements
- **Python**: 3.8+
- **AWS CLI**: Configured with appropriate permissions
- **Dependencies**: See `requirements.txt`
- **Platform**: macOS, Linux (Windows via WSL)

## Author
**Monika Badadhe**  
End-to-end ML engineer specializing in LLM fine-tuning and deployment

## License
MIT License

---
*A complete production-ready pipeline for fine-tuning and serving language models on AWS with RAG capabilities.*