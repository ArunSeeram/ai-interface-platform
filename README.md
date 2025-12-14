
## ML Component 

A PyTorch bidirectional LSTM model for detecting fake news. Classifies news articles as **FAKE** or **REAL** with ~92-95% accuracy.
The trained model is served using FastAPI for real-time inference.

### Overview

**Model**: Bidirectional LSTM with 2 layers (vocab: 10k words, embedding: 128-dim, hidden: 64-dim)
**Deployment**: FastAPI-based REST API

**Features**:
- GPU acceleration (CUDA support)
- Custom vocabulary building
- Early stopping with best model checkpointing
- Comprehensive evaluation metrics
- FastAPI-based real-time prediction service

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Vocab Size | 10,000 |
| Embedding Dim | 128 |
| Hidden Dim | 64 |
| Layers | 2 (bidirectional) |
| Dropout | 0.3 |
| Max Length | 300 tokens |
| Batch Size | 32 |
| Epochs | 20 |

### Performance

- **Accuracy**: ~92-95%
- **Precision**: ~0.93-0.96
- **Recall**: ~0.91-0.94
- **F1-Score**: ~0.92-0.95

### Dataset Format

CSV file with columns:
- `text`: News article content
- `label`: "FAKE" or "REAL"

Data split: 80% train (64% train, 16% validation), 20% test

### Notes

- Requires same vocabulary for training and inference
- Use `best_model.pt` over `fake_news_lstm_pytorch.pt`
- GPU recommended but not required
- Text preprocessing: lowercase, remove URLs/emails/special chars
- FastAPI is used to expose the model as a RESTful API for prediction

