# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
import torch
from transformers import pipeline

# Import the safe model loader
from app.model_utils import safe_load_model

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="API for sentiment analysis of Twitter tweets using BERT"
)

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Model configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "bert-base-uncased-sentiment-model")

# Load the sentiment model safely
classifier, error_message = safe_load_model(MODEL_NAME, device=device)

# Pydantic models
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str = None
    score: float = None
    input_text: str

# Sentiment labels mapping
SENTIMENT_LABELS = {
    "LABEL_0": "Very Negative",
    "LABEL_1": "Negative",
    "LABEL_2": "Neutral",
    "LABEL_3": "Positive",
    "LABEL_4": "Very Positive"
}

@app.post("/api/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze the sentiment of the provided text."""
    if not classifier:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        # Perform prediction
        prediction = classifier(request.text)
        
        # Extract the result
        if prediction and len(prediction) > 0:
            result = prediction[0]
            label = result.get("label", "")
            score = result.get("score", 0.0)
            
            # Map to human-readable sentiment
            sentiment = SENTIMENT_LABELS.get(label, label)
            
            return {
                "sentiment": sentiment,
                "score": score,
                "input_text": request.text
            }
        
        raise HTTPException(status_code=500, detail="Failed to get prediction")
            
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": classifier is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)