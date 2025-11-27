import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Literal
from uvicorn import run as app_run

from src.pipline.prediction_pipeline import PredictionPipeline
from src.constants import APP_HOST, APP_PORT
from src.exception import MyException
from src.logger import logging

# Initialize the FastAPI application
app = FastAPI(title="AI News Classifier Backend")

# Setup Jinja2 for rendering HTML pages
templates = Jinja2Templates(directory="templates")

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the prediction pipeline (loaded once at startup)
try:
    logging.info("Initializing PredictionPipeline for FastAPI app...")
    prediction_pipeline = PredictionPipeline()
    logging.info("PredictionPipeline initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize PredictionPipeline: {e}", exc_info=True)
    prediction_pipeline = None

# Category mapping: class index -> category name
CATEGORY_MAPPING = {
    1: "World",
    2: "Sport",
    3: "Business",
    4: "Tech"
}

# Define the possible output topics
ClassificationTopic = Literal["World", "Sport", "Business", "Tech"]


# --- Pydantic Models for API ---
class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    topic: ClassificationTopic


# --- GET Endpoint for Root Path (Home Page) ---
@app.get("/", tags=["Home Page"], response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Handles the root path and serves the home page (index.html).
    """
    return templates.TemplateResponse("index.html", {"request": request})


# --- GET Endpoint for Classifier Page ---
@app.get("/classifier", tags=["Classifier Page"], response_class=HTMLResponse)
async def show_classifier_page(request: Request):
    """
    Handles the request for the classifier page and serves classifier.html.
    """
    return templates.TemplateResponse("classifier.html", {"request": request})


# --- API Endpoint for Prediction ---
@app.post("/classify", response_model=PredictionOutput, tags=["API"])
def classify_text(input_data: TextInput):
    """
    Analyzes the input text and predicts its news category using the trained model.
    
    Args:
        input_data: JSON body with 'text' field containing the news title/description
        
    Returns:
        PredictionOutput with 'topic' field (World, Sport, Business, or Tech)
    """
    try:
        if prediction_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction service is not available. Model not loaded."
            )
        
        # Validate input
        text = input_data.text.strip()
        if len(text) < 5:
            raise HTTPException(
                status_code=400,
                detail="Text input must be at least 5 characters long."
            )
        
        # Get prediction from the pipeline
        logging.info(f"Processing prediction request for text: {text[:50]}...")
        predictions = prediction_pipeline.predict(text)
        
        if not predictions or len(predictions) == 0:
            raise HTTPException(
                status_code=500,
                detail="Model prediction returned empty result."
            )
        
        # Get the first prediction (class index: 1, 2, 3, or 4)
        class_index = predictions[0]
        
        # Map class index to category name
        topic = CATEGORY_MAPPING.get(class_index, "World")  # Default to "World" if mapping fails
        
        logging.info(f"Prediction successful: Class Index {class_index} -> {topic}")
        
        return PredictionOutput(topic=topic)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except MyException as e:
        # Handle custom exceptions from the pipeline
        logging.error(f"Custom exception in classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction pipeline error: {str(e)}")
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"Unexpected error during classification: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal prediction error: {str(e)}"
        )

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
