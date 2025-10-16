# Basic Translation Microservice
# This service will translate English text to Chinese

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug: Check if API key is loaded
api_key_loaded = os.getenv("OPENAI_API_KEY")
print(f"🔑 API Key loaded: {'✅ Yes' if api_key_loaded else '❌ No'}")
if api_key_loaded:
    print(f"🔑 API Key preview: {api_key_loaded[:10]}...{api_key_loaded[-4:]}")

# Step 3: Define data models
# These models define the structure of data our API expects and returns

class TranslateRequest(BaseModel):
    """
    Data model for translation requests
    Defines what data clients must send when calling /translate
    """
    text: str  # The English text to translate (required)

class TranslateResponse(BaseModel):
    """
    Data model for translation responses  
    Defines what data our API will return
    """
    success: bool           # Whether translation succeeded
    original_text: str      # The original English text
    translated_text: str    # The Chinese translation
    message: str = ""       # Optional status message

class SummarizeRequest(BaseModel):
    """
    Data model for summarization requests
    """
    text: str              # The text to summarize (required)
    max_points: int = 3    # Number of key points to extract (default: 3)

class SummarizeResponse(BaseModel):
    """
    Data model for summarization responses
    """
    success: bool          # Whether summarization succeeded
    original_text: str     # The original text
    summary_points: List[str]  # List of key points
    message: str = ""      # Optional status message

# Create the FastAPI application
app = FastAPI(
    title="Translation Service",
    description="A simple microservice that translates English to Chinese",
    version="1.0.0"
)

# Step 2: Add a health check endpoint
@app.get("/")
def read_root():
    """
    Health check endpoint
    Returns basic service information and status
    """
    return {
        "message": "AI Microservice is running!",
        "status": "healthy", 
        "endpoints": ["/translate", "/summarize"],
        "features": ["English to Chinese Translation", "Text Summarization"]
    }

# Step 4: Add the translate endpoint
@app.post("/translate")
def translate_text(request: TranslateRequest) -> TranslateResponse:
    """
    Translate English text to Chinese
    
    This endpoint receives English text and returns Chinese translation
    For now, we'll use simple mock translation logic
    """
    try:
        # Get the text to translate
        english_text = request.text
        
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            # Use OpenAI API for real translation via direct HTTP request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a professional English to Chinese translator. Translate the given English text to natural, fluent Chinese. Only return the Chinese translation, no explanations."
                    },
                    {
                        "role": "user", 
                        "content": f"Translate this English text to Chinese: {english_text}"
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                chinese_text = result["choices"][0]["message"]["content"].strip()
            else:
                chinese_text = f"[API Error {response.status_code}] {response.text}"
        else:
            # Fallback: Mock translation when no API key
            chinese_text = f"[Mock Translation] {english_text} → OPENAI_API_KEY not configured"
        
        return TranslateResponse(
            success=True,
            original_text=english_text,
            translated_text=chinese_text,
            message="Translation completed successfully"
        )
        
    except Exception as e:
        return TranslateResponse(
            success=False,
            original_text=request.text,
            translated_text="",
            message=f"Translation failed: {str(e)}"
        )

# Step 5: Add the summarize endpoint
@app.post("/summarize")
def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    """
    Summarize text into key points
    
    This endpoint receives text and returns key summary points
    """
    try:
        # Get the text to summarize
        text = request.text
        max_points = request.max_points
        
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            # Use OpenAI API for real summarization
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system", 
                        "content": f"You are a professional text summarizer. Extract exactly {max_points} key points from the given text. Return only the key points as a numbered list, no additional explanation."
                    },
                    {
                        "role": "user", 
                        "content": f"Summarize this text into {max_points} key points:\n\n{text}"
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                summary_text = result["choices"][0]["message"]["content"].strip()
                
                # Parse the numbered list into individual points
                summary_points = []
                for line in summary_text.split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                        # Remove numbering and clean up
                        clean_point = line.split('.', 1)[-1].strip() if '.' in line else line.strip()
                        clean_point = clean_point.lstrip('- •').strip()
                        if clean_point:
                            summary_points.append(clean_point)
                
                # Ensure we have the requested number of points
                if len(summary_points) < max_points:
                    summary_points.extend([f"Additional point {i+1}" for i in range(len(summary_points), max_points)])
                elif len(summary_points) > max_points:
                    summary_points = summary_points[:max_points]
                    
            else:
                summary_points = [f"[API Error {response.status_code}] {response.text}"]
        else:
            # Fallback: Mock summarization when no API key
            summary_points = [
                f"[Mock Summary] Key point 1 from: {text[:50]}...",
                f"[Mock Summary] Key point 2 from the text",
                f"[Mock Summary] Key point 3 - OPENAI_API_KEY not configured"
            ][:max_points]
        
        return SummarizeResponse(
            success=True,
            original_text=text,
            summary_points=summary_points,
            message="Summarization completed successfully"
        )
        
    except Exception as e:
        return SummarizeResponse(
            success=False,
            original_text=request.text,
            summary_points=[],
            message=f"Summarization failed: {str(e)}"
        )

# OpenAPI endpoint for Dify tool registration
@app.get("/openapi.json")
def get_openapi_spec():
    """
    Return OpenAPI specification for Dify tool registration
    """
    from fastapi.openapi.utils import get_openapi
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Translation Service",
        version="1.0.0", 
        description="Professional English to Chinese translation service",
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Run the service when this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)