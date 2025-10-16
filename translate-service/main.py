# Basic Translation Microservice
# This service will translate English text to Chinese

from fastapi import FastAPI
from pydantic import BaseModel
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
        "message": "Translation Service is running!",
        "status": "healthy", 
        "endpoints": ["/translate"]
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