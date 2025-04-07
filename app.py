from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from typing import Optional
import yfinance as yf
import tempfile
import pygame
from gtts import gTTS
import time
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="templates")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Stock symbol mappings for common voice inputs
STOCK_ALIASES = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "FACEBOOK": "META",
    "NETFLIX": "NFLX",
    "TESLA": "TSLA",
    "NVIDIA": "NVDA",
    "AMD": "AMD",
    "INTEL": "INTC"
}

# Cache for stock data to reduce API calls
STOCK_CACHE = {}
CACHE_TIMEOUT = 300  # seconds (5 minutes)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze-stock")
async def analyze_stock(stock_symbol: str = Form(...)):
    """Get stock data and analysis based on the provided stock symbol"""
    try:
        # Clean and standardize stock symbol
        stock_symbol = stock_symbol.upper().strip()
        
        # Check if it's a common name and convert to ticker
        if stock_symbol in STOCK_ALIASES:
            stock_symbol = STOCK_ALIASES[stock_symbol]
        
        # Fetch stock price with simplified approach
        stock_data = get_stock_data_simple(stock_symbol)
        if not stock_data:
            return JSONResponse(
                status_code=404,
                content={"error": f"Could not find stock data for {stock_symbol}"}
            )
        
        # Generate analysis using Gemini
        analysis = generate_stock_analysis(stock_symbol, stock_data)
        
        return {
            "symbol": stock_symbol.upper(),
            "price": stock_data["price"],
            "change": stock_data["change"],
            "change_percent": stock_data["change_percent"],
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error processing request for {stock_symbol}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing your request: {str(e)}"}
        )

def get_stock_data_simple(symbol: str) -> Optional[dict]:
    """
    Simplified approach to get stock data with caching and alternative methods
    """
    # Check cache first
    current_time = time.time()
    if symbol in STOCK_CACHE and (current_time - STOCK_CACHE[symbol]["timestamp"]) < CACHE_TIMEOUT:
        return STOCK_CACHE[symbol]["data"]
    
    # Add small random delay to avoid rate limits
    time.sleep(random.uniform(0.5, 2))
    
    # Try multiple methods
    stock_data = None
    
    # Method 1: Direct ticker.history() with minimal modules
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")  # Get 2 days to calculate change
        
        if not hist.empty and len(hist) >= 1:
            latest_price = hist['Close'].iloc[-1]
            
            # Calculate change
            if len(hist) >= 2:
                previous_close = hist['Close'].iloc[-2]
            else:
                previous_close = latest_price  # Fallback
            
            change = latest_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
            
            stock_data = {
                "price": latest_price,
                "change": change,
                "change_percent": f"{change_percent:.2f}%"
            }
    except Exception as e:
        logger.warning(f"Method 1 failed for {symbol}: {str(e)}")
    
    # Method 2: Alternative approach using direct API call if first method failed
    if not stock_data:
        try:
            # Use a more direct approach with requests
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                meta = data.get('chart', {}).get('result', [{}])[0].get('meta', {})
                price = meta.get('regularMarketPrice')
                previous_close = meta.get('previousClose')
                
                if price and previous_close:
                    change = price - previous_close
                    change_percent = (change / previous_close) * 100
                    
                    stock_data = {
                        "price": price,
                        "change": change,
                        "change_percent": f"{change_percent:.2f}%"
                    }
        except Exception as e:
            logger.warning(f"Method 2 failed for {symbol}: {str(e)}")
    
    # Store in cache if data was found
    if stock_data:
        STOCK_CACHE[symbol] = {
            "data": stock_data,
            "timestamp": current_time
        }
        return stock_data
    
    return None

def generate_stock_analysis(symbol: str, stock_data: dict) -> str:
    """Generate stock analysis using Google Gemini Flash 1.5"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f'Give a concise 2-3 line analysis of the stock {symbol}. Focus on recent performance, market sentiment, and short-term investment outlook. Keep it objective and informative.'
        
        # Add some context about the current price
        context = f"Current price: ${stock_data['price']:.2f}, Change: {stock_data['change']:.2f} ({stock_data['change_percent']})"
        full_prompt = f"{prompt}\n\nContext: {context}"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating analysis: {str(e)}")
        return "Unable to generate analysis at this time."

@app.post("/voice-stock")
async def voice_stock(transcription: str = Form(...)):
    """Process voice input to extract stock symbol"""
    try:
        # Log what we received
        logger.info(f"Voice input received: {transcription}")
        
        # Simple parsing logic - extract potential ticker symbols
        words = transcription.upper().split()
        
        # Common words to ignore
        ignore_words = ["THE", "STOCK", "PRICE", "OF", "FOR", "SHOW", "ME", "GET", "ANALYZE", "ANALYSIS"]
        
        # Filter out common words and keep potential stock symbols
        potential_symbols = [word for word in words if word not in ignore_words and word.isalpha()]
        
        if not potential_symbols:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not identify stock symbol in voice input"}
            )
        
        # Use the first potential symbol (this is simplistic and could be improved)
        stock_symbol = potential_symbols[0]
        
        # Check if it's a company name rather than a ticker
        if stock_symbol in STOCK_ALIASES:
            stock_symbol = STOCK_ALIASES[stock_symbol]
            
        logger.info(f"Identified stock symbol: {stock_symbol}")
        
        # Now proceed with analysis as in the regular endpoint
        stock_data = get_stock_data_simple(stock_symbol)
        if not stock_data:
            return JSONResponse(
                status_code=404,
                content={"error": f"Could not find stock data for {stock_symbol}"}
            )
        
        analysis = generate_stock_analysis(stock_symbol, stock_data)
        
        response_text = f"The current price of {stock_symbol} is ${stock_data['price']:.2f}, with a change of {stock_data['change_percent']}. {analysis}"
        
        # Generate speech for the response
        speech_file_path = speak_response(response_text)
        
        return {
            "symbol": stock_symbol,
            "price": stock_data["price"],
            "change": stock_data["change"],
            "change_percent": stock_data["change_percent"],
            "analysis": analysis,
            "speech_text": response_text,
            "speech_file": speech_file_path if speech_file_path else None
        }
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing your voice input: {str(e)}"}
        )

def speak_response(text):
    """
    Convert text to speech and play it using pygame
    
    Args:
        text (str): The text to convert to speech
        
    Returns:
        str: Path to the generated audio file (or None if error)
    """
    try:
        # Create a temporary file for the audio
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts.save(temp_audio_file.name)
            temp_file_path = temp_audio_file.name
        
        # Play the audio file
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)  # Check every 50ms for faster termination
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
        finally:
            pygame.mixer.quit()
            
        return temp_file_path
    except Exception as e:
        logger.error(f"Error in speech generation: {e}")
        return None

# Cleanup function to delete temporary files when the server shuts down
@app.on_event("shutdown")
def cleanup_temp_files():
    """Clean up any temporary files on server shutdown"""
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.endswith(".mp3") and os.path.isfile(os.path.join(temp_dir, file)):
            try:
                os.remove(os.path.join(temp_dir, file))
            except Exception as e:
                logger.error(f"Error removing temp file: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)