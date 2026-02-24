"""Configuration management for Stock Analyzer Agent."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables. "
        "Please create a .env file based on .env.example and add your API key."
    )

# Model Configuration
GEMINI_MODEL = 'gemini-2.5-flash'

# Function calling configuration
MAX_FUNCTION_CALL_ITERATIONS = 10  # Prevent infinite loops
