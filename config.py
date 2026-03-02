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

# Groq API Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "GROQ_API_KEY not found. Peer comparison features will be disabled."
    )

# Groq Model Selection
GROQ_MODEL = 'llama-3.3-70b-versatile'

# Feature Flags
ENABLE_PEER_COMPARISON = True
MAX_PEERS_TO_ANALYZE = 3

# Comparison Time Periods
COMPARISON_PERIODS = ['1Y', '2Y', '3Y']

# ── LangChain Settings ──────────────────────────────────────────
LANGCHAIN_VERBOSE = False  # Set True for debug chain traces

# Gemini via LangChain (Concept 1: ChatModel config)
GEMINI_MODEL_LANGCHAIN = 'gemini-2.5-flash'
GEMINI_TEMPERATURE = 0.3

# Groq via LangChain (Concept 1: ChatModel config)
GROQ_MODEL_LANGCHAIN = 'llama-3.3-70b-versatile'
GROQ_TEMPERATURE_PEERS = 0.2       # Deterministic for peer identification
GROQ_TEMPERATURE_ANALYSIS = 0.4    # Slightly creative for narratives
