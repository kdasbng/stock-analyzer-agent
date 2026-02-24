"""Function tools that Gemini can call - DEMO VERSION with mock data."""
import logging
from typing import Dict, Any
from datetime import datetime
import random

logger = logging.getLogger(__name__)


def get_stock_price(ticker: str) -> Dict[str, Any]:
    """
    Fetch current stock price and basic information for a given ticker.
    **DEMO VERSION**: Returns mock data for demonstration purposes.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'NATIONALUM.NS')
        
    Returns:
        Dictionary with stock information
    """
    try:
        logger.info(f"Fetching stock data for {ticker}")
        
        # Mock stock data for demonstration
        base_prices = {
            'NATIONALUM.NS': 235.50,
            'POWERGRID.NS': 318.75,
            'TATASTEEL.NS': 165.20,
            'AAPL': 175.23,
            'MSFT': 420.15,
            'GOOGL': 142.67
        }
        
        # Get base price or generate random one
        base_price = base_prices.get(ticker.upper(), random.uniform(100, 500))
        change_percent = random.uniform(-5, 5)
        previous_close = base_price / (1 + change_percent/100)
        
        return {
            "ticker": ticker.upper(),
            "current_price": round(base_price, 2),
            "previous_close": round(previous_close, 2),
            "change_percent": round(change_percent, 2),
            "volume": random.randint(1000000, 50000000),
            "market_cap": random.randint(10000000000, 500000000000),
            "company_name": f"{ticker} Corporation",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return {
            "error": f"Failed to fetch data for '{ticker}': {str(e)}",
            "ticker": ticker
        }
