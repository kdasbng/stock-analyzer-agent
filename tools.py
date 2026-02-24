"""Function tools that Gemini can call."""
import logging
from typing import Dict, Any
import yfinance as yf
from datetime import datetime
import ssl
import certifi

# Disable SSL verification for yfinance (for testing/corporate environments)
ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)


def get_stock_price(ticker: str) -> Dict[str, Any]:
    """
    Fetch current stock price and basic information for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        
    Returns:
        Dictionary with stock information or error message
    """
    try:
        logger.info(f"Fetching stock data for {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price and related data
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        previous_close = info.get('previousClose')
        
        if current_price is None:
            return {
                "error": f"Unable to fetch price data for ticker '{ticker}'. It may be invalid or delisted.",
                "ticker": ticker
            }
        
        # Calculate change percentage
        change_percent = None
        if previous_close and previous_close > 0:
            change_percent = ((current_price - previous_close) / previous_close) * 100
        
        return {
            "ticker": ticker.upper(),
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2) if previous_close else None,
            "change_percent": round(change_percent, 2) if change_percent else None,
            "volume": info.get('volume'),
            "market_cap": info.get('marketCap'),
            "company_name": info.get('longName') or info.get('shortName'),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return {
            "error": f"Failed to fetch data for '{ticker}': {str(e)}",
            "ticker": ticker
        }
