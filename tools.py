"""Function tools that Gemini can call."""
import logging
from typing import Any, Dict, List, Optional
import yfinance as yf
from datetime import datetime, timedelta
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


def get_stock_financials(ticker: str) -> Dict[str, Any]:
    """
    Fetch detailed financial metrics for a stock.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with PE ratios, revenue growth, profit margin, market cap,
        sector, industry, and company name.
    """
    try:
        logger.info(f"Fetching financials for {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info

        # Revenue YoY growth from annual financials
        revenue_yoy_growth: Optional[float] = None
        try:
            financials = stock.financials  # columns = years, rows = line items
            if financials is not None and not financials.empty:
                # Row label may vary: 'Total Revenue' or 'Revenue'
                rev_row = None
                for label in ['Total Revenue', 'Revenue']:
                    if label in financials.index:
                        rev_row = financials.loc[label]
                        break
                if rev_row is not None and len(rev_row) >= 2:
                    latest = rev_row.iloc[0]
                    previous = rev_row.iloc[1]
                    if previous and previous != 0:
                        revenue_yoy_growth = round(((latest - previous) / abs(previous)) * 100, 2)
        except Exception as e:
            logger.debug(f"Could not compute revenue growth for {ticker}: {e}")

        return {
            "ticker": ticker.upper(),
            "company_name": info.get('longName') or info.get('shortName'),
            "sector": info.get('sector'),
            "industry": info.get('industry'),
            "pe_ratio_trailing": info.get('trailingPE'),
            "pe_ratio_forward": info.get('forwardPE'),
            "revenue_yoy_growth": revenue_yoy_growth,
            "profit_margin": _safe_pct(info.get('profitMargins')),
            "market_cap": info.get('marketCap'),
        }
    except Exception as e:
        logger.error(f"Error fetching financials for {ticker}: {e}")
        return {"error": str(e), "ticker": ticker}


def get_stock_historical_performance(ticker: str) -> Dict[str, Any]:
    """
    Calculate stock price returns over 1-year, 2-year, and 3-year periods.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with return_1y, return_2y, return_3y (percentages)
        and current_price.
    """
    try:
        logger.info(f"Fetching historical performance for {ticker}")
        stock = yf.Ticker(ticker)

        # Fetch up to 3 years + a small buffer of history
        hist = stock.history(period="3y")

        if hist.empty:
            return {"error": "No historical data available", "ticker": ticker}

        current_price = hist['Close'].iloc[-1]
        today = hist.index[-1]

        def _return_for_years(years: int) -> Optional[float]:
            target_date = today - timedelta(days=years * 365)
            # Find the nearest available date
            mask = hist.index <= target_date
            if not mask.any():
                return None
            historical_price = hist.loc[mask, 'Close'].iloc[-1]
            if historical_price and historical_price != 0:
                return round(((current_price - historical_price) / historical_price) * 100, 2)
            return None

        return {
            "ticker": ticker.upper(),
            "current_price": round(float(current_price), 2),
            "return_1y": _return_for_years(1),
            "return_2y": _return_for_years(2),
            "return_3y": _return_for_years(3),
        }
    except Exception as e:
        logger.error(f"Error fetching historical performance for {ticker}: {e}")
        return {"error": str(e), "ticker": ticker}


def get_peer_comparison(
    base_ticker: str,
    peer_tickers: List[str],
) -> Dict[str, Any]:
    """
    Aggregate comparison data for a base stock and its peers.

    Args:
        base_ticker: Primary stock ticker
        peer_tickers: List of peer ticker symbols

    Returns:
        Dictionary with base_stock data, peers data, and comparison_summary.
    """
    logger.info(f"Building peer comparison: {base_ticker} vs {peer_tickers}")

    def _collect(ticker: str) -> Dict[str, Any]:
        financials = get_stock_financials(ticker)
        performance = get_stock_historical_performance(ticker)
        # Merge the two dicts; performance values win on overlap
        merged = {**financials, **performance}
        return merged

    base_data = _collect(base_ticker)

    peers_data = []
    for pt in peer_tickers:
        try:
            peers_data.append(_collect(pt))
        except Exception as e:
            logger.warning(f"Failed to collect data for peer {pt}: {e}")
            peers_data.append({"ticker": pt, "error": str(e)})

    # Compute peer averages (ignoring None / errored values)
    def _peer_avg(key: str) -> Optional[float]:
        vals = [p.get(key) for p in peers_data if p.get(key) is not None and 'error' not in p]
        return round(sum(vals) / len(vals), 2) if vals else None

    summary = {
        "avg_peer_pe": _peer_avg("pe_ratio_trailing"),
        "avg_peer_revenue_growth": _peer_avg("revenue_yoy_growth"),
        "avg_peer_return_1y": _peer_avg("return_1y"),
    }

    return {
        "base_stock": base_data,
        "peers": peers_data,
        "comparison_summary": summary,
    }


# ── Internal helpers ─────────────────────────────────────────────

def _safe_pct(value: Optional[float]) -> Optional[float]:
    """Convert a decimal fraction (e.g. 0.12) to a percentage (12.0), or None."""
    if value is None:
        return None
    return round(value * 100, 2)
