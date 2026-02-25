"""Groq API integration for industry peer identification and comparative analysis."""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from groq import Groq

import config

logger = logging.getLogger(__name__)


def _get_groq_client() -> Groq:
    """Initialize and return a Groq client."""
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured. Cannot use peer comparison features.")
    return Groq(api_key=config.GROQ_API_KEY)


def identify_industry_peers(
    ticker: str,
    company_name: str,
    sector: str,
    industry: str,
) -> List[str]:
    """
    Use Groq LLM to identify top 3 publicly traded industry peers.

    Args:
        ticker: Base stock ticker symbol
        company_name: Full company name
        sector: Business sector (e.g., "Transportation")
        industry: Specific industry (e.g., "Airlines")

    Returns:
        List of up to 3 peer ticker symbols (e.g., ["INDIGO.NS", "AIRTEL.NS"])
    """
    client = _get_groq_client()

    prompt = f"""You are a financial analyst expert in the Indian stock market.

Company: {company_name} ({ticker})
Sector: {sector}
Industry: {industry}

Task: Identify the top {config.MAX_PEERS_TO_ANALYZE} publicly traded competitors/peers of this company \
that are listed on NSE (National Stock Exchange) or BSE (Bombay Stock Exchange).

Rules:
- Do NOT include the base company ({ticker}) itself.
- Provide ONLY the ticker symbols, one per line, in this exact format:
1. TICKER1.NS
2. TICKER2.NS
3. TICKER3.NS

Use .NS (NSE) suffix by default. Only use .BO (BSE) if the stock is exclusively listed on BSE.

Response:"""

    logger.info(f"Asking Groq to identify peers for {ticker} ({company_name})")

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.GROQ_MODEL,
            temperature=0.2,
            max_tokens=256,
        )

        response_text = chat_completion.choices[0].message.content.strip()
        logger.debug(f"Groq peer response: {response_text}")

        # Parse ticker symbols from the response
        peers = _parse_peer_tickers(response_text, exclude_ticker=ticker)
        logger.info(f"Identified peers for {ticker}: {peers}")
        return peers[: config.MAX_PEERS_TO_ANALYZE]

    except Exception as e:
        logger.error(f"Groq peer identification failed for {ticker}: {e}")
        return []


def _parse_peer_tickers(text: str, exclude_ticker: str = "") -> List[str]:
    """
    Extract ticker symbols from Groq's response text.

    Looks for patterns like TICKER.NS or TICKER.BO.
    """
    # Match tickers that end with .NS or .BO
    pattern = r'\b([A-Z0-9&]+(?:\.NS|\.BO))\b'
    matches = re.findall(pattern, text.upper())

    # Deduplicate while preserving order, and exclude the base ticker
    seen = set()
    tickers = []
    exclude_upper = exclude_ticker.upper()
    for t in matches:
        if t not in seen and t != exclude_upper:
            seen.add(t)
            tickers.append(t)
    return tickers


def generate_comparison_insights(
    base_stock: Dict[str, Any],
    peers_data: List[Dict[str, Any]],
) -> str:
    """
    Generate narrative insights from comparison data using Groq LLM.

    Args:
        base_stock: Dictionary with base stock metrics
        peers_data: List of dictionaries with peer stock metrics

    Returns:
        AI-generated comparative analysis string
    """
    client = _get_groq_client()

    # Build a concise data summary for the prompt
    def _fmt(stock: Dict[str, Any]) -> str:
        lines = [
            f"  Ticker: {stock.get('ticker', 'N/A')}",
            f"  Company: {stock.get('company_name', 'N/A')}",
            f"  PE Ratio (Trailing): {stock.get('pe_ratio_trailing', 'N/A')}",
            f"  Revenue Growth (YoY): {_pct(stock.get('revenue_yoy_growth'))}",
            f"  Profit Margin: {_pct(stock.get('profit_margin'))}",
            f"  Market Cap: {_fmt_cap(stock.get('market_cap'))}",
            f"  1-Year Return: {_pct(stock.get('return_1y'))}",
            f"  2-Year Return: {_pct(stock.get('return_2y'))}",
            f"  3-Year Return: {_pct(stock.get('return_3y'))}",
        ]
        return "\n".join(lines)

    peer_sections = ""
    for i, peer in enumerate(peers_data, 1):
        peer_sections += f"\nPEER {i}:\n{_fmt(peer)}\n"

    prompt = f"""Analyze this stock comparison data and provide investment insights.

BASE STOCK:
{_fmt(base_stock)}

{peer_sections}

Provide a concise analysis covering:
1. Relative valuation analysis (PE comparison)
2. Growth trajectory comparison (revenue growth)
3. Stock performance trends (1Y/2Y/3Y returns)
4. Key takeaways and investment perspective

Keep the analysis under 300 words. Be data-driven and specific."""

    logger.info("Generating comparison insights via Groq...")

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.GROQ_MODEL,
            temperature=0.4,
            max_tokens=1024,
        )
        insights = chat_completion.choices[0].message.content.strip()
        return insights
    except Exception as e:
        logger.error(f"Groq insights generation failed: {e}")
        return "Unable to generate comparative insights at this time."


# ── Formatting helpers ──────────────────────────────────────────

def _pct(value: Optional[float]) -> str:
    """Format a value as a percentage string, or 'N/A'."""
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def _fmt_cap(value: Optional[int]) -> str:
    """Human-readable market cap."""
    if value is None:
        return "N/A"
    if value >= 1_000_000_000_000:
        return f"₹{value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"₹{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"₹{value / 1_000_000:.2f}M"
    return f"₹{value:,}"
