"""Stock Price Analyzer Agent - CLI Entry Point (LangChain edition).

Concept 7: Multi-LLM Orchestration
  - Gemini agent for base stock analysis (Step 1)
  - Groq LCEL chains for peer identification & insights (Step 2)
  - Groq agent for top-peer deep-dive (Step 3)
  All use the same .invoke() interface thanks to LangChain.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from tools import get_stock_financials, get_peer_comparison
from chains import (
    create_gemini_stock_analyzer,
    create_peer_identifier_chain,
    create_comparison_insights_chain,
    create_groq_stock_analyzer,
    select_top_peer,
    parse_peer_tickers,
    format_comparison_for_prompt,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _extract_agent_text(result: dict) -> str:
    """Extract clean text from a create_agent result.

    The last message's `.content` may be:
      - a plain str  (Groq)
      - a list of content blocks like [{'type':'text','text':'...'}, ...] (Gemini)
    This helper normalises both to a single string.
    """
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content
    # list of blocks — keep only text blocks
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def read_stock_list(file_path: str) -> List[str]:
    """
    Read stock tickers from input file.
    Supports:
    - Plain text files with one ticker per line
    - CSV files with a 'ticker' or 'symbol' column
    
    Args:
        file_path: Path to the input file
        
    Returns:
        List of stock ticker symbols
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    tickers = []
    content = path.read_text().strip()
    
    # Check if it's a CSV file
    if file_path.lower().endswith('.csv'):
        lines = content.split('\n')
        if len(lines) > 0:
            header = lines[0].lower().split(',')
            # Find ticker column (could be 'ticker', 'symbol', etc.)
            ticker_col = None
            for i, col in enumerate(header):
                if 'ticker' in col or 'symbol' in col:
                    ticker_col = i
                    break
            
            if ticker_col is not None:
                for line in lines[1:]:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) > ticker_col:
                            tickers.append(parts[ticker_col].strip())
            else:
                # No header found, treat first column as tickers
                for line in lines:
                    if line.strip():
                        parts = line.split(',')
                        if parts[0].strip():
                            tickers.append(parts[0].strip())
    else:
        # Plain text file - one ticker per line
        for line in content.split('\n'):
            ticker = line.strip()
            if ticker and not ticker.startswith('#'):  # Ignore comments
                tickers.append(ticker)
    
    if not tickers:
        raise ValueError(f"No stock tickers found in {file_path}")
    
    logger.info(f"Loaded {len(tickers)} stock tickers from {file_path}")
    return tickers


def run_stock_analyzer(tickers: List[str], verbose: bool = False) -> str:
    """
    Run the stock analyzer using the Gemini LangChain agent.

    Concept 6: AgentExecutor — replaces the manual function-call loop.
    The agent autonomously decides which tools to call and when to stop.

    Args:
        tickers: List of stock ticker symbols to analyze
        verbose: Whether to show detailed function call logs

    Returns:
        Final analysis text from Gemini
    """
    logger.info(f"Analyzing {len(tickers)} stocks with Gemini LangChain agent...")

    # Concept 1 + 5 + 6: Create the Gemini agent
    gemini_agent = create_gemini_stock_analyzer()

    ticker_list = ", ".join(tickers)
    prompt = (
        f"Analyze these stocks: {ticker_list}\n\n"
        "For each stock, fetch the current price data and then provide:\n"
        "1. A brief overview of each stock's current performance\n"
        "2. Notable movers (biggest gainers or losers)\n"
        "3. Any potential concerns or opportunities\n"
        "4. Actionable investment insights based on the data\n\n"
        "Be concise but informative."
    )

    if verbose:
        logger.info(f"Prompt: {prompt}")

    # Concept 7: .invoke() — same interface regardless of which LLM is behind it
    result = gemini_agent.invoke({"messages": [("human", prompt)]})
    return _extract_agent_text(result)


def run_peer_comparison(ticker: str, verbose: bool = False) -> str:
    """
    Execute complete peer comparison + top-peer deep-dive using LangChain chains.

    Concept 3 (LCEL): peer_chain.invoke(), insights_chain.invoke()
    Concept 7 (Multi-LLM): Groq chains for peers, Groq agent for deep-dive

    Args:
        ticker: Stock ticker to analyze
        verbose: Enable detailed logging

    Returns:
        Formatted comparison report string (includes top-peer section)
    """
    separator = "\u2500" * 76

    # ── Step 2a: Get base stock financials ──
    logger.info(f"[Peer Comparison] Fetching base info for {ticker}")
    base_financials = get_stock_financials(ticker)
    if "error" in base_financials:
        return f"\nPeer comparison skipped for {ticker}: {base_financials['error']}"

    company_name = base_financials.get("company_name", ticker)
    sector = base_financials.get("sector", "Unknown")
    industry = base_financials.get("industry", "Unknown")

    # ── Step 2b: Identify peers via Groq LCEL chain (Concept 3) ──
    logger.info(f"[Peer Comparison] Identifying peers for {company_name} ({ticker})")
    peer_chain = create_peer_identifier_chain()
    peer_response = peer_chain.invoke({
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
    })
    peer_tickers = parse_peer_tickers(peer_response, exclude_ticker=ticker)
    if not peer_tickers:
        return f"\nPeer comparison skipped for {ticker}: Could not identify industry peers."

    logger.info(f"[Peer Comparison] Peers identified: {peer_tickers}")

    # ── Step 2c: Fetch comparison data (yfinance — no LLM call) ──
    comparison = get_peer_comparison(ticker, peer_tickers)
    base = comparison["base_stock"]
    peers = comparison["peers"]
    summary = comparison["comparison_summary"]

    # ── Step 2d: Generate insights via Groq LCEL chain (Concept 3) ──
    insights_chain = create_comparison_insights_chain()
    insights = insights_chain.invoke({
        "comparison_data": format_comparison_for_prompt(comparison),
    })

    # ── Format peer comparison report ──
    def _pct(val):
        return f"{val:+.2f}%" if val is not None else "N/A"

    def _fmt_cap(val):
        if val is None:
            return "N/A"
        if val >= 1_000_000_000_000:
            return f"\u20b9{val / 1_000_000_000_000:.2f}T"
        if val >= 1_000_000_000:
            return f"\u20b9{val / 1_000_000_000:.2f}B"
        if val >= 1_000_000:
            return f"\u20b9{val / 1_000_000:.2f}M"
        return f"\u20b9{val:,}"

    report_lines = [
        "",
        "=" * 80,
        "PEER COMPARISON ANALYSIS (Groq via LangChain)",
        "=" * 80,
        "",
        f"BASE STOCK: {base.get('ticker', ticker)} ({base.get('company_name', 'N/A')})",
        separator,
        f"  Current Price : \u20b9{base.get('current_price', 'N/A')}",
        f"  PE Ratio      : {base.get('pe_ratio_trailing', 'N/A')}",
        f"  Revenue Growth: {_pct(base.get('revenue_yoy_growth'))}",
        f"  Profit Margin : {_pct(base.get('profit_margin'))}",
        f"  Market Cap    : {_fmt_cap(base.get('market_cap'))}",
        "",
        "  Performance:",
        f"    1 Year : {_pct(base.get('return_1y'))}",
        f"    2 Years: {_pct(base.get('return_2y'))}",
        f"    3 Years: {_pct(base.get('return_3y'))}",
        "",
        "IDENTIFIED PEERS:",
        separator,
    ]

    for i, peer in enumerate(peers, 1):
        if "error" in peer and "ticker" in peer:
            report_lines.append(f"  {i}. {peer['ticker']} \u2014 data unavailable ({peer.get('error', '')})")
        else:
            report_lines.extend([
                f"  {i}. {peer.get('ticker', 'N/A')} ({peer.get('company_name', 'N/A')})",
                f"     Price: \u20b9{peer.get('current_price', 'N/A')} | "
                f"PE: {peer.get('pe_ratio_trailing', 'N/A')} | "
                f"Revenue Growth: {_pct(peer.get('revenue_yoy_growth'))}",
                f"     Returns: 1Y: {_pct(peer.get('return_1y'))} | "
                f"2Y: {_pct(peer.get('return_2y'))} | "
                f"3Y: {_pct(peer.get('return_3y'))}",
                "",
            ])

    # Peer averages
    report_lines.extend([
        "",
        "PEER AVERAGES:",
        separator,
        f"  Avg PE Ratio      : {summary.get('avg_peer_pe', 'N/A')}",
        f"  Avg Revenue Growth: {_pct(summary.get('avg_peer_revenue_growth'))}",
        f"  Avg 1Y Return     : {_pct(summary.get('avg_peer_return_1y'))}",
    ])

    # Groq insights
    report_lines.extend([
        "",
        "COMPARATIVE INSIGHTS (Groq LangChain LCEL Chain):",
        separator,
        insights,
        "",
        "=" * 80,
    ])

    # ── Step 3: Top Peer Deep-Dive (NEW — Groq Agent) ──
    top_peer = select_top_peer(peers)
    if top_peer:
        top_ticker = top_peer.get("ticker", "N/A")
        logger.info(f"[Top Peer] Running deep-dive analysis for {top_ticker}")

        report_lines.extend([
            "",
            "=" * 80,
            f"TOP PEER DEEP-DIVE: {top_ticker} (Groq Agent via LangChain)",
            "=" * 80,
            f"Selected as top peer based on: "
            f"PE {top_peer.get('pe_ratio_trailing', 'N/A')}, "
            f"Revenue Growth {_pct(top_peer.get('revenue_yoy_growth'))}, "
            f"1Y Return {_pct(top_peer.get('return_1y'))}",
            "",
        ])

        try:
            # Concept 5+6: Groq agent with tools for autonomous analysis
            groq_agent = create_groq_stock_analyzer()
            peer_result = groq_agent.invoke(
                {"messages": [("human", f"Perform a detailed stock analysis for: {top_ticker}")]}
            )
            report_lines.append(_extract_agent_text(peer_result))
        except Exception as e:
            logger.error(f"Groq top-peer deep-dive failed for {top_ticker}: {e}")
            report_lines.append(f"Top peer deep-dive failed: {e}")

        report_lines.extend(["", "=" * 80])
    else:
        report_lines.extend([
            "",
            "(Top peer deep-dive skipped — no valid peers with sufficient data)",
        ])

    return "\n".join(report_lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Price Analyzer Agent powered by Gemini AI"
    )
    parser.add_argument(
        '--input-file',
        required=True,
        help="Path to file containing stock tickers (TXT or CSV)"
    )
    parser.add_argument(
        '--output-file',
        help="Path to save analysis output (default: print to console)"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Show detailed logging including function calls"
    )
    parser.add_argument(
        '--compare-peers',
        action='store_true',
        help="Enable industry peer comparison analysis (requires GROQ_API_KEY)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Read stock list
        tickers = read_stock_list(args.input_file)
        
        # Run analysis
        analysis = run_stock_analyzer(tickers, verbose=args.verbose)
        
        # Run peer comparison if requested
        if args.compare_peers:
            if not config.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not configured. Skipping peer comparison.")
            else:
                for ticker in tickers:
                    try:
                        peer_analysis = run_peer_comparison(ticker, verbose=args.verbose)
                        analysis += "\n" + peer_analysis
                    except Exception as e:
                        logger.error(f"Peer comparison failed for {ticker}: {e}")
        
        # Output results
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.write_text(analysis)
            logger.info(f"Analysis saved to {args.output_file}")
        else:
            print("\n" + "="*80)
            print("STOCK ANALYSIS (Gemini via LangChain)")
            print("="*80)
            print(analysis)
            print("="*80)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
