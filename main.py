"""Stock Price Analyzer Agent - CLI Entry Point."""
import argparse
import logging
import sys
from pathlib import Path
from typing import List

from google import genai

import config
from tools import get_stock_price, get_stock_financials, get_stock_historical_performance, get_peer_comparison
from groq_analyzer import identify_industry_peers, generate_comparison_insights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    Run the stock analyzer using Gemini with function calling.
    
    Args:
        tickers: List of stock ticker symbols to analyze
        verbose: Whether to show detailed function call logs
        
    Returns:
        Final analysis text from Gemini
    """
    # Configure Gemini API
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    # Model configuration with function calling
    model_id = config.GEMINI_MODEL
    
    # Create initial prompt
    ticker_list = ", ".join(tickers)
    initial_prompt = f"""You are a stock market analyst. Analyze these stocks: {ticker_list}

For each stock, fetch the current price data and then provide:
1. A brief overview of each stock's current performance
2. Notable movers (biggest gainers or losers)
3. Any potential concerns or opportunities
4. Actionable investment insights based on the data

Be concise but informative."""
    
    logger.info(f"Analyzing {len(tickers)} stocks with Gemini...")
    if verbose:
        logger.info(f"Initial prompt: {initial_prompt}")
    
    # Generate content with function calling enabled
    response = client.models.generate_content(
        model=model_id,
        contents=initial_prompt,
        config=genai.types.GenerateContentConfig(
            tools=[get_stock_price]
        )
    )
    
    # Function calling loop
    iteration = 0
    max_iterations = config.MAX_FUNCTION_CALL_ITERATIONS
    messages = [{'role': 'user', 'parts': [{'text': initial_prompt}]}]
    
    while iteration < max_iterations:
        iteration += 1
        
        # Check for function calls in the response
        if not response.candidates:
            logger.warning("No response candidates from Gemini")
            break
            
        candidate = response.candidates[0]
        
        # Check if we have a final text response
        if candidate.content.parts and hasattr(candidate.content.parts[0], 'text'):
            final_text = candidate.content.parts[0].text
            logger.info("Analysis complete!")
            return final_text
        
        # Check for function calls
        function_calls = []
        if candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part)
        
        if not function_calls:
            logger.warning("No function calls or text in response")
            break
        
        # Execute function calls
        messages.append({'role': 'model', 'parts': candidate.content.parts})
        function_responses = []
        
        for fc_part in function_calls:
            function_call = fc_part.function_call
            function_name = function_call.name
            function_args = dict(function_call.args) if function_call.args else {}
            
            if verbose:
                logger.info(f"Gemini called function: {function_name}({function_args})")
            
            # Execute the function
            if function_name == "get_stock_price":
                result = get_stock_price(**function_args)
                
                if verbose:
                    logger.info(f"Function result: {result}")
                
                function_responses.append({
                    'function_response': {
                        'name': function_name,
                        'response': result
                    }
                })
            else:
                logger.warning(f"Unknown function called: {function_name}")
        
        if not function_responses:
            break
        
        # Send function responses back to Gemini
        messages.append({'role': 'user', 'parts': function_responses})
        
        response = client.models.generate_content(
            model=model_id,
            contents=messages,
            config=genai.types.GenerateContentConfig(
                tools=[get_stock_price]
            )
        )
    return "Analysis incomplete."


def run_peer_comparison(ticker: str, verbose: bool = False) -> str:
    """
    Execute complete peer comparison analysis workflow.

    Args:
        ticker: Stock ticker to analyze
        verbose: Enable detailed logging

    Returns:
        Formatted comparison report string
    """
    separator = "\u2500" * 76

    # Step 1: Get base stock financials (for company name, sector, industry)
    logger.info(f"[Peer Comparison] Fetching base info for {ticker}")
    base_financials = get_stock_financials(ticker)
    if "error" in base_financials:
        return f"\nPeer comparison skipped for {ticker}: {base_financials['error']}"

    company_name = base_financials.get("company_name", ticker)
    sector = base_financials.get("sector", "Unknown")
    industry = base_financials.get("industry", "Unknown")

    # Step 2: Identify peers using Groq
    logger.info(f"[Peer Comparison] Identifying peers for {company_name} ({ticker})")
    peer_tickers = identify_industry_peers(ticker, company_name, sector, industry)
    if not peer_tickers:
        return f"\nPeer comparison skipped for {ticker}: Could not identify industry peers."

    logger.info(f"[Peer Comparison] Peers identified: {peer_tickers}")

    # Step 3: Fetch comparison data
    comparison = get_peer_comparison(ticker, peer_tickers)
    base = comparison["base_stock"]
    peers = comparison["peers"]
    summary = comparison["comparison_summary"]

    # Step 4: Generate insights using Groq
    insights = generate_comparison_insights(base, peers)

    # Step 5: Format output
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
        "PEER COMPARISON ANALYSIS",
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
            report_lines.append(f"  {i}. {peer['ticker']} — data unavailable ({peer.get('error', '')})")
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
        "COMPARATIVE INSIGHTS (Groq Analysis):",
        separator,
        insights,
        "",
        "=" * 80,
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
            print("STOCK ANALYSIS")
            print("="*80)
            print(analysis)
            print("="*80)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
