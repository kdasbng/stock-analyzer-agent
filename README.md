# Stock Price Analyzer Agent

A simple CLI tool that leverages **Gemini AI with function calling** to fetch and analyze stock prices. Gemini orchestrates the entire workflow - we just provide the tools and display the results.

## Features

- 📊 **Gemini-Orchestrated**: Gemini AI controls when and how to fetch stock data
- 🤖 **Function Calling**: Uses Gemini's native function calling capability
- 📈 **Real-time Data**: Fetches current stock prices via yfinance
- 💡 **Investment Insights**: AI-powered analysis and recommendations
- 📝 **Flexible Input**: Supports TXT and CSV file formats

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Uses the new `google-genai` package (the old `google.generativeai` is deprecated).

### 2. Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 3. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# GEMINI_API_KEY=your_actual_api_key_here
```

## Usage

### Basic Usage

```bash
python main.py --input-file stocks.txt
```

### Save Output to File

```bash
python main.py --input-file stocks.txt --output-file analysis.txt
```

### Verbose Mode (Show Function Calls)

```bash
python main.py --input-file stocks.txt --verbose
```

## Input File Formats

### Plain Text (stocks.txt)

```
AAPL
GOOGL
MSFT
TSLA
AMZN
```

### CSV (stocks.csv)

```csv
ticker,name
AAPL,Apple Inc.
GOOGL,Alphabet Inc.
MSFT,Microsoft Corporation
```

## How It Works

1. **Read Stock List**: Load tickers from input file
2. **Initialize Gemini**: Create model with function calling enabled
3. **Orchestration Loop**:
   - Send analysis request to Gemini
   - Gemini decides which stocks to analyze
   - Gemini calls `get_stock_price()` function for each ticker
   - We execute the function and return results
   - Gemini processes data and generates insights
4. **Display Results**: Show Gemini's final analysis

## Architecture

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │ stocks.txt
       ↓
┌─────────────────┐
│   main.py       │ ← CLI Entry Point
└────────┬────────┘
         │
         ↓
┌─────────────────┐     ┌──────────────┐
│  Gemini API     │────→│  tools.py    │
│ (Function Call) │     │ (yfinance)   │
└─────────────────┘     └──────────────┘
         │
         ↓ (Final Analysis)
┌─────────────────┐
│   Console/File  │
└─────────────────┘
```

## Example Output

```
================================================================================
STOCK ANALYSIS
================================================================================
Based on the current market data:

**Apple (AAPL)**: Trading at $175.23 (+1.5%), showing positive momentum...

**Alphabet (GOOGL)**: Currently at $142.67 (-0.8%), slight pullback...

**Key Insights**:
- Tech sector showing mixed performance
- TSLA with highest volatility (+3.2%)
- Consider diversification...

**Recommendations**:
1. AAPL and MSFT show stable growth
2. Monitor TSLA for entry points
3. Energy sector may offer opportunities
================================================================================
```

## Troubleshooting

### "GEMINI_API_KEY not found"

**Solution**: Create a `.env` file based on `.env.example` and add your API key.

### "Unable to fetch price data for ticker"

**Possible causes**:
- Invalid ticker symbol
- Stock delisted or trading halted
- Network connectivity issues

**Solution**: Verify ticker symbols are correct and currently traded.

### "Maximum function call iterations reached"

**Cause**: Gemini made too many function calls (safety limit).

**Solution**: Reduce the number of stocks in your input file or check for API rate limiting.

## Development

### Project Structure

```
my-agent/
├── main.py              # CLI entry point & orchestration loop
├── tools.py             # Function declarations & stock fetcher
├── config.py            # Configuration & environment variables
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── stocks.txt           # Sample stock list
└── README.md            # This file
```

### Key Components

- **main.py**: Handles CLI arguments, file I/O, and Gemini chat session
- **tools.py**: Defines `get_stock_price()` function and its schema for Gemini
- **config.py**: Loads environment variables and configuration constants

## Limitations

- Stock data is delayed (not real-time) via yfinance free tier
- Analysis quality depends on Gemini's current knowledge and reasoning
- Rate limits apply to both Gemini API and yfinance
- Function calling has a maximum iteration limit (10 by default)

## Future Enhancements

- [ ] Support for more detailed financial metrics
- [ ] Historical data analysis
- [ ] Portfolio tracking
- [ ] Multiple data sources (fallback APIs)
- [ ] Caching to reduce API calls
- [ ] Asyncio for concurrent requests

## License

MIT License - feel free to modify and use as needed.
