# Stock Price Analyzer Agent with Gemini - Implementation Plan (Revised)

## Overview

A **CLI tool** that reads stock tickers from a file and uses **Gemini AI with function calling** to orchestrate stock price fetching and analysis. Gemini controls the entire workflow - we just provide tools it can call and display its final response.

## Implementation Steps

### 1. Set up project structure in `my-agent/`
- Create `main.py` as the CLI entry point with argument parsing (`argparse`)
- Create `tools.py` to define functions that Gemini can call (stock price fetcher)
- Create `config.py` to manage settings and load environment variables
- Create `.env.example` template showing required API keys
- Create `requirements.txt` with dependencies: `google-generativeai`, `yfinance`, `python-dotenv`

### 2. Define function tools in `tools.py`
- Create `get_stock_price(ticker: str)` function that fetches current price using `yfinance`
- Returns JSON-serializable dict with: ticker, price, change_percent, volume, timestamp
- Add error handling for invalid tickers (return error message in structured format)
- Define function declaration for Gemini with proper schema (name, description, parameters)

### 3. Implement Gemini orchestration in `main.py`
- Initialize Gemini model with function calling: `genai.GenerativeModel('gemini-pro', tools=[function_declarations])`
- Read stock list from input file
- Send initial prompt to Gemini: "Analyze these stocks: [list]. Fetch current prices and provide investment insights."
- Implement function calling loop:
  - Send chat message to Gemini
  - If Gemini requests function calls, execute them and send results back
  - Continue until Gemini provides final text response
- Display Gemini's final analysis to console (or save to file if specified)

### 4. Configuration management in `config.py`
- Use `python-dotenv` to load `.env` file
- Validate `GEMINI_API_KEY` exists (raise clear error if missing)
- Define constants: model name, max function call iterations (prevent infinite loops)

### 5. Build CLI interface in `main.py`
- Accept `--input-file` argument for stock list file path (required)
- Optional `--output-file` to save analysis (default: print to console)
- Optional `--verbose` flag to show function calls Gemini makes
- Simple flow: parse args → load config → read stocks → call Gemini → display result
- Use `logging` module for verbose output showing Gemini's function calls

### 6. Create example files
- `stocks.txt` with sample tickers (AAPL, GOOGL, MSFT, TSLA, AMZN)
- `.env.example` showing `GEMINI_API_KEY=your_key_here`

### 7. Documentation in `README.md`
- Overview: Simple Gemini-powered stock analyzer
- Setup: install dependencies, configure API key
- Usage examples: `python main.py --input-file stocks.txt`
- How it works: Gemini orchestrates everything via function calling
- Troubleshooting: API key issues, invalid tickers

## Verification

- Run: `python my-agent/main.py --input-file my-agent/stocks.txt`
- With `--verbose`, verify Gemini calls `get_stock_price()` for each stock
- Check that Gemini provides coherent analysis with insights
- Test error handling: missing API key, invalid ticker in list
- Verify output format is readable (save to file and inspect)

## Key Decisions

- **Architecture**: Gemini orchestrates everything via function calling - we just provide tools and display results
- **Removed**: No manual data fetching orchestration, no prompt engineering complexity, no data formatting logic
- **Stock data source**: Still using `yfinance` but only as a tool Gemini calls (not directly managed by our code)
- **Gemini model**: `gemini-pro` with function calling enabled
- **Simplified**: ~70% less code than original plan - Gemini does the heavy lifting

## Requirements Captured

- **Agent Role**: Analyze stock data and provide insights/recommendations (Gemini-orchestrated)
- **Stock Data**: Current prices (Gemini decides what data to fetch)
- **Interface**: Command-line interface (CLI)
- **Stock Source**: File input (CSV/TXT)
- **Key Change**: Everything offloaded to Gemini - we just provide tools and display output