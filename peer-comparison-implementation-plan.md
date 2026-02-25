# Industry Peer Comparison Feature - Implementation Plan

**Date:** February 24, 2026  
**Feature:** Add Groq API integration for industry peer identification and comparative analysis

---

## Overview

Expand the Stock Analyzer Agent to identify top 3 industry peers using Groq API and compare key financial metrics including PE ratio, revenue growth, and stock price performance over multiple time periods.

---

## Architecture Design

```
┌───────────────────────────────────────────────────────────────┐
│                         main.py                                │
│  - CLI with --compare-peers flag                              │
│  - Orchestrates Gemini + Groq workflow                        │
│  - Manages analysis pipeline                                  │
└─────────────────┬─────────────────────────────────────────────┘
                  │
          ┌───────┴────────┐
          │                │
          ↓                ↓
┌──────────────────┐  ┌──────────────────────────────┐
│    tools.py      │  │    groq_analyzer.py (NEW)    │
│  - get_stock_    │  │  - identify_industry_peers() │
│    price()       │  │  - generate_comparison_      │
│  - get_stock_    │  │    insights()                │
│    financials()  │  │  - Groq LLM integration      │
│  - get_stock_    │  │                              │
│    historical_   │  │                              │
│    performance() │  │                              │
│  - get_peer_     │  │                              │
│    comparison()  │  │                              │
└────────┬─────────┘  └─────────────┬────────────────┘
         │                          │
         ↓                          ↓
   Yahoo Finance                Groq API
   (stock data)            (peer discovery & analysis)
```

---

## Phase 1: Configuration Setup

### 1.1 Update config.py

**Add new configuration parameters:**

```python
# Groq API Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found. Peer comparison features will be disabled.")

# Groq Model Selection
GROQ_MODEL = 'mixtral-8x7b-32768'  # Alternative: 'llama2-70b-4096'

# Feature Flags
ENABLE_PEER_COMPARISON = True
MAX_PEERS_TO_ANALYZE = 3

# Comparison Time Periods
COMPARISON_PERIODS = ['1Y', '2Y', '3Y']
```

### 1.2 Update .env.example

```bash
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Groq API Configuration (for peer comparison)
GROQ_API_KEY=your_groq_api_key_here
```

### 1.3 Update requirements.txt

```txt
google-genai==1.64.0
yfinance==0.2.48
python-dotenv==1.0.0
groq>=0.4.0
pandas>=2.0.0
```

**Note:** pandas needed for historical date calculations

---

## Phase 2: Create New Groq Module

### 2.1 Create groq_analyzer.py

**File:** `my-agent/groq_analyzer.py`

**Purpose:** Handle all Groq API interactions for peer discovery and comparative analysis

**Functions to implement:**

#### Function 1: `identify_industry_peers()`

```python
def identify_industry_peers(
    ticker: str, 
    company_name: str, 
    sector: str,
    industry: str
) -> List[str]:
    """
    Use Groq LLM to identify top 3 publicly traded industry peers.
    
    Args:
        ticker: Base stock ticker symbol
        company_name: Full company name
        sector: Business sector (e.g., "Transportation")
        industry: Specific industry (e.g., "Airlines")
        
    Returns:
        List of 3 peer ticker symbols (e.g., ["INDIGO.NS", "JETAIRWAYS.NS", "..."])
        
    Implementation:
        1. Initialize Groq client
        2. Create prompt asking for peer companies
        3. Request ticker symbols for Indian stock exchanges (NSE/BSE)
        4. Parse response and validate ticker format
        5. Return list of peer tickers
    """
```

**Prompt Template:**
```
You are a financial analyst expert in the Indian stock market.

Company: {company_name} ({ticker})
Sector: {sector}
Industry: {industry}

Task: Identify the top 3 publicly traded competitors/peers of this company 
that are listed on NSE (National Stock Exchange) or BSE (Bombay Stock Exchange).

Provide ONLY the ticker symbols in this format:
1. TICKER1.NS or TICKER1.BO
2. TICKER2.NS or TICKER2.BO
3. TICKER3.NS or TICKER3.BO

Examples: INDIGO.NS, TATASTEEL.BO

Response:
```

#### Function 2: `generate_comparison_insights()`

```python
def generate_comparison_insights(
    base_stock: Dict[str, Any],
    peers_data: List[Dict[str, Any]]
) -> str:
    """
    Generate narrative insights from comparison data using Groq LLM.
    
    Args:
        base_stock: Dictionary with base stock metrics
        peers_data: List of dictionaries with peer stock metrics
        
    Returns:
        String with AI-generated comparative analysis
        
    Implementation:
        1. Format comparison data as structured text
        2. Send to Groq with analysis request
        3. Return formatted insights
    """
```

**Prompt Template:**
```
Analyze this stock comparison data and provide investment insights:

BASE STOCK: {ticker}
- PE Ratio: {pe_ratio}
- Revenue Growth (YoY): {revenue_growth}%
- Stock Returns: 1Y: {return_1y}%, 2Y: {return_2y}%, 3Y: {return_3y}%
- Market Cap: {market_cap}

PEER 1: {peer1_ticker}
- PE Ratio: {pe_ratio}
- Revenue Growth (YoY): {revenue_growth}%
- Stock Returns: 1Y: {return_1y}%, 2Y: {return_2y}%, 3Y: {return_3y}%

[... peers 2 and 3 ...]

Provide:
1. Relative valuation analysis (PE comparison)
2. Growth trajectory comparison
3. Performance trends
4. Investment recommendation with reasoning
```

---

## Phase 3: Extend Stock Analysis Tools

### 3.1 Update tools.py

**Add new functions:**

#### Function 3: `get_stock_financials()`

```python
def get_stock_financials(ticker: str) -> Dict[str, Any]:
    """
    Fetch detailed financial metrics for a stock.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with:
        - pe_ratio_trailing: Trailing P/E ratio
        - pe_ratio_forward: Forward P/E ratio
        - revenue_yoy_growth: Year-over-year revenue growth %
        - profit_margin: Net profit margin %
        - market_cap: Market capitalization
        - sector: Business sector
        - industry: Specific industry
        - company_name: Full company name
        
    Uses yfinance:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
    """
```

#### Function 4: `get_stock_historical_performance()`

```python
def get_stock_historical_performance(ticker: str) -> Dict[str, Any]:
    """
    Calculate stock price returns over multiple periods.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with:
        - return_1y: 1-year return percentage
        - return_2y: 2-year return percentage
        - return_3y: 3-year return percentage
        - current_price: Current stock price
        
    Implementation:
        1. Use yfinance history() method
        2. Calculate dates: 1Y, 2Y, 3Y ago from today
        3. Fetch historical prices for those dates
        4. Calculate percentage returns
        5. Handle missing data (stock may not have 3Y history)
        
    Formula:
        return_percent = ((current_price - historical_price) / historical_price) * 100
    """
```

#### Function 5: `get_peer_comparison()`

```python
def get_peer_comparison(
    base_ticker: str, 
    peer_tickers: List[str]
) -> Dict[str, Any]:
    """
    Aggregate comparison data for base stock and peers.
    
    Args:
        base_ticker: Primary stock ticker
        peer_tickers: List of peer ticker symbols
        
    Returns:
        Dictionary with structure:
        {
            "base_stock": {
                "ticker": str,
                "company_name": str,
                "pe_ratio": float,
                "revenue_growth": float,
                "return_1y": float,
                "return_2y": float,
                "return_3y": float,
                "market_cap": int
            },
            "peers": [
                {same structure as base_stock},
                ...
            ],
            "comparison_summary": {
                "avg_peer_pe": float,
                "avg_peer_revenue_growth": float,
                "avg_peer_return_1y": float
            }
        }
        
    Implementation:
        1. Call get_stock_financials() for base + all peers
        2. Call get_stock_historical_performance() for base + all peers
        3. Merge data into structured format
        4. Calculate peer averages
        5. Handle errors gracefully (peer data may be missing)
    """
```

---

## Phase 4: Main Orchestration Update

### 4.1 Update main.py

**Changes needed:**

#### 4.1.1 Add CLI Argument

```python
parser.add_argument(
    '--compare-peers',
    action='store_true',
    help="Enable industry peer comparison analysis (requires GROQ_API_KEY)"
)
```

#### 4.1.2 New Function: `run_peer_comparison()`

```python
def run_peer_comparison(ticker: str, verbose: bool = False) -> str:
    """
    Execute complete peer comparison analysis workflow.
    
    Args:
        ticker: Stock ticker to analyze
        verbose: Enable detailed logging
        
    Returns:
        Formatted comparison report
        
    Workflow:
        Step 1: Get base stock info
            - Call get_stock_price() via Gemini
            - Get company name, sector, industry
            
        Step 2: Identify peers using Groq
            - Call groq_analyzer.identify_industry_peers()
            - Validate peer tickers
            
        Step 3: Fetch comparison metrics
            - Call get_peer_comparison() from tools.py
            - Aggregate all financial and performance data
            
        Step 4: Generate insights using Groq
            - Call groq_analyzer.generate_comparison_insights()
            - Get AI-generated narrative
            
        Step 5: Format output
            - Create formatted table
            - Include Groq insights
            - Return complete report
    """
```

#### 4.1.3 Update `main()` Function

```python
def main():
    # ... existing arg parsing ...
    
    try:
        tickers = read_stock_list(args.input_file)
        
        # Run basic analysis with Gemini
        analysis = run_stock_analyzer(tickers, verbose=args.verbose)
        
        # NEW: Run peer comparison if flag enabled
        if args.compare_peers:
            if not config.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not configured. Skipping peer comparison.")
            else:
                for ticker in tickers:
                    peer_analysis = run_peer_comparison(ticker, verbose=args.verbose)
                    analysis += "\n\n" + peer_analysis
        
        # Output results
        # ... existing output code ...
```

---

## Phase 5: Implementation Order

### Step 1: Foundation Setup (Day 1)
1. ✅ Update `requirements.txt` - add groq and pandas
2. ✅ Update `config.py` - add Groq settings
3. ✅ Update `.env.example` - add GROQ_API_KEY
4. ✅ Install dependencies: `pip install -r requirements.txt`
5. ✅ Get Groq API key from https://console.groq.com

### Step 2: Create Groq Module (Day 1-2)
1. ✅ Create `groq_analyzer.py` file
2. ✅ Implement `identify_industry_peers()` function
3. ✅ Test peer identification with sample stock
4. ✅ Implement `generate_comparison_insights()` function
5. ✅ Test insights generation with sample data

### Step 3: Extend Tools (Day 2-3)
1. ✅ Add `get_stock_financials()` to `tools.py`
2. ✅ Add `get_stock_historical_performance()` to `tools.py`
3. ✅ Add `get_peer_comparison()` to `tools.py`
4. ✅ Test each function individually with known tickers
5. ✅ Handle edge cases (missing data, delisted stocks)

### Step 4: Main Integration (Day 3)
1. ✅ Add `--compare-peers` CLI argument to `main.py`
2. ✅ Implement `run_peer_comparison()` function
3. ✅ Update `main()` to conditionally run peer analysis
4. ✅ Test end-to-end workflow

### Step 5: Testing & Refinement (Day 4)
1. ✅ Test with SpiceJet (SPICEJET.BO)
2. ✅ Test with other stocks (NATIONALUM.NS, TATASTEEL.NS)
3. ✅ Verify error handling for invalid tickers
4. ✅ Test with VPN on/off scenarios
5. ✅ Optimize Groq prompts for better results

---

## Data Structure Examples

### Example Output: Peer Comparison Data

```json
{
  "base_stock": {
    "ticker": "SPICEJET.BO",
    "company_name": "SpiceJet Limited",
    "sector": "Transportation",
    "industry": "Airlines",
    "pe_ratio": 12.5,
    "revenue_growth_yoy": -5.2,
    "profit_margin": -2.3,
    "market_cap": 21840000000,
    "return_1y": -30.5,
    "return_2y": -45.2,
    "return_3y": -60.1,
    "current_price": 14.31
  },
  "peers": [
    {
      "ticker": "INDIGO.NS",
      "company_name": "InterGlobe Aviation Limited",
      "sector": "Transportation",
      "industry": "Airlines",
      "pe_ratio": 18.3,
      "revenue_growth_yoy": 15.2,
      "profit_margin": 8.5,
      "market_cap": 850000000000,
      "return_1y": 25.3,
      "return_2y": 45.8,
      "return_3y": 120.5,
      "current_price": 2250.50
    },
    {
      "ticker": "JETAIRWAYS.NS",
      "company_name": "Jet Airways (India) Ltd",
      "sector": "Transportation", 
      "industry": "Airlines",
      "pe_ratio": null,
      "revenue_growth_yoy": null,
      "profit_margin": null,
      "market_cap": null,
      "return_1y": null,
      "return_2y": null,
      "return_3y": null,
      "current_price": null,
      "note": "Suspended from trading"
    }
  ],
  "comparison_summary": {
    "avg_peer_pe": 18.3,
    "avg_peer_revenue_growth": 15.2,
    "avg_peer_return_1y": 25.3,
    "base_vs_peer_pe_diff": -5.8,
    "base_vs_peer_growth_diff": -20.4
  }
}
```

---

## Expected Usage Examples

### Basic Analysis (Existing)
```bash
python my-agent/main.py --input-file stocks.txt
```

### With Peer Comparison (New)
```bash
python my-agent/main.py --input-file stocks.txt --compare-peers
```

### Single Stock with Detailed Peer Analysis
```bash
python my-agent/main.py --input-file spicejet.txt --compare-peers --verbose
```

### Expected Output Format

```
================================================================================
STOCK ANALYSIS
================================================================================
[... existing Gemini analysis ...]

================================================================================
PEER COMPARISON ANALYSIS
================================================================================

BASE STOCK: SPICEJET.BO (SpiceJet Limited)
────────────────────────────────────────────────────────────────────────────
Current Price: ₹14.31 (-9.72%)
PE Ratio: 12.5
Revenue Growth (YoY): -5.2%
Market Cap: ₹21.84B

Performance:
  1 Year:  -30.5%
  2 Years: -45.2%
  3 Years: -60.1%

IDENTIFIED PEERS:
────────────────────────────────────────────────────────────────────────────
1. INDIGO.NS (InterGlobe Aviation Limited)
   Price: ₹2,250.50 | PE: 18.3 | Revenue Growth: 15.2%
   Returns: 1Y: +25.3% | 2Y: +45.8% | 3Y: +120.5%

2. [Peer 2 data...]

3. [Peer 3 data...]

COMPARATIVE INSIGHTS (Groq Analysis):
────────────────────────────────────────────────────────────────────────────
[AI-generated narrative comparing SpiceJet with peers, including:
 - Valuation assessment (PE comparison)
 - Growth trajectory analysis
 - Performance trends
 - Investment recommendation]

================================================================================
```

---

## Benefits of This Design

### 1. Modular Architecture
- **Separation of Concerns:** Groq functionality isolated in dedicated module
- **Easy to Test:** Each component can be tested independently
- **Maintainable:** Clear boundaries between Gemini and Groq features

### 2. Optional Feature
- **CLI Flag Control:** Users can opt-in to peer comparison
- **Graceful Degradation:** Works without Groq API key (basic analysis only)
- **No Breaking Changes:** Existing functionality remains unchanged

### 3. Two-AI Approach
- **Gemini:** Orchestrates stock data fetching with function calling
- **Groq:** Provides fast LLM inference for peer discovery and insights
- **Best of Both:** Leverage strengths of each AI system

### 4. Extensible
- Easy to add more comparison metrics (dividend yield, beta, etc.)
- Can expand to more peers (currently capped at 3)
- Future: Add technical indicators comparison

### 5. Error Handling
- Each peer fetch independent (one failure doesn't break entire analysis)
- Handles missing historical data gracefully
- Validates ticker formats before fetching

---

## Technical Considerations

### API Rate Limits
- **Groq:** Check rate limits for chosen model
- **yfinance:** No official rate limit, but be respectful
- **Gemini:** Already handled in existing code (15 req/min free tier)

### Data Availability
- Not all stocks have 3 years of historical data
- Some metrics may be missing (especially for small-cap stocks)
- Implement null checks and default values

### Performance
- Peer comparison adds ~5-10 seconds per stock
- Multiple API calls (1 Groq + 4-7 yfinance calls)
- Consider adding caching for repeated analyses

### Error Scenarios to Handle
1. Groq fails to identify peers → fallback to manual list
2. Peer ticker invalid/delisted → skip and note in output
3. Historical data missing → show "N/A" instead of error
4. Financial metrics unavailable → use available data only

---

## Dependencies & Prerequisites

### Required API Keys
1. **Gemini API Key** (existing)
   - Get from: https://makersuite.google.com/app/apikey
   
2. **Groq API Key** (new)
   - Sign up: https://console.groq.com
   - Free tier available
   - Fast inference for LLama2 and Mixtral models

### Python Packages
- `groq>=0.4.0` - Groq API client
- `pandas>=2.0.0` - Date calculations and data manipulation
- `yfinance==0.2.48` - Already installed
- `google-genai==1.64.0` - Already installed

### Development Environment
- Python 3.9+ recommended
- Virtual environment: `python -m venv venv`
- VPN considerations for yfinance access

---

## Testing Strategy

### Unit Tests
1. Test `identify_industry_peers()` with mock Groq responses
2. Test `get_stock_financials()` with known ticker
3. Test `get_stock_historical_performance()` date calculations
4. Test `get_peer_comparison()` data aggregation

### Integration Tests
1. End-to-end test with SPICEJET.BO
2. Test with stock having full 3Y history
3. Test with newly listed stock (< 1Y history)
4. Test with invalid ticker in peer list

### Edge Cases
1. Groq returns invalid ticker formats
2. All peers fail to fetch data
3. Base stock has no financial data
4. Network timeout scenarios

---

## Future Enhancements

### Phase 6 (Future)
1. **Technical Indicators Comparison**
   - RSI, MACD, Moving Averages
   - Add `ta-lib` or `pandas-ta` library

2. **Dividend Analysis**
   - Dividend yield comparison
   - Payout ratio analysis

3. **Fundamental Ratios**
   - Debt-to-Equity
   - ROE, ROA, ROIC
   - Current Ratio, Quick Ratio

4. **Caching Layer**
   - Cache peer identifications
   - Cache historical data for 24 hours
   - Use SQLite or Redis

5. **Visualization**
   - Generate comparison charts
   - Export to CSV/Excel
   - Web dashboard interface

6. **Sector-Wide Analysis**
   - Compare against sector averages
   - Identify sector leaders/laggards

---

## Success Metrics

### Feature is successful if:
1. ✅ Correctly identifies relevant industry peers 90%+ of time
2. ✅ Comparison completes within 15 seconds per stock
3. ✅ Handles missing data gracefully without crashes
4. ✅ Provides actionable insights from Groq analysis
5. ✅ Works with existing CLI without breaking changes

---

## Implementation Checklist

- [ ] Phase 1: Configuration Setup
  - [ ] Update config.py with Groq settings
  - [ ] Update .env.example
  - [ ] Update requirements.txt
  - [ ] Install new dependencies

- [ ] Phase 2: Create Groq Module
  - [ ] Create groq_analyzer.py
  - [ ] Implement identify_industry_peers()
  - [ ] Implement generate_comparison_insights()
  - [ ] Test Groq functions independently

- [ ] Phase 3: Extend Tools
  - [ ] Implement get_stock_financials()
  - [ ] Implement get_stock_historical_performance()
  - [ ] Implement get_peer_comparison()
  - [ ] Test each function with sample data

- [ ] Phase 4: Main Integration
  - [ ] Add --compare-peers CLI argument
  - [ ] Implement run_peer_comparison()
  - [ ] Update main() function
  - [ ] Integrate error handling

- [ ] Phase 5: Testing & Documentation
  - [ ] End-to-end testing with multiple stocks
  - [ ] Test error scenarios
  - [ ] Update DEVELOPER_GUIDE.md
  - [ ] Create usage examples
  - [ ] Performance optimization

---

## Notes & Considerations

1. **Groq Model Selection:**
   - `mixtral-8x7b-32768`: Balanced performance, large context
   - `llama2-70b-4096`: High quality, smaller context
   - Start with Mixtral, can experiment later

2. **Indian Stock Market Specifics:**
   - NSE (.NS) and BSE (.BO) suffixes required
   - Some stocks trade on both exchanges
   - Groq needs context about Indian market

3. **Data Quality:**
   - yfinance data may have gaps
   - Always validate before calculations
   - Provide clear error messages to users

4. **Cost Considerations:**
   - Groq free tier is generous
   - yfinance is free
   - Main cost is Gemini API (existing)

---

**Status:** Planning Complete - Ready for Implementation  
**Next Step:** Begin Phase 1 (Configuration Setup)
