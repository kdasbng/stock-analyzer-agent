# LangChain Integration Plan — Stock Analyzer Agent

**Date:** February 25, 2026  
**Goal:** Refactor the agent to use LangChain for orchestrating all LLM calls across Gemini and Groq, and add a new "top peer deep-dive" analysis step.

---

## Motivation

1. **Learn LangChain** — Understand chains, tools, prompts, and multi-LLM orchestration
2. **Clean separation** — LangChain abstracts LLM provider details; switching models becomes a config change
3. **New feature** — After peer comparison, select the top-performing peer and run a full stock analysis on it using Groq (saves Gemini free-tier quota)

---

## New End-to-End Flow

```
stocks.txt  →  read tickers
                  │
         ┌────────┴────────────────────────────────────┐
         │  STEP 1: Base Stock Analysis (Gemini)       │
         │  LangChain chain with yfinance tools        │
         │  Model: gemini-2.5-flash                    │
         └────────┬────────────────────────────────────┘
                  │
         ┌────────┴────────────────────────────────────┐
         │  STEP 2: Peer Comparison (Groq)             │
         │  a) Identify 3 peers       → Groq LLM call │
         │  b) Fetch data for all     → yfinance       │
         │  c) Generate comparison    → Groq LLM call  │
         │  d) Select top 1 peer      → scoring logic  │
         └────────┬────────────────────────────────────┘
                  │
         ┌────────┴────────────────────────────────────┐
         │  STEP 3: Top Peer Deep-Dive (Groq) ← NEW   │
         │  Full stock analysis of best peer           │
         │  Reuses same prompt style as Step 1          │
         │  Model: llama-3.3-70b-versatile (Groq)      │
         └────────┬────────────────────────────────────┘
                  │
                  ▼
           Combined Report
```

**LLM Budget per stock:**
| Step | LLM | Calls | Notes |
|---|---|---|---|
| Step 1 | Gemini | 1-2 | Function-calling loop (fetch + analyze) |
| Step 2a | Groq | 1 | Identify peers |
| Step 2c | Groq | 1 | Comparison insights |
| Step 3 | Groq | 1 | Top peer analysis |
| **Total** | | **4-5** | Only 1-2 Gemini calls (free-tier friendly) |

---

## Architecture After LangChain

```
┌────────────────────────────────────────────────────────────┐
│                      main.py                                │
│  CLI: --input-file, --compare-peers, --verbose              │
│                                                             │
│  Orchestration:                                             │
│    1. run_base_analysis()      → LangChain + Gemini        │
│    2. run_peer_comparison()    → LangChain + Groq          │
│    3. run_top_peer_analysis()  → LangChain + Groq  ← NEW  │
└──────────────┬─────────────────────────────────────────────┘
               │
       ┌───────┴────────┐
       ↓                ↓
┌──────────────┐  ┌─────────────────────────────────────┐
│  chains.py   │  │  tools.py                           │
│  (NEW)       │  │  (existing yfinance functions        │
│              │  │   wrapped as LangChain Tools)        │
│  - Gemini    │  │                                      │
│    chain     │  │  get_stock_price()                   │
│  - Groq      │  │  get_stock_financials()              │
│    chain     │  │  get_stock_historical_performance()  │
│  - Prompts   │  │  get_peer_comparison()               │
│              │  │                                      │
└──────┬───────┘  └──────────────────────────────────────┘
       │
       ├──→ LangChain ChatGoogleGenerativeAI  (Gemini)
       └──→ LangChain ChatGroq                (Groq)
```

### Files Changed/Created

| File | Action | Purpose |
|---|---|---|
| `chains.py` | **NEW** | LangChain chains, prompts, and tool bindings |
| `main.py` | **MODIFY** | Replace direct API calls with LangChain chain invocations |
| `tools.py` | **MODIFY** | Add `@tool` decorators or wrap functions as LangChain `Tool` objects |
| `config.py` | **MODIFY** | Add LangChain-specific settings |
| `requirements.txt` | **MODIFY** | Add LangChain packages |
| `groq_analyzer.py` | **DEPRECATE** | Logic moves into `chains.py`; file kept temporarily for reference |

---

## Phase 1: Dependencies & Configuration

### 1.1 Update requirements.txt

```txt
google-genai
yfinance
python-dotenv
groq>=0.4.0
pandas>=2.0.0
langchain>=0.3.0
langchain-google-genai>=2.0.0
langchain-groq>=0.2.0
langchain-community>=0.3.0
```

### 1.2 Update config.py

```python
# LangChain settings
LANGCHAIN_VERBOSE = False  # Set True for debug chain traces

# Gemini via LangChain
GEMINI_MODEL_LANGCHAIN = 'gemini-2.5-flash'
GEMINI_TEMPERATURE = 0.3

# Groq via LangChain
GROQ_MODEL_LANGCHAIN = 'llama-3.3-70b-versatile'
GROQ_TEMPERATURE_PEERS = 0.2       # Deterministic for peer identification
GROQ_TEMPERATURE_ANALYSIS = 0.4    # Slightly creative for narratives
```

---

## Phase 2: LangChain Tool Wrappers (tools.py)

Wrap existing yfinance functions as LangChain tools so they can be used in agent chains.

### Approach: Use `@tool` decorator from `langchain_core.tools`

```python
from langchain_core.tools import tool

@tool
def get_stock_price_tool(ticker: str) -> str:
    """Fetch current stock price and basic information for a given ticker symbol.
    Use this when you need the current price, volume, market cap, or daily change
    for a stock. Input should be a valid stock ticker like 'NATIONALUM.NS'."""
    result = get_stock_price(ticker)
    return json.dumps(result, indent=2)

@tool
def get_stock_financials_tool(ticker: str) -> str:
    """Fetch detailed financial metrics (PE ratio, revenue growth, profit margin,
    sector, industry) for a stock. Input should be a valid ticker symbol."""
    result = get_stock_financials(ticker)
    return json.dumps(result, indent=2)

@tool
def get_stock_historical_performance_tool(ticker: str) -> str:
    """Calculate historical stock returns over 1, 2, and 3 year periods.
    Returns percentage returns for each period. Input should be a valid ticker."""
    result = get_stock_historical_performance(ticker)
    return json.dumps(result, indent=2)
```

**Key point:** LangChain tools return strings (serialized JSON). The docstring becomes the tool description the LLM sees.

---

## Phase 3: Create chains.py — LangChain Chains

### 3.1 Gemini Stock Analysis Chain

Uses LangChain's tool-calling agent pattern with Gemini.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

def create_gemini_stock_analyzer():
    """
    Create a LangChain agent that uses Gemini to analyze stocks.
    Gemini can call yfinance tools to fetch live data.
    """
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_LANGCHAIN,
        google_api_key=config.GEMINI_API_KEY,
        temperature=config.GEMINI_TEMPERATURE,
    )

    tools = [get_stock_price_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a stock market analyst. When analyzing stocks,
         use the available tools to fetch real-time data. Provide concise,
         data-driven insights including current performance, concerns,
         opportunities, and actionable recommendations."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=config.LANGCHAIN_VERBOSE)
```

### 3.2 Groq Peer Identification Chain

Simple LLM chain (no tools needed — just prompting).

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_peer_identifier_chain():
    """
    LangChain chain that asks Groq to identify industry peers.
    Returns raw text with ticker symbols.
    """
    llm = ChatGroq(
        model=config.GROQ_MODEL_LANGCHAIN,
        api_key=config.GROQ_API_KEY,
        temperature=config.GROQ_TEMPERATURE_PEERS,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst expert in the Indian stock market."),
        ("human", """Company: {company_name} ({ticker})
Sector: {sector}
Industry: {industry}

Identify the top 3 publicly traded competitors listed on NSE/BSE.
Provide ONLY ticker symbols, one per line:
1. TICKER1.NS
2. TICKER2.NS
3. TICKER3.NS"""),
    ])

    return prompt | llm | StrOutputParser()
```

### 3.3 Groq Comparison Insights Chain

```python
def create_comparison_insights_chain():
    """
    LangChain chain that generates comparative analysis using Groq.
    """
    llm = ChatGroq(
        model=config.GROQ_MODEL_LANGCHAIN,
        api_key=config.GROQ_API_KEY,
        temperature=config.GROQ_TEMPERATURE_ANALYSIS,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a stock market analyst providing data-driven insights."),
        ("human", """Analyze this comparison data:

{comparison_data}

Provide concise analysis (under 300 words):
1. Relative valuation (PE comparison)
2. Growth trajectory (revenue growth)
3. Performance trends (1Y/2Y/3Y returns)
4. Key takeaways"""),
    ])

    return prompt | llm | StrOutputParser()
```

### 3.4 Top Peer Selection Logic (Python, not LLM)

```python
def select_top_peer(peers_data: List[Dict]) -> Dict:
    """
    Score each peer and return the best one.

    Scoring criteria:
    - Lower PE ratio is better (value)         → weight 0.3
    - Higher revenue growth is better           → weight 0.3
    - Higher 1Y return is better (momentum)     → weight 0.4

    Peers with errors or all-None metrics are excluded.
    """
```

### 3.5 Groq Top Peer Analysis Chain ← NEW

```python
def create_groq_stock_analyzer():
    """
    LangChain agent using Groq + tools for deep-dive analysis of the top peer.
    Same analysis quality as the Gemini chain, but uses Groq (free tier friendly).
    """
    llm = ChatGroq(
        model=config.GROQ_MODEL_LANGCHAIN,
        api_key=config.GROQ_API_KEY,
        temperature=config.GROQ_TEMPERATURE_ANALYSIS,
    )

    tools = [get_stock_price_tool, get_stock_financials_tool,
             get_stock_historical_performance_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a stock market analyst. Analyze the given stock
         using the available tools. Provide a comprehensive analysis including
         price action, valuation, growth metrics, historical performance,
         and investment recommendations."""),
        ("human", "Perform a detailed stock analysis for: {input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=config.LANGCHAIN_VERBOSE)
```

---

## Phase 4: Update main.py — Orchestration

### 4.1 New Orchestration Flow

```python
from chains import (
    create_gemini_stock_analyzer,
    create_peer_identifier_chain,
    create_comparison_insights_chain,
    create_groq_stock_analyzer,
    select_top_peer,
)

def run_full_analysis(ticker: str, compare_peers: bool, verbose: bool) -> str:
    """
    Complete analysis pipeline using LangChain.

    Step 1: Base stock analysis (Gemini)
    Step 2: Peer comparison (Groq)         — if --compare-peers
    Step 3: Top peer deep-dive (Groq)      — if --compare-peers
    """
    report_parts = []

    # ── Step 1: Gemini base analysis ──
    gemini_agent = create_gemini_stock_analyzer()
    base_result = gemini_agent.invoke({
        "input": f"Analyze this stock: {ticker}. Fetch its current price and provide insights."
    })
    report_parts.append(base_result["output"])

    if not compare_peers:
        return "\n".join(report_parts)

    # ── Step 2: Peer comparison (Groq) ──
    # 2a. Get base stock info for peer identification
    financials = get_stock_financials(ticker)
    
    # 2b. Identify peers via Groq chain
    peer_chain = create_peer_identifier_chain()
    peer_response = peer_chain.invoke({
        "ticker": ticker,
        "company_name": financials.get("company_name", ""),
        "sector": financials.get("sector", ""),
        "industry": financials.get("industry", ""),
    })
    peer_tickers = parse_peer_tickers(peer_response)

    # 2c. Fetch comparison data (yfinance — no LLM call)
    comparison = get_peer_comparison(ticker, peer_tickers)

    # 2d. Generate comparison insights via Groq chain
    insights_chain = create_comparison_insights_chain()
    insights = insights_chain.invoke({
        "comparison_data": format_comparison_for_prompt(comparison)
    })
    report_parts.append(format_peer_report(comparison, insights))

    # ── Step 3: Top peer deep-dive (Groq) ── NEW
    top_peer = select_top_peer(comparison["peers"])
    if top_peer:
        groq_agent = create_groq_stock_analyzer()
        peer_result = groq_agent.invoke({
            "input": top_peer["ticker"]
        })
        report_parts.append(format_top_peer_report(top_peer, peer_result["output"]))

    return "\n\n".join(report_parts)
```

---

## Phase 5: Implementation Order

### Step 1: Foundation (Est. 30 min)
- [ ] Update `requirements.txt` with LangChain packages
- [ ] `pip install -r requirements.txt`
- [ ] Update `config.py` with LangChain settings
- [ ] Verify imports: `from langchain_google_genai import ChatGoogleGenerativeAI`

### Step 2: Tool Wrappers (Est. 20 min)
- [ ] Add `@tool` wrappers in `tools.py` for existing functions
- [ ] Test tool wrappers return valid JSON strings
- [ ] Ensure tool docstrings are clear (LLM reads them)

### Step 3: Create chains.py (Est. 45 min)
- [ ] Implement `create_gemini_stock_analyzer()` — Gemini agent with tools
- [ ] Implement `create_peer_identifier_chain()` — Groq simple chain
- [ ] Implement `create_comparison_insights_chain()` — Groq simple chain
- [ ] Implement `select_top_peer()` — Python scoring logic
- [ ] Implement `create_groq_stock_analyzer()` — Groq agent with tools
- [ ] Test each chain independently

### Step 4: Update main.py (Est. 30 min)
- [ ] Refactor `main()` to use LangChain chains
- [ ] Wire up the 3-step flow: Gemini → Peer Comparison → Top Peer
- [ ] Keep `--compare-peers` and `--verbose` flags
- [ ] Preserve report formatting

### Step 5: Test & Cleanup (Est. 30 min)
- [ ] End-to-end test with NATIONALUM.NS
- [ ] Verify Gemini free-tier usage (should be 1-2 calls only)
- [ ] Verify Groq handles all peer + top-peer calls
- [ ] Deprecate/remove `groq_analyzer.py` (logic now in `chains.py`)
- [ ] Update Confluence documentation

---

## LangChain Concepts Used

This implementation will exercise these core LangChain patterns:

| Concept | Where Used | What It Does |
|---|---|---|
| **ChatModel** | `ChatGoogleGenerativeAI`, `ChatGroq` | Unified interface to call different LLMs |
| **Tool** | `@tool` decorator on yfinance functions | Lets LLM agents call Python functions |
| **PromptTemplate** | `ChatPromptTemplate.from_messages()` | Reusable, parameterized prompts |
| **Agent** | `create_tool_calling_agent()` | LLM decides which tools to call and when |
| **AgentExecutor** | Wraps agent + tools | Runs the agent loop (tool calls → results → next) |
| **Chain (LCEL)** | `prompt \| llm \| parser` | Composable pipeline for simple LLM calls |
| **OutputParser** | `StrOutputParser` | Extracts text from LLM response objects |

---

## Top Peer Selection Scoring

The `select_top_peer()` function uses a simple weighted scoring model:

```
Score = (PE_score × 0.3) + (Growth_score × 0.3) + (Return_score × 0.4)

Where:
  PE_score     = normalize(1 / pe_ratio)    # Lower PE → higher score
  Growth_score = normalize(revenue_growth)   # Higher growth → higher score
  Return_score = normalize(return_1y)        # Higher 1Y return → higher score

normalize() = (value - min) / (max - min)  across the peer set
```

Peers with missing data are scored only on available metrics. Peers with errors are excluded.

---

## Expected Output Format

```
================================================================================
STOCK ANALYSIS (Gemini)
================================================================================
[Gemini's analysis of the base stock using live data...]

================================================================================
PEER COMPARISON ANALYSIS (Groq)
================================================================================
[Same peer comparison table as before...]

COMPARATIVE INSIGHTS:
[Groq-generated comparison narrative...]

================================================================================
TOP PEER DEEP-DIVE: HINDALCO.NS (Groq)                              ← NEW
================================================================================
Selected as top peer based on: PE 12.97, Revenue Growth +10.34%, 1Y Return +52.44%

[Groq agent's full analysis of HINDALCO.NS using yfinance tools:
 - Current price and momentum
 - Valuation metrics
 - Historical performance
 - Growth analysis
 - Investment recommendations]

================================================================================
```

---

## Key Design Decisions

### Why LangChain?
- Provides a consistent interface for both Gemini and Groq
- Built-in support for tool-calling agents (no manual function-call loops)
- LCEL (LangChain Expression Language) makes simple chains composable with `|`
- Easy to swap models later (e.g., switch Groq model without code changes)

### Why keep Step 1 on Gemini?
- Gemini has excellent function-calling support via LangChain
- Demonstrates multi-LLM orchestration (Gemini for base, Groq for peers)
- Free tier is sufficient for 1-2 calls per run

### Why do Top Peer analysis on Groq?
- Saves Gemini quota (Groq free tier is more generous)
- Groq is fast (inference speed)
- Demonstrates that the same tool-calling pattern works across different LLMs

### Why Python scoring instead of LLM for top-peer selection?
- Deterministic — same data always produces the same pick
- No extra LLM call needed
- Transparent — scoring weights are visible and adjustable

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| LangChain Gemini tool-calling quirks | Agent may not call tools correctly | Fallback to direct Gemini API if needed |
| Groq tool-calling support | Groq may not support all tool patterns | Use simple chains for Groq (no tools) where possible |
| Package version conflicts | Import errors | Pin versions in requirements.txt |
| LangChain deprecation warnings | Noisy logs | Use latest stable APIs (`langchain_core`, not legacy) |

---

## Cleanup After Migration

Once LangChain integration is complete and tested:

1. **`groq_analyzer.py`** — Delete (logic fully replaced by `chains.py`)
2. **Direct `google-genai` import** — Remove from `main.py` (replaced by `langchain-google-genai`)
3. **Manual function-call loop** — Remove from `main.py` (replaced by `AgentExecutor`)
4. **`groq` package** — Can potentially be removed if `langchain-groq` handles everything

---

**Status:** Plan Ready — Awaiting Implementation  
**Next Step:** Begin Phase 1 (Dependencies & Configuration)
