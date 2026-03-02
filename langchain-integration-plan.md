# LangChain Integration Plan — Stock Analyzer Agent

**Date:** February 25, 2026  
**Goal:** Refactor the agent to use LangChain for orchestrating all LLM calls across Gemini and Groq, and add a new "top peer deep-dive" analysis step.

---

## Motivation

1. **Learn LangChain** — Understand chains, tools, prompts, and multi-LLM orchestration
2. **Clean separation** — LangChain abstracts LLM provider details; switching models becomes a config change
3. **New feature** — After peer comparison, select the top-performing peer and run a full stock analysis on it using Groq (saves Gemini free-tier quota)

---

## LangChain Learning Roadmap

This project teaches **7 core LangChain concepts** in a logical progression. Each implementation phase introduces new concepts, building on the previous one.

```
  CONCEPT 1          CONCEPT 2           CONCEPT 3
  ChatModel    ──→   PromptTemplate ──→  LCEL Chain (│)
  (Phase 1)          (Phase 3)           (Phase 3)
       │                                      │
       ▼                                      ▼
  CONCEPT 4          CONCEPT 5           CONCEPT 6
  Tool / @tool ──→   Agent          ──→  AgentExecutor
  (Phase 2)          (Phase 3)           (Phase 3)
                                              │
                                              ▼
                                         CONCEPT 7
                                         Multi-LLM Orchestration
                                         (Phase 4)
```

### Concept-by-Concept Guide with Video References

---

#### Concept 1: ChatModel — The LLM Wrapper

**What it is:** LangChain provides a unified `BaseChatModel` interface. Every LLM provider (OpenAI, Gemini, Groq, Anthropic…) implements this same interface. You swap one LLM for another by changing a single class — your prompt code stays identical.

**Where we use it:**
- `ChatGoogleGenerativeAI` — wraps Gemini API
- `ChatGroq` — wraps Groq API

**Why it matters:** Without LangChain, our code has two completely different API clients (`google.genai` and `groq.Groq`). With LangChain, both are just `ChatModel` instances with the same `.invoke()` method.

```python
# Before (two different APIs):
from google import genai                   # Gemini
client = genai.Client(api_key=...)
response = client.models.generate_content(model=..., contents=...)

from groq import Groq                      # Groq
client = Groq(api_key=...)
chat = client.chat.completions.create(messages=..., model=...)

# After (one unified interface):
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", ...)
groq   = ChatGroq(model="llama-3.3-70b-versatile", ...)

# Both use the exact same method:
gemini.invoke("Analyze this stock")
groq.invoke("Analyze this stock")
```

**📺 Video:** [LangChain Chat Models — Complete Guide](https://www.youtube.com/watch?v=wVwxSzOF3Fo) by Krish Naik (covers ChatModel basics, how to initialize different providers)

---

#### Concept 2: PromptTemplate — Reusable, Parameterized Prompts

**What it is:** Instead of building prompt strings with f-strings, LangChain `ChatPromptTemplate` creates structured, reusable prompt objects with named variables. This separates prompt engineering from business logic.

**Where we use it:**
- System + human message pairs for every LLM call
- `{input}`, `{ticker}`, `{sector}` etc. as template variables
- `{agent_scratchpad}` placeholder for agent chains (LangChain fills this automatically)

**Why it matters:** Templates are composable, testable, and version-controllable. You can modify a prompt without touching any Python logic.

```python
# Before (f-strings mixed with logic):
prompt = f"You are a stock analyst. Analyze: {ticker}. Sector: {sector}."

# After (structured template):
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial analyst expert in the Indian stock market."),
    ("human", "Company: {company_name} ({ticker})\nSector: {sector}\n..."),
])

# Invoke with named variables:
prompt.invoke({"company_name": "National Aluminium", "ticker": "NATIONALUM.NS", "sector": "Mining"})
```

**📺 Video:** [LangChain Prompt Templates Explained](https://www.youtube.com/watch?v=RflBcK0oDH0) by Sam Witteveen (prompt types, message roles, variable substitution)

---

#### Concept 3: LCEL Chain (LangChain Expression Language) — The `|` Pipe

**What it is:** LCEL lets you compose LangChain components using the `|` (pipe) operator, like Unix pipes. Data flows left-to-right through each component: `prompt | llm | parser`. This is the modern, recommended way to build chains in LangChain (replacing the legacy `LLMChain`).

**Where we use it:**
- Peer identification: `prompt | groq_llm | StrOutputParser()`
- Comparison insights: `prompt | groq_llm | StrOutputParser()`

**Why it matters:** Each component is independent and testable. You can insert logging, caching, or fallback logic at any point in the pipe.

```python
from langchain_core.output_parsers import StrOutputParser

# Build a chain with the pipe operator:
chain = prompt | llm | StrOutputParser()

# Invoke — data flows through each stage:
result = chain.invoke({"ticker": "NATIONALUM.NS", "sector": "Mining", ...})
# 1. prompt.invoke() → formats the ChatPromptTemplate into messages
# 2. llm.invoke()    → sends messages to Groq, gets AIMessage back
# 3. StrOutputParser().invoke() → extracts .content string from AIMessage
```

**📺 Video:** [LCEL Explained — LangChain Expression Language](https://www.youtube.com/watch?v=O_o7-JLsVLk) by James Briggs (pipe operator, Runnable interface, how data flows through chains)

---

#### Concept 4: Tool / @tool — Letting LLMs Call Python Functions

**What it is:** A LangChain `Tool` wraps a Python function so that an LLM agent can decide to call it. The `@tool` decorator converts any function into a tool — the function's docstring becomes the description the LLM reads to decide when to use it.

**Where we use it:**
- `get_stock_price_tool` — LLM calls this to fetch live stock prices
- `get_stock_financials_tool` — LLM calls this for PE ratio, revenue growth, etc.
- `get_stock_historical_performance_tool` — LLM calls this for 1Y/2Y/3Y returns

**Why it matters:** This is what makes an "agent" different from a simple chatbot. The LLM can autonomously decide which tools to call, in what order, and how to combine the results. The docstring is critical — it's the LLM's only guide.

```python
from langchain_core.tools import tool
import json

@tool
def get_stock_price_tool(ticker: str) -> str:
    """Fetch current stock price and basic information for a given ticker symbol.
    Use this when you need the current price, volume, market cap, or daily change
    for a stock. Input should be a valid stock ticker like 'NATIONALUM.NS'."""
    result = get_stock_price(ticker)   # existing yfinance function
    return json.dumps(result, indent=2)

# The LLM sees:
#   Tool name: get_stock_price_tool
#   Description: "Fetch current stock price and basic information..."
#   Input: ticker (str)
```

**📺 Video:** [LangChain Tools & Function Calling Deep Dive](https://www.youtube.com/watch?v=q-HNphrWsDE) by Krish Naik (creating tools, @tool decorator, tool schema, how LLMs see tools)

---

#### Concept 5: Agent — The LLM That Decides What To Do

**What it is:** An Agent is an LLM that has been given tools and a prompt, and can autonomously decide: (a) which tool to call, (b) with what arguments, (c) whether to call another tool or return a final answer. `create_tool_calling_agent()` creates an agent that uses the LLM's native tool-calling API.

**Where we use it:**
- **Gemini agent** (Step 1): Given `get_stock_price_tool` → decides to fetch price → processes result → writes analysis
- **Groq agent** (Step 3): Given 3 tools → decides to fetch price, financials, performance → processes all → writes deep-dive

**Why it matters:** This replaces our entire manual function-calling loop in `main.py` (the `while iteration < max_iterations` loop). LangChain handles the loop, error recovery, and result aggregation.

```python
from langchain.agents import create_tool_calling_agent

# The agent is a "runnable" — it doesn't execute yet
agent = create_tool_calling_agent(llm, tools, prompt)

# Think of it as: "Here's an LLM, here are tools it can call, here's its instructions"
# The agent itself is just a decision-maker — it needs an executor to run
```

**📺 Video:** [LangChain Agents Explained](https://www.youtube.com/watch?v=cACbj7mhMCA) by AI Jason (agent concepts, ReAct pattern, how agents reason about tool use)

---

#### Concept 6: AgentExecutor — The Agent Runtime Loop

**What it is:** `AgentExecutor` takes an Agent and runs the execution loop: call LLM → if LLM wants to use a tool → execute the tool → feed result back → repeat until LLM returns a final answer. It handles max iterations, error handling, and timeouts.

**Where we use it:**
- Wraps both the Gemini agent (Step 1) and the Groq agent (Step 3)
- `verbose=True` shows the full reasoning trace (tool calls, results, decisions)

**Why it matters:** This is the biggest code simplification. Our manual 40-line `while` loop in `main.py` becomes a single `executor.invoke()` call.

```python
from langchain.agents import AgentExecutor

executor = AgentExecutor(
    agent=agent,          # the decision-maker
    tools=tools,          # functions the agent can call
    verbose=True,         # show reasoning trace
    max_iterations=10,    # safety limit
)

# One call replaces our entire manual function-calling loop:
result = executor.invoke({"input": "Analyze NATIONALUM.NS"})
print(result["output"])   # final analysis text

# Behind the scenes, AgentExecutor runs:
# 1. LLM decides: "I should call get_stock_price_tool('NATIONALUM.NS')"
# 2. Executor runs: get_stock_price_tool("NATIONALUM.NS") → returns JSON
# 3. LLM sees result, decides: "I have enough data, here's my analysis..."
# 4. Executor returns: {"output": "National Aluminium is trading at ₹359..."}
```

**📺 Video:** [Build AI Agents with LangChain](https://www.youtube.com/watch?v=jSP-gSEyVeI) by Tech With Tim (complete agent + executor walkthrough, building from scratch, verbose tracing)

---

#### Concept 7: Multi-LLM Orchestration — Combining Everything

**What it is:** Using multiple different LLMs in a single application, each for the task it's best suited for. LangChain's unified interface makes this seamless — you create different ChatModel instances and use them in different chains/agents.

**Where we use it:**
- **Gemini** → Step 1 (base stock analysis) — strong function-calling support
- **Groq** → Steps 2 & 3 (peer ID, comparison, top peer deep-dive) — fast inference, generous free tier

**Why it matters:** This is the real-world pattern. Production AI apps rarely use a single model. You route different tasks to different LLMs based on cost, speed, capability, and quota limits.

```python
# Same orchestration code, different LLMs:
gemini_agent = AgentExecutor(agent=gemini_based_agent, tools=tools)
groq_agent   = AgentExecutor(agent=groq_based_agent, tools=tools)

# Step 1 uses Gemini
base_analysis = gemini_agent.invoke({"input": "Analyze NATIONALUM.NS"})

# Step 3 uses Groq (same tools, different LLM)
peer_analysis = groq_agent.invoke({"input": "Analyze HINDALCO.NS"})
```

**📺 Video:** [Multi-Model AI Agents with LangChain](https://www.youtube.com/watch?v=xUKGNRKYwhA) by Dave Ebbelaar (routing between models, cost optimization, practical patterns)

---

### 📚 Bonus: Full Course References

For a complete LangChain learning path beyond this project:

| Resource | Link | Best For |
|---|---|---|
| **LangChain Official Tutorials** | [python.langchain.com/docs/tutorials](https://python.langchain.com/docs/tutorials/) | Definitive reference |
| **LangChain Crash Course (1hr)** | [youtube.com/watch?v=aywZrzNaKjs](https://www.youtube.com/watch?v=aywZrzNaKjs) by Tech With Tim | Quick full overview |
| **LangChain Master Class (4hr)** | [youtube.com/watch?v=yF9kGESAi3M](https://www.youtube.com/watch?v=yF9kGESAi3M) by Brandon Hancock | Deep dive with projects |
| **LangChain for LLM App Dev** | [deeplearning.ai/short-courses/langchain-for-llm-application-development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) | Andrew Ng + Harrison Chase (free) |
| **LCEL Deep Dive** | [youtube.com/watch?v=O_o7-JLsVLk](https://www.youtube.com/watch?v=O_o7-JLsVLk) by James Briggs | Understanding the pipe |

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

> **🎓 Concepts introduced:** Concept 1 (ChatModel) — installing the LLM provider packages

### 1.1 Update requirements.txt

```txt
google-genai
yfinance
python-dotenv
groq>=0.4.0
pandas>=2.0.0
langchain>=0.3.0
langchain-google-genai>=2.0.0    # ← Concept 1: Gemini ChatModel
langchain-groq>=0.2.0             # ← Concept 1: Groq ChatModel
langchain-community>=0.3.0
```

**What's happening:** Each `langchain-<provider>` package gives us a `ChatModel` class that wraps that provider's API behind LangChain's unified interface.

### 1.2 Update config.py

```python
# LangChain settings
LANGCHAIN_VERBOSE = False  # Set True for debug chain traces

# Gemini via LangChain (Concept 1: ChatModel config)
GEMINI_MODEL_LANGCHAIN = 'gemini-2.5-flash'
GEMINI_TEMPERATURE = 0.3

# Groq via LangChain (Concept 1: ChatModel config)
GROQ_MODEL_LANGCHAIN = 'llama-3.3-70b-versatile'
GROQ_TEMPERATURE_PEERS = 0.2       # Deterministic for peer identification
GROQ_TEMPERATURE_ANALYSIS = 0.4    # Slightly creative for narratives
```

---

## Phase 2: LangChain Tool Wrappers (tools.py)

> **🎓 Concepts introduced:** Concept 4 (Tool / @tool) — making Python functions callable by LLM agents

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

> **🎓 Concepts introduced:** Concept 2 (PromptTemplate), Concept 3 (LCEL Chain), Concept 5 (Agent), Concept 6 (AgentExecutor)
>
> This is the most concept-dense phase. Take it step by step.

### 3.1 Gemini Stock Analysis Chain — 🎓 Concepts 2 + 5 + 6

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

### 3.2 Groq Peer Identification Chain — 🎓 Concepts 2 + 3

Simple LLM chain (no tools needed — just prompting). This is where you learn LCEL pipes.

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

### 3.3 Groq Comparison Insights Chain — 🎓 Concept 3 (reinforcement)

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

### 3.5 Groq Top Peer Analysis Chain ← NEW — 🎓 Concepts 5 + 6 (reinforcement with different LLM)

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

> **🎓 Concepts introduced:** Concept 7 (Multi-LLM Orchestration) — tying everything together

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

### Step 1: Foundation (Est. 30 min) — 🎓 Concept 1: ChatModel
- [ ] Update `requirements.txt` with LangChain packages
- [ ] `pip install -r requirements.txt`
- [ ] Update `config.py` with LangChain settings
- [ ] Verify imports: `from langchain_google_genai import ChatGoogleGenerativeAI`
- [ ] Quick test: instantiate both ChatModels and call `.invoke("hello")`

### Step 2: Tool Wrappers (Est. 20 min) — 🎓 Concept 4: Tool / @tool
- [ ] Add `@tool` wrappers in `tools.py` for existing functions
- [ ] Test tool wrappers return valid JSON strings
- [ ] Ensure tool docstrings are clear (LLM reads them)
- [ ] Inspect tool schema: `print(get_stock_price_tool.args_schema.schema())`

### Step 3: Create chains.py (Est. 45 min) — 🎓 Concepts 2, 3, 5, 6
- [ ] Implement `create_gemini_stock_analyzer()` — **Concept 2** (PromptTemplate) + **Concept 5** (Agent) + **Concept 6** (AgentExecutor)
- [ ] Implement `create_peer_identifier_chain()` — **Concept 2** (PromptTemplate) + **Concept 3** (LCEL pipe)
- [ ] Implement `create_comparison_insights_chain()` — **Concept 3** reinforcement
- [ ] Implement `select_top_peer()` — Python scoring logic (no LangChain concept)
- [ ] Implement `create_groq_stock_analyzer()` — **Concepts 5+6** reinforcement with Groq
- [ ] Test each chain independently

### Step 4: Update main.py (Est. 30 min) — 🎓 Concept 7: Multi-LLM Orchestration
- [ ] Refactor `main()` to use LangChain chains
- [ ] Wire up the 3-step flow: Gemini → Peer Comparison → Top Peer
- [ ] Keep `--compare-peers` and `--verbose` flags
- [ ] Preserve report formatting
- [ ] Observe: same `.invoke()` method works across Gemini and Groq

### Step 5: Test & Cleanup (Est. 30 min)
- [ ] End-to-end test with NATIONALUM.NS
- [ ] Run with `LANGCHAIN_VERBOSE=True` to see full agent reasoning traces
- [ ] Verify Gemini free-tier usage (should be 1-2 calls only)
- [ ] Verify Groq handles all peer + top-peer calls
- [ ] Deprecate/remove `groq_analyzer.py` (logic now in `chains.py`)
- [ ] Update Confluence documentation

---

## LangChain Concepts — Summary Matrix

Quick reference: which concept is learned where, and what it replaces in our old code.

| # | Concept | Phase | Old Code It Replaces | LangChain Class | 📺 Video |
|---|---|---|---|---|---|
| 1 | **ChatModel** | Phase 1 | `genai.Client()`, `Groq()` | `ChatGoogleGenerativeAI`, `ChatGroq` | [Krish Naik: Chat Models](https://www.youtube.com/watch?v=wVwxSzOF3Fo) |
| 2 | **PromptTemplate** | Phase 3 | f-string prompts in `main.py` & `groq_analyzer.py` | `ChatPromptTemplate.from_messages()` | [Sam Witteveen: Prompt Templates](https://www.youtube.com/watch?v=RflBcK0oDH0) |
| 3 | **LCEL Chain** | Phase 3 | Manual Groq API calls | `prompt \| llm \| StrOutputParser()` | [James Briggs: LCEL](https://www.youtube.com/watch?v=O_o7-JLsVLk) |
| 4 | **Tool / @tool** | Phase 2 | Manually dispatched `get_stock_price()` calls | `@tool` decorator, `langchain_core.tools` | [Krish Naik: Tools](https://www.youtube.com/watch?v=q-HNphrWsDE) |
| 5 | **Agent** | Phase 3 | `if function_name == "get_stock_price"` dispatch | `create_tool_calling_agent()` | [AI Jason: Agents](https://www.youtube.com/watch?v=cACbj7mhMCA) |
| 6 | **AgentExecutor** | Phase 3 | 40-line `while iteration < max` loop | `AgentExecutor(agent, tools)` | [Tech With Tim: AI Agents](https://www.youtube.com/watch?v=jSP-gSEyVeI) |
| 7 | **Multi-LLM** | Phase 4 | Separate codepaths for Gemini vs Groq | Same `.invoke()` on different ChatModels | [Dave Ebbelaar: Multi-Model](https://www.youtube.com/watch?v=xUKGNRKYwhA) |

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

---

## 🎓 Learning Checkpoint — What You'll Know After This Project

After implementing all 5 phases, you'll have hands-on experience with:

| ✅ | Skill | Proof |
|---|---|---|
| ☐ | Initialize any LLM through LangChain's unified interface | You used both Gemini and Groq via `ChatModel` |
| ☐ | Write structured, reusable prompts | You built 4 different `ChatPromptTemplate` instances |
| ☐ | Compose chains with the `\|` pipe operator | You built 2 LCEL chains for Groq calls |
| ☐ | Wrap Python functions as LLM-callable tools | You decorated 3 yfinance functions with `@tool` |
| ☐ | Build autonomous agents that use tools | You created 2 agents (Gemini + Groq) |
| ☐ | Run agents with AgentExecutor | You replaced a manual loop with `executor.invoke()` |
| ☐ | Orchestrate multiple LLMs in one app | You routed Gemini for analysis, Groq for peers |

**What's next after this project:**
- Memory / Conversation History → `ConversationBufferMemory`, `ChatMessageHistory`
- RAG (Retrieval Augmented Generation) → `VectorStore`, `Retriever`, document loaders
- LangGraph → Stateful multi-agent workflows with branching and cycles
- LangSmith → Observability, tracing, and evaluation for production

📺 **Next steps video:** [LangGraph Explained](https://www.youtube.com/watch?v=PqS1kib7RTw) by Sam Witteveen

---

**Status:** Plan Ready — Awaiting Implementation  
**Next Step:** Begin Phase 1 (Dependencies & Configuration)
