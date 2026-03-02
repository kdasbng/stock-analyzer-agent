"""
LangChain chains and agents for Stock Analyzer.

Concepts covered:
  1. ChatModel         — ChatGoogleGenerativeAI, ChatGroq
  2. PromptTemplate    — ChatPromptTemplate.from_messages()
  3. LCEL Chain        — prompt | llm | StrOutputParser()
  4. Tool / @tool      — (defined in tools.py)
  5. Agent             — create_tool_calling_agent()
  6. AgentExecutor     — AgentExecutor(agent, tools)
  7. Multi-LLM         — same .invoke() across Gemini & Groq
"""

import logging
import re
from typing import Any, Dict, List, Optional

# ── Concept 1: ChatModels ───────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ── Concept 2: PromptTemplate ──────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate

# ── Concept 3: LCEL components ─────────────────────────────────
from langchain_core.output_parsers import StrOutputParser

# ── Concepts 5 & 6: Agent (create_agent replaces AgentExecutor) ──
from langchain.agents import create_agent

import config
from tools import (
    get_stock_price_tool,
    get_stock_financials_tool,
    get_stock_historical_performance_tool,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 3.1  Gemini Stock Analysis Agent  (Concepts 1, 2, 5, 6)
# ═══════════════════════════════════════════════════════════════════

def create_gemini_stock_analyzer():
    """
    Create a LangChain agent that uses Gemini to analyze stocks.
    Gemini can call yfinance tools to fetch live data.

    Concepts used:
      - Concept 1 (ChatModel): ChatGoogleGenerativeAI
      - Concept 5 (Agent): create_agent — tool-calling agent
      - Concept 6 (AgentExecutor): create_agent returns a runnable graph

    Returns a CompiledStateGraph with .invoke({'messages': [...]}) interface.
    """
    # Concept 1: Instantiate the Gemini ChatModel
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_LANGCHAIN,
        google_api_key=config.GEMINI_API_KEY,
        temperature=config.GEMINI_TEMPERATURE,
    )

    # Concept 4: Tools the agent can call (defined in tools.py)
    tools = [get_stock_price_tool]

    system_prompt = (
        "You are a stock market analyst. When analyzing stocks, "
        "use the available tools to fetch real-time data. Provide concise, "
        "data-driven insights including current performance, concerns, "
        "opportunities, and actionable recommendations."
    )

    # Concept 5+6: create_agent builds and compiles the agent graph
    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
        debug=config.LANGCHAIN_VERBOSE,
    )


# ═══════════════════════════════════════════════════════════════════
# 3.2  Groq Peer Identification Chain  (Concepts 1, 2, 3)
# ═══════════════════════════════════════════════════════════════════

def create_peer_identifier_chain():
    """
    LCEL chain that asks Groq to identify industry peers.
    Returns raw text containing ticker symbols.

    Concepts used:
      - Concept 1 (ChatModel): ChatGroq
      - Concept 2 (PromptTemplate): ChatPromptTemplate.from_messages
      - Concept 3 (LCEL): prompt | llm | StrOutputParser()
    """
    # Concept 1: Groq ChatModel with low temperature for determinism
    llm = ChatGroq(
        model=config.GROQ_MODEL_LANGCHAIN,
        api_key=config.GROQ_API_KEY,
        temperature=config.GROQ_TEMPERATURE_PEERS,
    )

    # Concept 2: Prompt with placeholders for company info
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst expert in the Indian stock market."),
        ("human", """Company: {company_name} ({ticker})
Sector: {sector}
Industry: {industry}

Identify the top 3 publicly traded competitors listed on NSE/BSE.
Do NOT include the base company ({ticker}) itself.
Provide ONLY ticker symbols, one per line:
1. TICKER1.NS
2. TICKER2.NS
3. TICKER3.NS

Use .NS (NSE) suffix by default. Only use .BO (BSE) if exclusively on BSE."""),
    ])

    # Concept 3: LCEL pipe — prompt → LLM → parse string output
    return prompt | llm | StrOutputParser()


# ═══════════════════════════════════════════════════════════════════
# 3.3  Groq Comparison Insights Chain  (Concepts 1, 2, 3)
# ═══════════════════════════════════════════════════════════════════

def create_comparison_insights_chain():
    """
    LCEL chain that generates comparative analysis narratives using Groq.

    Concepts used:
      - Concept 1 (ChatModel): ChatGroq
      - Concept 2 (PromptTemplate): ChatPromptTemplate.from_messages
      - Concept 3 (LCEL): prompt | llm | StrOutputParser()
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


# ═══════════════════════════════════════════════════════════════════
# 3.4  Top Peer Selection (Pure Python — deterministic scoring)
# ═══════════════════════════════════════════════════════════════════

def select_top_peer(peers_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Score each peer and return the best one for a deep-dive analysis.

    Scoring:
      PE_score     = normalize(1 / pe_ratio)    weight 0.3  (lower PE → better)
      Growth_score = normalize(revenue_growth)   weight 0.3
      Return_score = normalize(return_1y)        weight 0.4  (momentum)

    Peers with errors or all-None metrics are excluded.
    """
    # Filter out peers with errors
    valid = [p for p in peers_data if "error" not in p]
    if not valid:
        logger.warning("No valid peers for top-peer selection.")
        return None

    def _val(d: dict, key: str) -> Optional[float]:
        v = d.get(key)
        return float(v) if v is not None else None

    # Collect raw values
    pe_vals = [1.0 / _val(p, "pe_ratio_trailing") for p in valid if _val(p, "pe_ratio_trailing")]
    growth_vals = [_val(p, "revenue_yoy_growth") for p in valid if _val(p, "revenue_yoy_growth") is not None]
    return_vals = [_val(p, "return_1y") for p in valid if _val(p, "return_1y") is not None]

    def _normalize(value: float, all_values: List[float]) -> float:
        if len(all_values) < 2:
            return 0.5  # default when there's only one value
        mn, mx = min(all_values), max(all_values)
        if mx == mn:
            return 0.5
        return (value - mn) / (mx - mn)

    best_peer = None
    best_score = -1.0

    for peer in valid:
        score = 0.0
        components = 0

        pe = _val(peer, "pe_ratio_trailing")
        if pe and pe > 0 and pe_vals:
            score += 0.3 * _normalize(1.0 / pe, pe_vals)
            components += 1

        growth = _val(peer, "revenue_yoy_growth")
        if growth is not None and growth_vals:
            score += 0.3 * _normalize(growth, growth_vals)
            components += 1

        ret = _val(peer, "return_1y")
        if ret is not None and return_vals:
            score += 0.4 * _normalize(ret, return_vals)
            components += 1

        # Normalize score by number of components to be fair
        if components > 0:
            score = score / (0.3 * min(components, 2) + 0.4 * min(components, 1))
            # Simpler: just use raw weighted score — higher is better
            pass

        logger.debug(f"Peer {peer.get('ticker')}: score={score:.4f} (components={components})")

        if score > best_score and components > 0:
            best_score = score
            best_peer = peer

    if best_peer:
        logger.info(
            f"Top peer selected: {best_peer.get('ticker')} "
            f"(score={best_score:.4f})"
        )
    return best_peer


# ═══════════════════════════════════════════════════════════════════
# 3.5  Groq Stock Analysis Agent — NEW  (Concepts 1, 2, 5, 6)
# ═══════════════════════════════════════════════════════════════════

def create_groq_stock_analyzer():
    """
    LangChain agent using Groq + tools for deep-dive analysis of the top peer.
    Demonstrates the same agent pattern working with a different LLM.

    Concepts used:
      - Concept 1 (ChatModel): ChatGroq — same interface as ChatGoogleGenerativeAI
      - Concept 5 (Agent): create_agent (same function, different LLM!)
      - Concept 6 (AgentExecutor): same graph-based execution

    Returns a CompiledStateGraph with .invoke({'messages': [...]}) interface.
    """
    llm = ChatGroq(
        model=config.GROQ_MODEL_LANGCHAIN,
        api_key=config.GROQ_API_KEY,
        temperature=config.GROQ_TEMPERATURE_ANALYSIS,
    )

    # Give the Groq agent all three tools for comprehensive analysis
    tools = [
        get_stock_price_tool,
        get_stock_financials_tool,
        get_stock_historical_performance_tool,
    ]

    system_prompt = (
        "You are a stock market analyst. Analyze the given stock "
        "using the available tools. Provide a comprehensive analysis including "
        "price action, valuation, growth metrics, historical performance, "
        "and investment recommendations."
    )

    # Concept 5+6: Same create_agent call, different LLM — that's the power!
    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
        debug=config.LANGCHAIN_VERBOSE,
    )


# ═══════════════════════════════════════════════════════════════════
#  Helper utilities
# ═══════════════════════════════════════════════════════════════════

def parse_peer_tickers(text: str, exclude_ticker: str = "") -> List[str]:
    """
    Extract ticker symbols (TICKER.NS or TICKER.BO) from raw text.
    Deduplicates and excludes the base ticker.
    """
    pattern = r'\b([A-Z0-9&]+(?:\.NS|\.BO))\b'
    matches = re.findall(pattern, text.upper())

    seen = set()
    tickers = []
    exclude_upper = exclude_ticker.upper()
    for t in matches:
        if t not in seen and t != exclude_upper:
            seen.add(t)
            tickers.append(t)
    return tickers[:config.MAX_PEERS_TO_ANALYZE]


def format_comparison_for_prompt(comparison: Dict[str, Any]) -> str:
    """
    Format the comparison data dict into a readable string for the LLM prompt.
    """
    def _fmt_stock(stock: Dict[str, Any], label: str) -> str:
        lines = [
            f"{label}: {stock.get('ticker', 'N/A')} ({stock.get('company_name', 'N/A')})",
            f"  PE Ratio (Trailing): {stock.get('pe_ratio_trailing', 'N/A')}",
            f"  Revenue Growth (YoY): {_pct(stock.get('revenue_yoy_growth'))}",
            f"  Profit Margin: {_pct(stock.get('profit_margin'))}",
            f"  Market Cap: {_fmt_cap(stock.get('market_cap'))}",
            f"  1-Year Return: {_pct(stock.get('return_1y'))}",
            f"  2-Year Return: {_pct(stock.get('return_2y'))}",
            f"  3-Year Return: {_pct(stock.get('return_3y'))}",
        ]
        return "\n".join(lines)

    parts = [_fmt_stock(comparison["base_stock"], "BASE STOCK")]
    for i, peer in enumerate(comparison.get("peers", []), 1):
        if "error" not in peer:
            parts.append(_fmt_stock(peer, f"PEER {i}"))
        else:
            parts.append(f"PEER {i}: {peer.get('ticker', 'N/A')} — data unavailable")
    return "\n\n".join(parts)


# ── Formatting helpers ──────────────────────────────────────────

def _pct(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def _fmt_cap(value) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000_000_000:
        return f"₹{value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"₹{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"₹{value / 1_000_000:.2f}M"
    return f"₹{value:,}"
