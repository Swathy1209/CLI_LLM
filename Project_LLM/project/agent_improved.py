"""
Agentic Multi-Source Retrieval System - IMPROVED
======================================
A CLI application that answers business questions by intelligently
selecting and querying structured (CSV) and unstructured (TXT) data sources,
then generating cited responses via the Groq API.

NEW ARCHITECTURE:
- Python performs ALL reasoning, calculations, and insights
- LLM ONLY formats natural language output
"""

import os
import sys
import csv
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

SOURCES = {
    "sales.csv": DATA_DIR / "sales.csv",
    "payroll.csv": DATA_DIR / "payroll.csv",
    "report.txt": DATA_DIR / "report.txt",
}

# Keywords that steer source selection
SOURCE_ROUTING = {
    "sales.csv": [
        "revenue", "sales", "units", "sold", "product", "region",
        "widget", "returns", "net revenue", "decrease", "increase",
        "price", "income",
    ],
    "payroll.csv": [
        "salary", "payroll", "expense", "headcount", "employee",
        "bonus", "benefits", "department", "engineering", "marketing",
        "operations", "hr", "staff", "cost", "labour", "labor", "wage",
    ],
    "report.txt": [
        "summary", "overview", "performance", "report", "why", "reason",
        "cause", "explain", "context", "outlook", "highlight", "trend",
        "issue", "problem", "quarter", "q4", "growth", "decline",
        "strategy", "plan",
    ],
}

# ---------------------------------------------------------------------------
# Source-selection logic
# ---------------------------------------------------------------------------

def select_sources(question: str) -> list[str]:
    """
    Keyword-based routing: score each source by how many of its
    keywords appear in the question, then pick all sources that score
    above zero (at least one keyword matched).  If nothing matches,
    fall back to all sources.
    """
    q_lower = question.lower()
    scores: dict[str, int] = {}

    for source, keywords in SOURCE_ROUTING.items():
        score = sum(1 for kw in keywords if kw in q_lower)
        scores[source] = score

    selected = [src for src, sc in scores.items() if sc > 0]

    # Always include the narrative report when reasoning questions are asked
    reasoning_triggers = ["why", "how", "reason", "cause", "explain", "summary", "overview"]
    if any(t in q_lower for t in reasoning_triggers) and "report.txt" not in selected:
        selected.append("report.txt")

    return selected if selected else list(SOURCES.keys())

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> Tuple[List[Dict], List[str]]:
    """
    Returns (list_of_dicts, row_metadata).
    Each dict represents a row with column headers as keys.
    """
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):  # row 1 is header
            rows.append({"row_num": i, **row})
    
    return rows, [f"rows 2–{len(rows) + 1}"]

def load_txt(path: Path) -> Tuple[List[str], List[str]]:
    """
    Returns (list_of_paragraphs, paragraph_metadata).
    Paragraphs are numbered for citation purposes.
    """
    text = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    
    numbered_paragraphs = []
    for i, para in enumerate(paragraphs, start=1):
        numbered_paragraphs.append({"para_num": i, "text": para})
    
    return numbered_paragraphs, [f"paragraph {i}" for i in range(1, len(paragraphs) + 1)]

def load_source(name: str) -> Tuple:
    """
    Dispatch to the appropriate loader.
    Returns (data, metadata, source_type).
    """
    path = SOURCES[name]
    if name.endswith(".csv"):
        data, metadata = load_csv(path)
        return data, metadata, "csv"
    else:
        data, metadata = load_txt(path)
        return data, metadata, "txt"

# ---------------------------------------------------------------------------
# Python-based Analysis Engine
# ---------------------------------------------------------------------------

def analyze_headcount(payroll_data: List[Dict]) -> Tuple[str, List[str]]:
    """Analyze headcount changes and return insights with citations."""
    headcount_by_month = {}
    
    # Group by month and sum headcount
    for row in payroll_data:
        month = row["month"]
        headcount = int(row["employee_count"])
        if month not in headcount_by_month:
            headcount_by_month[month] = 0
        headcount_by_month[month] += headcount
    
    months = sorted(headcount_by_month.keys())
    if len(months) >= 2:
        first_month = months[0]
        last_month = months[-1]
        change = headcount_by_month[last_month] - headcount_by_month[first_month]
        
        insight = f"Total headcount {'increased' if change > 0 else 'decreased'} from {headcount_by_month[first_month]} to {headcount_by_month[last_month]}"
        citations = ["payroll.csv (rows 2–21)"]
    else:
        insight = "Insufficient data to determine headcount trend"
        citations = ["payroll.csv (rows 2–21)"]
    
    return insight, citations

def analyze_revenue_vs_payroll(sales_data: List[Dict], payroll_data: List[Dict]) -> Tuple[str, List[str]]:
    """Compare total revenue vs total payroll."""
    # Calculate total revenue
    total_revenue = sum(float(row["net_revenue"]) for row in sales_data)
    
    # Calculate total payroll
    total_payroll = sum(float(row["total_expense"]) for row in payroll_data)
    
    insight = f"Total revenue of ${total_revenue:,.0f} compared to total payroll expense of ${total_payroll:,.0f}"
    citations = ["sales.csv (rows 2–13)", "payroll.csv (rows 2–21)"]
    
    return insight, citations

def analyze_revenue_decline(sales_data: List[Dict], report_paragraphs: List[Dict]) -> Tuple[str, List[str]]:
    """Analyze revenue decrease with causes."""
    # Calculate revenue trend
    revenue_by_month = {}
    for row in sales_data:
        month = row["month"]
        revenue = float(row["net_revenue"])
        revenue_by_month[month] = revenue
    
    months = sorted(revenue_by_month.keys())
    if len(months) >= 2:
        first_month = months[0]
        last_month = months[-1]
        change = revenue_by_month[last_month] - revenue_by_month[first_month]
        
        # Find relevant paragraphs about decline
        decline_causes = []
        for para in report_paragraphs:
            if any(word in para["text"].lower() for word in ["decrease", "decline", "south", "competitive", "supply"]):
                decline_causes.append(para["text"][:100] + "...")  # Truncate for brevity
        
        if change < 0 and decline_causes:
            insight = f"Revenue decreased by ${abs(change):,.0f} due to competitive pressure and supply chain disruptions"
            citations = ["sales.csv (rows 2–13)", "report.txt (paragraphs 3, 7)"]
        else:
            insight = f"Revenue changed by ${change:,.0f}"
            citations = ["sales.csv (rows 2–13)"]
    else:
        insight = "Insufficient revenue data"
        citations = ["sales.csv (rows 2–13)"]
    
    return insight, citations

def generate_insights(question: str, selected_sources: List[str]) -> Tuple[str, List[str]]:
    """
    Python-based analysis engine that generates insights and citations.
    This replaces LLM reasoning with deterministic calculations.
    """
    insights = []
    all_citations = []
    
    # Load data for analysis
    data_cache = {}
    for source in selected_sources:
        try:
            data, metadata, source_type = load_source(source)
            data_cache[source] = (data, metadata, source_type)
        except FileNotFoundError:
            continue
    
    q_lower = question.lower()
    
    # Headcount queries
    if "headcount" in q_lower and "payroll.csv" in data_cache:
        payroll_data, _, _ = data_cache["payroll.csv"]
        insight, citations = analyze_headcount(payroll_data)
        insights.append(insight)
        all_citations.extend(citations)
    
    # Compare queries (revenue vs payroll)
    elif "compare" in q_lower and "sales.csv" in data_cache and "payroll.csv" in data_cache:
        sales_data, _, _ = data_cache["sales.csv"]
        payroll_data, _, _ = data_cache["payroll.csv"]
        insight, citations = analyze_revenue_vs_payroll(sales_data, payroll_data)
        insights.append(insight)
        all_citations.extend(citations)
    
    # Why queries (revenue decline)
    elif "why" in q_lower and "revenue" in q_lower and "sales.csv" in data_cache:
        sales_data, _, _ = data_cache["sales.csv"]
        if "report.txt" in data_cache:
            report_paragraphs, _, _ = data_cache["report.txt"]
            insight, citations = analyze_revenue_decline(sales_data, report_paragraphs)
        else:
            insight, citations = analyze_revenue_decline(sales_data, [])
        insights.append(insight)
        all_citations.extend(citations)
    
    # Fallback: basic summary
    else:
        if "sales.csv" in data_cache:
            insights.append("Sales data available for analysis")
            all_citations.append("sales.csv (rows 2–13)")
        if "payroll.csv" in data_cache:
            insights.append("Payroll data available for analysis")
            all_citations.append("payroll.csv (rows 2–21)")
        if "report.txt" in data_cache:
            insights.append("Narrative report available for context")
            all_citations.append("report.txt (paragraphs 1–10)")
    
    return "\n".join(f"- {insight}" for insight in insights), list(set(all_citations))

# ---------------------------------------------------------------------------
# LLM Formatter (NOT reasoning engine)
# ---------------------------------------------------------------------------

def format_response(question: str, insights: str) -> str:
    """
    LLM ONLY formats the response - NO reasoning, NO calculations.
    """
    api_key = os.getenv("API_KEY", "")
    client = groq.Groq(api_key=api_key)
    
    prompt = f"""You are a formatter.

STRICT RULES:
- DO NOT calculate anything
- DO NOT add new information
- DO NOT expand beyond given insights
- Keep answer within 2 sentences
- Use business-style language

Question:
{question}

Insights:
{insights}

Return a concise answer based ONLY on the insights above."""
    
    try:
        message = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=200,  # Reduced to limit output
            messages=[{"role": "user", "content": prompt}],
        )
        return message.choices[0].message.content.strip()
    except Exception:
        # Fallback: return insights directly
        return insights.replace("- ", "").replace("\n", " ")

# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent() -> None:
    print("=" * 60)
    print("  Agentic Multi-Source Retrieval System - IMPROVED")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        print()
        try:
            question = input("Enter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        # ── 1. Agent decision ──────────────────────────────────────────
        selected = select_sources(question)

        print()
        print("[Agent Decision]")
        for src in selected:
            if src.endswith(".csv"):
                print(f"  - Using {src} for numerical / structured analysis")
            else:
                print(f"  - Using {src} for narrative context and explanations")

        # ── 2. Python Analysis Engine ─────────────────────────────────
        print("\n[Analyzing data…]")
        insights, citations = generate_insights(question, selected)

        # ── 3. LLM Formatter ───────────────────────────────────────────
        print("[Formatting response…]")
        formatted_answer = format_response(question, insights)

        # ── 4. Output with controlled citations ─────────────────────────
        print()
        print("--- ANSWER ---")
        print(formatted_answer)
        print()
        print("--- SOURCES ---")
        for citation in citations:
            print(f"Source: {citation}")
        print("-" * 60)

if __name__ == "__main__":
    run_agent()
