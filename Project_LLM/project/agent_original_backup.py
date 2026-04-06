"""
Agentic Multi-Source Retrieval System
======================================
A CLI application that answers business questions by intelligently
selecting and querying structured (CSV) and unstructured (TXT) data sources,
then generating cited responses via the Anthropic API.
"""

import os
import sys
import csv
import re
import textwrap
from pathlib import Path

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

def load_csv(path: Path) -> tuple[str, list[str]]:
    """
    Returns (formatted_table_string, row_range_metadata).
    The formatted string is a compact CSV representation with row numbers.
    """
    rows: list[list[str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return "", []

    header = rows[0]
    data_rows = rows[1:]

    lines = ["Row | " + " | ".join(header)]
    lines.append("-" * 80)
    for i, row in enumerate(data_rows, start=2):   # row 1 is header
        lines.append(f"{i:3d} | " + " | ".join(row))

    return "\n".join(lines), [f"rows 2–{len(rows)}"]


def load_txt(path: Path) -> tuple[str, list[str]]:
    """
    Returns (full_text, paragraph_metadata).
    Paragraphs are numbered for citation purposes.
    """
    text = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    numbered: list[str] = []
    for i, para in enumerate(paragraphs, start=1):
        numbered.append(f"[Paragraph {i}]\n{para}")

    return "\n\n".join(numbered), [f"paragraph {i}" for i in range(1, len(paragraphs) + 1)]


def load_source(name: str) -> tuple[str, str]:
    """
    Dispatch to the appropriate loader.
    Returns (content_string, source_label_for_prompt).
    """
    path = SOURCES[name]
    if name.endswith(".csv"):
        content, _ = load_csv(path)
        label = f"SOURCE: {name} (structured CSV data)"
    else:
        content, _ = load_txt(path)
        label = f"SOURCE: {name} (narrative report)"
    return content, label


# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------

def build_prompt(question: str, source_blocks: list[tuple[str, str]]) -> str:
    """
    Assemble the user message that will be sent to Claude.
    """
    source_text = "\n\n".join(
        f"--- {label} ---\n{content}"
        for label, content in source_blocks
    )

    return textwrap.dedent(f"""
        You are a business analyst assistant. Answer the following question using ONLY
        the data provided below.  Your response MUST end with a "Sources:" section that
        cites exactly which file and which rows / paragraphs you used.

        Citation format examples:
          Source: sales.csv (rows 5–12)
          Source: report.txt (paragraph 3)

        Be concise and factual.  If the data is insufficient to fully answer, say so.

        QUESTION:
        {question}

        DATA:
        {source_text}
    """).strip()


def query_llm(prompt: str) -> str:
    """
    Send the prompt to Groq and return the text response.
    """
    api_key = "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"
    client = groq.Groq(api_key=api_key)
    message = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.choices[0].message.content


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent() -> None:
    print("=" * 60)
    print("  Agentic Multi-Source Retrieval System")
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

        # ── 2. Load selected sources ───────────────────────────────────
        source_blocks: list[tuple[str, str]] = []
        for src in selected:
            try:
                content, label = load_source(src)
                source_blocks.append((label, content))
            except FileNotFoundError:
                print(f"  [Warning] {src} not found — skipping.")

        if not source_blocks:
            print("[Error] No data sources could be loaded.")
            continue

        # ── 3. Build prompt & call LLM ────────────────────────────────
        print("\n[Retrieving answer…]")
        prompt = build_prompt(question, source_blocks)

        try:
            answer = query_llm(prompt)
        except groq.APIConnectionError:
            print("[Error] Could not connect to the Groq API.")
            print("  Make sure GROQ_API_KEY is set and you have internet access.")
            continue
        except groq.AuthenticationError:
            print("[Error] Invalid API key.  Set the GROQ_API_KEY environment variable.")
            sys.exit(1)
        except Exception as exc:          # noqa: BLE001
            print(f"[Error] API call failed: {exc}")
            continue

        # ── 4. Print answer ────────────────────────────────────────────
        print()
        print("─" * 60)
        print(answer)
        print("─" * 60)


if __name__ == "__main__":
    run_agent()
