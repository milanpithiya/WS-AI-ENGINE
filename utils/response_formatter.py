"""
Response Formatter — Cleans and formats LLM responses.
Ensures consistent output format regardless of model quirks.
"""

import re
from typing import Optional


def clean_response(text: str) -> str:
    """
    Clean LLM response:
    - Remove leading/trailing whitespace
    - Fix common markdown issues
    - Remove model artifacts (e.g., "As an AI language model...")
    """
    if not text:
        return ""

    # Strip whitespace
    text = text.strip()

    # Remove common AI self-references
    remove_patterns = [
        r"^As an AI( language model)?,?\s*",
        r"^I'm an AI( assistant)?,?\s*",
        r"^Based on my training,?\s*",
    ]
    for pattern in remove_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Fix double line breaks (more than 2 consecutive)
    text = re.sub(r'\n{4,}', '\n\n', text)

    return text.strip()


def format_currency_inr(amount: float) -> str:
    """Format amount in Indian currency style (e.g., 12,50,000 or 1.25 Cr)."""
    if amount >= 10_000_000:  # 1 Cr+
        return "{0:.2f} Cr".format(amount / 10_000_000)
    elif amount >= 100_000:  # 1 Lakh+
        return "{0:.2f} L".format(amount / 100_000)
    else:
        return "{0:,.0f}".format(amount)


def extract_sections(text: str) -> dict:
    """
    Extract markdown sections from response.
    Returns dict of {section_heading: content}.
    """
    sections = {}
    current_heading = "Summary"
    current_content = []

    for line in text.split("\n"):
        if line.startswith("## ") or line.startswith("### "):
            # Save previous section
            if current_content:
                sections[current_heading] = "\n".join(current_content).strip()
            current_heading = line.lstrip("#").strip()
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_content:
        sections[current_heading] = "\n".join(current_content).strip()

    return sections
