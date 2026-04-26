"""
Aggregator — Summarizes large datasets into compact text summaries.
Converts 150 holdings into a digestible summary for the LLM.
"""

from typing import Dict, List, Any, Optional
from loguru import logger


def aggregate_holdings(holdings: List[Dict[str, Any]], top_n: int = 10) -> str:
    """
    Aggregate holdings into a compact text summary.
    Instead of sending all holdings, creates:
      - Total value summary
      - Asset class breakdown
      - Sector breakdown
      - Top N holdings by value
      - Underperformers (negative or low returns)
    """
    if not holdings:
        return "No holdings data available."

    total_invested = sum(h.get("invested", 0) or 0 for h in holdings)
    total_current = sum(h.get("current", 0) or 0 for h in holdings)
    total_gain = total_current - total_invested
    total_gain_pct = (total_gain / total_invested * 100) if total_invested > 0 else 0

    # Group by asset class
    asset_groups = {}  # type: Dict[str, Dict[str, Any]]
    for h in holdings:
        ac = h.get("asset", h.get("asset_class", "Other"))
        if ac not in asset_groups:
            asset_groups[ac] = {"count": 0, "invested": 0, "current": 0}
        asset_groups[ac]["count"] += 1
        asset_groups[ac]["invested"] += h.get("invested", 0) or 0
        asset_groups[ac]["current"] += h.get("current", 0) or 0

    # Group by sector
    sector_groups = {}  # type: Dict[str, float]
    for h in holdings:
        sector = h.get("sector", "Other")
        if sector:
            weight = h.get("weight", 0) or 0
            sector_groups[sector] = sector_groups.get(sector, 0) + weight

    # Sort holdings by market value descending
    sorted_by_value = sorted(holdings, key=lambda x: x.get("current", 0) or 0, reverse=True)

    # Find underperformers
    underperformers = [
        h for h in holdings
        if (h.get("gain_pct", 0) or 0) < 0
    ]

    # Build summary
    lines = []
    lines.append("PORTFOLIO SUMMARY:")
    lines.append("Total Invested: {0:,.0f} | Current Value: {1:,.0f} | Gain: {2:,.0f} ({3:.1f}%)".format(
        total_invested, total_current, total_gain, total_gain_pct))
    lines.append("Total Holdings: {0}".format(len(holdings)))
    lines.append("")

    # Asset class breakdown
    lines.append("ASSET CLASS BREAKDOWN:")
    for ac, data in sorted(asset_groups.items(), key=lambda x: x[1]["current"], reverse=True):
        pct = (data["current"] / total_current * 100) if total_current > 0 else 0
        lines.append("  {0}: {1:,.0f} ({2:.1f}%) - {3} holdings".format(
            ac, data["current"], pct, data["count"]))
    lines.append("")

    # Top sectors
    if sector_groups:
        lines.append("TOP SECTORS (by weight):")
        sorted_sectors = sorted(sector_groups.items(), key=lambda x: x[1], reverse=True)[:8]
        for sector, weight in sorted_sectors:
            lines.append("  {0}: {1:.1f}%".format(sector, weight))
        lines.append("")

    # Top N holdings
    lines.append("TOP {0} HOLDINGS (by value):".format(min(top_n, len(sorted_by_value))))
    for h in sorted_by_value[:top_n]:
        name = h.get("name", "Unknown")
        current = h.get("current", 0) or 0
        gain = h.get("gain_pct", 0) or 0
        weight = h.get("weight", 0) or 0
        xirr = h.get("xirr", 0) or 0
        lines.append("  {0}: Value={1:,.0f} | Weight={2:.1f}% | Gain={3:.1f}% | XIRR={4:.1f}%".format(
            name, current, weight, gain, xirr))
    lines.append("")

    # Underperformers
    if underperformers:
        lines.append("UNDERPERFORMERS (negative returns):")
        sorted_under = sorted(underperformers, key=lambda x: x.get("gain_pct", 0) or 0)
        for h in sorted_under[:5]:
            name = h.get("name", "Unknown")
            gain = h.get("gain_pct", 0) or 0
            current = h.get("current", 0) or 0
            lines.append("  {0}: Value={1:,.0f} | Loss={2:.1f}%".format(name, current, gain))
        lines.append("")

    return "\n".join(lines)


def aggregate_performance(performance: List[Dict[str, Any]]) -> str:
    """Aggregate performance data into a readable summary."""
    if not performance:
        return "No performance data available."

    lines = []

    # Separate total (level="" or "Total") from asset class entries
    total_entry = None
    asset_entries = []
    for p in performance:
        level = p.get("level", "")
        name = p.get("name", "")
        if level == "" or name == "Total":
            total_entry = p
        else:
            asset_entries.append(p)

    # Portfolio-level summary
    if total_entry:
        lines.append("OVERALL PORTFOLIO PERFORMANCE:")
        lines.append("  Invested: {0:,.0f} | Current: {1:,.0f} | Gain: {2:,.0f}".format(
            total_entry.get("invested", 0) or 0,
            total_entry.get("current", 0) or 0,
            total_entry.get("gain", 0) or 0,
        ))
        lines.append("  XIRR: {0:.2f}% | Benchmark XIRR: {1:.2f}%".format(
            total_entry.get("xirr", 0) or 0,
            total_entry.get("bm_xirr", 0) or 0,
        ))
        lines.append("  Absolute Return: {0:.2f}% | BM Return: {1:.2f}%".format(
            total_entry.get("return_pct", 0) or 0,
            total_entry.get("bm_return_pct", 0) or 0,
        ))
        bm = total_entry.get("benchmark", "")
        period_from = total_entry.get("fromDate", "")
        period_to = total_entry.get("toDate", "")
        if bm:
            lines.append("  Benchmark: {0}".format(bm))
        if period_from and period_to:
            lines.append("  Period: {0} to {1}".format(period_from, period_to))
        lines.append("")

    # Asset class breakdown
    if asset_entries:
        lines.append("PERFORMANCE BY ASSET CLASS:")
        for p in sorted(asset_entries, key=lambda x: x.get("current", 0) or 0, reverse=True):
            lines.append("  {0}: Invested={1:,.0f} | Current={2:,.0f} | Gain={3:,.0f}".format(
                p.get("name", "Unknown"),
                p.get("invested", 0) or 0,
                p.get("current", 0) or 0,
                p.get("gain", 0) or 0,
            ))
            lines.append("    XIRR={0:.2f}% (BM: {1:.2f}%) | Return={2:.2f}% (BM: {3:.2f}%) | BM: {4}".format(
                p.get("xirr", 0) or 0,
                p.get("bm_xirr", 0) or 0,
                p.get("return_pct", 0) or 0,
                p.get("bm_return_pct", 0) or 0,
                p.get("benchmark", "N/A"),
            ))
        lines.append("")

    return "\n".join(lines)


def aggregate_allocation(allocation: List[Dict[str, Any]]) -> str:
    """Aggregate allocation data into a summary."""
    if not allocation:
        return "No allocation data available."

    # Check if target allocation is available
    has_target = any(a.get("target_pct") for a in allocation)

    if has_target:
        lines = ["ASSET ALLOCATION (Current vs Target):"]
        for a in allocation:
            name = a.get("asset_class", "Unknown")
            current = a.get("current_pct", 0) or 0
            target = a.get("target_pct", 0) or 0
            deviation = current - target
            status = "OK" if abs(deviation) < 3 else ("OVER" if deviation > 0 else "UNDER")
            value = a.get("value", 0) or 0
            if value:
                lines.append("  {0}: Current={1:.1f}% | Target={2:.1f}% | Dev={3:+.1f}% [{4}] | Value={5:,.0f}".format(
                    name, current, target, deviation, status, value))
            else:
                lines.append("  {0}: Current={1:.1f}% | Target={2:.1f}% | Dev={3:+.1f}% [{4}]".format(
                    name, current, target, deviation, status))
    else:
        lines = ["ASSET ALLOCATION:"]
        for a in allocation:
            name = a.get("asset_class", "Unknown")
            current = a.get("current_pct", 0) or 0
            value = a.get("value", 0) or 0
            if value:
                lines.append("  {0}: {1:.1f}% (Value: {2:,.0f})".format(name, current, value))
            else:
                lines.append("  {0}: {1:.1f}%".format(name, current))

    return "\n".join(lines)


def aggregate_capital_gains(gains: List[Dict[str, Any]]) -> str:
    """Aggregate capital gain impact data."""
    if not gains:
        return "No capital gain data available."

    total_stcg = sum(g.get("stcg", 0) or 0 for g in gains)
    total_ltcg = sum(g.get("ltcg", 0) or 0 for g in gains)

    lines = []
    lines.append("CAPITAL GAIN SUMMARY:")
    lines.append("  Total STCG: {0:,.0f} | Total LTCG: {1:,.0f}".format(total_stcg, total_ltcg))
    lines.append("")

    # Top gainers
    sorted_gains = sorted(gains, key=lambda x: (x.get("stcg", 0) or 0) + (x.get("ltcg", 0) or 0), reverse=True)
    lines.append("TOP CAPITAL GAIN SECURITIES:")
    for g in sorted_gains[:10]:
        lines.append("  {0}: STCG={1:,.0f} | LTCG={2:,.0f}".format(
            g.get("name", ""), g.get("stcg", 0) or 0, g.get("ltcg", 0) or 0))

    return "\n".join(lines)
