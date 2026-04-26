"""
Portfolio Optimizer — The core data optimization layer.
Reduces raw WealthSpectrum portfolio data to fit within LLM context windows.

Flow:
  1. Receive raw portfolio JSON from Java (holdings, performance, etc.)
  2. Filter fields (keep only what the LLM needs)
  3. Aggregate large datasets into summaries
  4. Select data based on user question (question-aware filtering)
  5. Return optimized text ready for the prompt
"""

from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from optimizer.field_mappings import (
    HOLDINGS_FIELDS,
    HOLDINGS_DEBT_FIELDS,
    PERFORMANCE_FIELDS,
    ALLOCATION_FIELDS,
    TRANSACTION_FIELDS,
    CAPITAL_GAIN_FIELDS,
    IPS_FIELDS,
    get_relevant_tags,
)
from optimizer.aggregator import (
    aggregate_holdings,
    aggregate_performance,
    aggregate_allocation,
    aggregate_capital_gains,
)


class PortfolioOptimizer:
    """Optimizes portfolio data for LLM consumption."""

    def __init__(self, config: Dict[str, Any]):
        opt_cfg = config.get("optimizer", {})
        self.max_holdings_detail = opt_cfg.get("max_holdings_detail", 15)
        self.max_tokens_portfolio = opt_cfg.get("max_tokens_portfolio", 2000)
        self.max_tokens_total = opt_cfg.get("max_tokens_total", 6000)

    # ------------------------------------------------------------------
    # Main Optimization Entry Point
    # ------------------------------------------------------------------

    def optimize(
        self,
        portfolio_data: Dict[str, Any],
        question: str,
        tags: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Optimize portfolio data for LLM prompt.

        Args:
            portfolio_data: Raw data dict from Java. Keys like:
                            holdings, performance, allocation, transactions,
                            capital_gain, cashflow, ips_review, look_through, diversification
            question: User's question (used for question-aware filtering)
            tags: Explicit data tags to include. If None, auto-detected from question.

        Returns:
            Tuple of (optimized_text, sources_used)
        """
        if not portfolio_data:
            return ("No portfolio data provided.", [])

        # Determine which data sections to include
        if tags:
            relevant_tags = tags
        else:
            relevant_tags = get_relevant_tags(question)

        logger.info("Optimizer | question='{0}' | tags={1}", question[:60], relevant_tags)

        sections = []
        sources = []

        # Process each data section
        tag_processors = {
            "Holdings": self._process_holdings,
            "Performance": self._process_performance,
            "Return Summary": self._process_performance,
            "Allocation": self._process_allocation,
            "Diversification": self._process_allocation,
            "Allocation History": self._process_allocation,
            "Transaction": self._process_transactions,
            "Capital Gain Impact": self._process_capital_gains,
            "Cash Flow Projection": self._process_cashflow,
            "IPS Review": self._process_ips,
            "Look Through": self._process_look_through,
            "Recommend Product": self._process_recommendations,
        }

        for tag in relevant_tags:
            processor = tag_processors.get(tag)
            if processor:
                # Map tag to data key
                data_key = self._tag_to_data_key(tag)
                raw_data = portfolio_data.get(data_key)
                if raw_data:
                    section_text = processor(raw_data, question)
                    if section_text:
                        sections.append(section_text)
                        sources.append(data_key)
                else:
                    logger.debug("No data found for tag '{0}' (key: {1})", tag, data_key)

        if not sections:
            return ("No relevant portfolio data found for this question.", [])

        optimized = "\n\n".join(sections)

        # Rough token estimate (1 token ~ 4 chars)
        estimated_tokens = len(optimized) // 4
        logger.info("Optimizer | output_tokens~{0} | sections={1} | sources={2}",
                     estimated_tokens, len(sections), sources)

        return (optimized, sources)

    # ------------------------------------------------------------------
    # Data Key Mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _tag_to_data_key(tag: str) -> str:
        """Map WealthSpectrum tag name to portfolio_data dict key."""
        mapping = {
            "Holdings": "holdings",
            "Performance": "performance",
            "Return Summary": "performance",
            "Allocation": "allocation",
            "Diversification": "allocation",
            "Allocation History": "allocation",
            "Transaction": "transactions",
            "Capital Gain Impact": "capital_gain",
            "Cash Flow Projection": "cashflow",
            "IPS Review": "ips_review",
            "Look Through": "look_through",
            "Recommend Product": "recommendations",
        }
        return mapping.get(tag, tag.lower().replace(" ", "_"))

    # ------------------------------------------------------------------
    # Field Filtering
    # ------------------------------------------------------------------

    def _filter_fields(
        self,
        data: List[Dict[str, Any]],
        field_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Filter and rename fields based on field mapping config."""
        keep = field_config.get("keep", [])
        rename = field_config.get("rename", {})
        decimals = field_config.get("round_decimals", 2)

        result = []
        for item in data:
            filtered = {}
            for field in keep:
                value = item.get(field)
                if value is not None:
                    # Round numeric values
                    if isinstance(value, float):
                        value = round(value, decimals)
                    # Apply rename
                    out_field = rename.get(field, field)
                    filtered[out_field] = value
            if filtered:
                result.append(filtered)

        return result

    # ------------------------------------------------------------------
    # Section Processors
    # ------------------------------------------------------------------

    def _process_holdings(self, data: Any, question: str) -> str:
        """Process holdings data — filter fields, then aggregate."""
        if isinstance(data, list):
            holdings = data
        elif isinstance(data, dict):
            holdings = data.get("dataList",
                       data.get("holdings",
                       data.get("data", [])))
            if not isinstance(holdings, list):
                holdings = [data]
        else:
            return ""

        if not holdings:
            return ""

        # Determine if question is debt-focused
        question_lower = question.lower()
        is_debt = any(kw in question_lower for kw in
                      ["debt", "fixed income", "bond", "yield", "ytm", "credit", "rating", "duration"])

        field_config = HOLDINGS_DEBT_FIELDS if is_debt else HOLDINGS_FIELDS
        filtered = self._filter_fields(holdings, field_config)

        # If holdings count is small enough, return filtered JSON-like text
        if len(filtered) <= self.max_holdings_detail:
            return self._to_text_table("HOLDINGS DATA", filtered)

        # Otherwise, aggregate into summary
        return aggregate_holdings(filtered, top_n=self.max_holdings_detail)

    def _process_performance(self, data: Any, question: str) -> str:
        """Process performance data."""
        if isinstance(data, list):
            perf_list = data
        elif isinstance(data, dict):
            perf_list = data.get("dataList", data.get("data", []))
            if not isinstance(perf_list, list):
                perf_list = [data]
        else:
            return ""

        filtered = self._filter_fields(perf_list, PERFORMANCE_FIELDS)
        return aggregate_performance(filtered)

    def _process_allocation(self, data: Any, question: str) -> str:
        """Process allocation/diversification data."""
        if isinstance(data, list):
            alloc_list = data
        elif isinstance(data, dict):
            alloc_list = data.get("astclsDataList",
                         data.get("dataList",
                         data.get("data", [])))
            if not isinstance(alloc_list, list):
                alloc_list = [data]
        else:
            return ""

        filtered = self._filter_fields(alloc_list, ALLOCATION_FIELDS)
        return aggregate_allocation(filtered)

    def _process_transactions(self, data: Any, question: str) -> str:
        """Process transaction data — summarize recent transactions."""
        if isinstance(data, list):
            txns = data
        elif isinstance(data, dict):
            txns = data.get("data", [])
        else:
            return ""

        filtered = self._filter_fields(txns, TRANSACTION_FIELDS)

        # Limit to last 20 transactions for token efficiency
        if len(filtered) > 20:
            filtered = filtered[:20]

        return self._to_text_table("RECENT TRANSACTIONS (last 20)", filtered)

    def _process_capital_gains(self, data: Any, question: str) -> str:
        """Process capital gain impact data."""
        if isinstance(data, list):
            gains = data
        elif isinstance(data, dict):
            gains = data.get("securityList", data.get("data", []))
            if not isinstance(gains, list):
                gains = [data]
        else:
            return ""

        filtered = self._filter_fields(gains, CAPITAL_GAIN_FIELDS)
        return aggregate_capital_gains(filtered)

    def _process_cashflow(self, data: Any, question: str) -> str:
        """Process cash flow projection data."""
        if isinstance(data, dict):
            return "CASH FLOW PROJECTION:\n{0}".format(
                self._dict_to_text(data, max_depth=2))
        elif isinstance(data, list):
            return self._to_text_table("CASH FLOW PROJECTION", data[:15])
        return ""

    def _process_ips(self, data: Any, question: str) -> str:
        """Process IPS review data."""
        if isinstance(data, dict):
            # IPS data structure varies — extract key compliance info
            lines = ["IPS COMPLIANCE REVIEW:"]
            for key, value in data.items():
                if isinstance(value, list):
                    filtered = self._filter_fields(value, IPS_FIELDS)
                    if filtered:
                        lines.append(self._to_text_table("  " + key, filtered))
                elif isinstance(value, (str, int, float, bool)):
                    lines.append("  {0}: {1}".format(key, value))
            return "\n".join(lines)
        return ""

    def _process_look_through(self, data: Any, question: str) -> str:
        """Process look-through analysis data."""
        if isinstance(data, dict):
            lines = ["LOOK-THROUGH ANALYSIS:"]
            total = data.get("totalMktVal", 0)
            if total:
                lines.append("  Total Market Value: {0:,.0f}".format(total))
            equity_alloc = data.get("equityAllocMF", 0)
            debt_alloc = data.get("debtAllocMF", 0)
            if equity_alloc or debt_alloc:
                lines.append("  Equity Allocation (MF): {0:.1f}%".format(equity_alloc or 0))
                lines.append("  Debt Allocation (MF): {0:.1f}%".format(debt_alloc or 0))

            # Underlying holdings
            sec_list = data.get("secHoldingList", [])
            if sec_list and len(sec_list) > 15:
                sec_list = sec_list[:15]
            if sec_list:
                lines.append("  TOP UNDERLYING HOLDINGS:")
                for s in sec_list:
                    name = s.get("securityName", s.get("name", ""))
                    weight = s.get("weight", s.get("allocation", 0))
                    lines.append("    {0}: {1:.1f}%".format(name, weight or 0))

            return "\n".join(lines)
        return ""

    def _process_recommendations(self, data: Any, question: str) -> str:
        """Process recommended products data."""
        if isinstance(data, list):
            return self._to_text_table("RECOMMENDED PRODUCTS", data[:10])
        return ""

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _to_text_table(title: str, data: List[Dict[str, Any]]) -> str:
        """Convert a list of dicts to a readable text table."""
        if not data:
            return ""

        lines = ["{0}:".format(title)]
        for item in data:
            parts = []
            for k, v in item.items():
                if isinstance(v, float):
                    parts.append("{0}={1:.2f}".format(k, v))
                else:
                    parts.append("{0}={1}".format(k, v))
            lines.append("  " + " | ".join(parts))

        return "\n".join(lines)

    @staticmethod
    def _dict_to_text(data: Dict[str, Any], max_depth: int = 2, depth: int = 0) -> str:
        """Convert a nested dict to readable text."""
        indent = "  " * (depth + 1)
        lines = []
        for key, value in data.items():
            if isinstance(value, dict) and depth < max_depth:
                lines.append("{0}{1}:".format(indent, key))
                lines.append(PortfolioOptimizer._dict_to_text(value, max_depth, depth + 1))
            elif isinstance(value, list):
                lines.append("{0}{1}: [{2} items]".format(indent, key, len(value)))
            else:
                lines.append("{0}{1}: {2}".format(indent, key, value))
        return "\n".join(lines)
