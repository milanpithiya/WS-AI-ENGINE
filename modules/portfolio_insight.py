"""
Portfolio Insight Module — Deep portfolio analysis and pattern detection.
Focuses on: concentration risk, sector trends, rebalancing signals,
performance attribution, and actionable insights.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from modules.base_module import BaseModule


class PortfolioInsightModule(BaseModule):
    """Client Portfolio Insight Engine — deep analysis and patterns."""

    MODULE_NAME = "portfolio_insight"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("Portfolio Insight module initialized")

    async def analyze_portfolio(
        self,
        question: str,
        model: Optional[str] = None,
        context: Optional[Any] = None,
        portfolio_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced portfolio analysis with insight-specific prompting.
        Adds extra analysis instructions for deeper insights.
        """
        # Force comprehensive tags for insight analysis
        if not tags:
            tags = [
                "Holdings", "Performance", "Diversification",
                "IPS Review", "Capital Gain Impact",
            ]

        # Add insight-specific instructions
        extra_question = (
            "{0}\n\nAdditionally, identify:\n"
            "1. Top 3 concentration risks\n"
            "2. Performance outliers (best and worst)\n"
            "3. Any IPS compliance deviations\n"
            "4. Actionable rebalancing opportunities\n"
            "5. Tax-efficient restructuring suggestions"
        ).format(question)

        return await super().analyze_portfolio(
            question=extra_question,
            model=model,
            context=context,
            portfolio_data=portfolio_data,
            tags=tags,
            history=history,
        )
