"""
Goal Planner Module — Goal-based financial planning AI.
Analyzes client goals, current portfolio, and recommends strategies
to achieve financial objectives.

Phase 2 module — basic structure ready for expansion.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from modules.base_module import BaseModule


class GoalPlannerModule(BaseModule):
    """Goal-Based Financial Planning AI."""

    MODULE_NAME = "goal_planner"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("Goal Planner module initialized")

    async def plan(
        self,
        question: str,
        model: Optional[str] = None,
        context: Optional[Any] = None,
        goals: Optional[List[Dict[str, Any]]] = None,
        current_portfolio: Optional[Dict[str, Any]] = None,
        income_details: Optional[Dict[str, Any]] = None,
        risk_tolerance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate goal-based financial plan.
        """
        ctx = context.dict() if context and hasattr(context, "dict") else (context or {})

        # Prepare goal data as text
        goals_text = "No goals specified."
        if goals:
            lines = []
            for i, goal in enumerate(goals):
                lines.append("Goal {0}: {1}".format(i + 1, goal.get("name", "Unnamed")))
                lines.append("  Target Amount: {0}".format(goal.get("target_amount", "N/A")))
                lines.append("  Timeline: {0}".format(goal.get("timeline_years", "N/A")))
                lines.append("  Priority: {0}".format(goal.get("priority", "N/A")))
                lines.append("  Current Corpus: {0}".format(goal.get("current_corpus", 0)))
            goals_text = "\n".join(lines)

        # Optimize portfolio data if provided
        portfolio_text = "No current portfolio data."
        sources = []
        if current_portfolio:
            portfolio_text, sources = self.optimizer.optimize(
                portfolio_data=current_portfolio,
                question=question,
            )

        income_text = str(income_details) if income_details else "Not provided."

        extra_vars = {
            "goals_data": goals_text,
            "portfolio_data": portfolio_text,
            "income_data": income_text,
            "risk_tolerance": risk_tolerance or ctx.get("risk_tolerance", "Moderate"),
        }

        messages = self.prompt_engine.build_prompt(
            question=question,
            template_name="goal_planning",
            portfolio_data=portfolio_text,
            context=ctx,
            extra_vars=extra_vars,
        )

        llm = await self._get_llm()
        try:
            result = await llm.chat(messages, model=model)
            result["sources"] = sources
            return result
        finally:
            await self._close_llm()
