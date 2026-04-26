"""
Risk Profiler Module — AI-powered behavioral and portfolio risk assessment.
Combines questionnaire analysis with actual portfolio behavior to determine
true risk tolerance.

Phase 2 module — basic structure ready for expansion.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from modules.base_module import BaseModule


class RiskProfilerModule(BaseModule):
    """AI Risk Profiler — behavioral and portfolio risk assessment."""

    MODULE_NAME = "risk_profiler"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("Risk Profiler module initialized")

    async def profile(
        self,
        question: str,
        model: Optional[str] = None,
        context: Optional[Any] = None,
        portfolio_data: Optional[Dict[str, Any]] = None,
        questionnaire_answers: Optional[Dict[str, Any]] = None,
        transaction_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate risk profile assessment.
        """
        ctx = context.dict() if context and hasattr(context, "dict") else (context or {})

        # Optimize portfolio data
        portfolio_text = "No portfolio data."
        sources = []
        if portfolio_data:
            portfolio_text, sources = self.optimizer.optimize(
                portfolio_data=portfolio_data,
                question=question,
            )

        # Format questionnaire
        questionnaire_text = "No questionnaire responses."
        if questionnaire_answers:
            lines = []
            for q, a in questionnaire_answers.items():
                lines.append("Q: {0}\nA: {1}".format(q, a))
            questionnaire_text = "\n\n".join(lines)

        # Format transaction history (last 20)
        txn_text = "No transaction history."
        if transaction_history:
            lines = ["Recent transactions:"]
            for txn in transaction_history[:20]:
                lines.append("  {0}: {1} {2} — {3}".format(
                    txn.get("date", ""),
                    txn.get("type", ""),
                    txn.get("name", ""),
                    txn.get("amount", ""),
                ))
            txn_text = "\n".join(lines)

        extra_vars = {
            "questionnaire_data": questionnaire_text,
            "portfolio_data": portfolio_text,
            "transaction_data": txn_text,
        }

        messages = self.prompt_engine.build_prompt(
            question=question,
            template_name="risk_profiler",
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
