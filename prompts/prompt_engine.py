"""
Prompt Engine — Loads templates and builds complete prompts for the LLM.
Selects the right template based on question context and fills in variables.
"""

import os
from typing import Dict, Any, Optional, List
from loguru import logger


class PromptEngine:
    """Loads prompt templates and builds complete prompts."""

    TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

    # Keyword-to-template mapping for auto-detection
    TEMPLATE_KEYWORDS = {
        "rebalancing": ["rebalanc", "overweight", "underweight", "adjust allocation"],
        "tax_planning": ["tax", "capital gain", "stcg", "ltcg", "tax harvest", "tax loss"],
        "risk_assessment": ["risk assess", "risk analys", "concentrat", "risk score"],
        "market_commentary": ["market comment", "market summar", "market outlook"],
        "goal_planning": ["goal", "retirement", "child education", "house purchase", "financial plan"],
        "risk_profiler": ["risk profil", "risk toleran", "questionnaire", "behavioral"],
    }

    def __init__(self, config: Dict[str, Any] = None):
        self._cache = {}  # type: Dict[str, str]
        self._load_all_templates()

    # ------------------------------------------------------------------
    # Template Loading
    # ------------------------------------------------------------------

    def _load_all_templates(self):
        """Pre-load all templates into cache."""
        if not os.path.isdir(self.TEMPLATES_DIR):
            logger.warning("Templates directory not found: {0}", self.TEMPLATES_DIR)
            return

        for filename in os.listdir(self.TEMPLATES_DIR):
            if filename.endswith(".txt"):
                name = filename[:-4]  # Remove .txt
                filepath = os.path.join(self.TEMPLATES_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    self._cache[name] = f.read()
                logger.debug("Loaded template: {0}", name)

        logger.info("Loaded {0} prompt templates", len(self._cache))

    def get_template(self, name: str) -> str:
        """Get a template by name."""
        return self._cache.get(name, "")

    # ------------------------------------------------------------------
    # Auto-Detect Template
    # ------------------------------------------------------------------

    def detect_template(self, question: str) -> str:
        """Auto-detect the best template based on question keywords."""
        question_lower = question.lower()

        for template_name, keywords in self.TEMPLATE_KEYWORDS.items():
            for kw in keywords:
                if kw in question_lower:
                    logger.debug("Auto-detected template: {0} (keyword: {1})",
                                 template_name, kw)
                    return template_name

        # Default to portfolio_analysis for portfolio-related questions
        portfolio_keywords = ["portfolio", "holding", "perform", "return", "allocat"]
        for kw in portfolio_keywords:
            if kw in question_lower:
                return "portfolio_analysis"

        return "general_chat"

    # ------------------------------------------------------------------
    # Build Prompt
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        question: str,
        template_name: Optional[str] = None,
        portfolio_data: str = "",
        context: Optional[Dict[str, Any]] = None,
        extra_vars: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build complete messages list for the LLM.

        Args:
            question: User's question
            template_name: Explicit template name, or auto-detected if None
            portfolio_data: Optimized portfolio text from PortfolioOptimizer
            context: ClientContext dict (client_name, as_on_date, risk_profile, etc.)
            extra_vars: Additional template variables

        Returns:
            List of message dicts: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        # Auto-detect template if not specified
        if not template_name:
            template_name = self.detect_template(question)

        template = self.get_template(template_name)
        if not template:
            logger.warning("Template '{0}' not found, using general_chat", template_name)
            template = self.get_template("general_chat")

        # Prepare template variables
        ctx = context or {}
        variables = {
            "question": question,
            "client_name": ctx.get("client_name", "Not specified"),
            "as_on_date": ctx.get("as_on_date", "Not specified"),
            "risk_profile": ctx.get("risk_profile", "Not specified"),
            "risk_tolerance": ctx.get("risk_tolerance", "Not specified"),
            "advisor_name": ctx.get("advisor_name", ""),
            "portfolio_data": portfolio_data or "No portfolio data provided.",
            "context_section": "",
            "goals_data": "Not provided.",
            "income_data": "Not provided.",
            "questionnaire_data": "Not provided.",
            "transaction_data": "Not provided.",
            "market_data": "Not provided.",
            "commentary_type": "daily",
            "length_guide": "300-400 words",
        }

        # Add extra variables
        if extra_vars:
            variables.update(extra_vars)

        # Add context section for general chat
        if portfolio_data and template_name == "general_chat":
            variables["context_section"] = (
                "CONTEXT:\nClient: {0}\nDate: {1}\n\nPORTFOLIO DATA:\n{2}".format(
                    variables["client_name"],
                    variables["as_on_date"],
                    portfolio_data,
                )
            )

        # Fill template
        try:
            system_prompt = template.format(**variables)
        except KeyError as e:
            logger.warning("Template variable not found: {0}, using raw template", str(e))
            # Fallback: just replace what we can
            system_prompt = template
            for key, value in variables.items():
                system_prompt = system_prompt.replace("{" + key + "}", str(value))

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # If portfolio data is in the system prompt, user message is just the question
        # If not, include portfolio data in user message
        if portfolio_data and "{portfolio_data}" not in template:
            user_content = "Portfolio Data:\n{0}\n\nQuestion: {1}".format(
                portfolio_data, question)
        else:
            user_content = question

        messages.append({"role": "user", "content": user_content})

        logger.debug("Built prompt | template={0} | system_len={1} | user_len={2}",
                      template_name, len(system_prompt), len(user_content))

        return messages

    # ------------------------------------------------------------------
    # Build with Conversation History
    # ------------------------------------------------------------------

    def build_prompt_with_history(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        template_name: Optional[str] = None,
        portfolio_data: str = "",
        context: Optional[Dict[str, Any]] = None,
        extra_vars: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build messages with conversation history (for multi-turn chat).
        History items: [{"question": "...", "answer": "..."}, ...]
        """
        # Start with system + initial user message
        messages = self.build_prompt(
            question=question,
            template_name=template_name,
            portfolio_data=portfolio_data,
            context=context,
            extra_vars=extra_vars,
        )

        # If history exists, insert past Q&A before the current question
        if history:
            system_msg = messages[0]
            current_user_msg = messages[-1]

            new_messages = [system_msg]

            # Add history (last 5 turns max to save tokens)
            for turn in history[-5:]:
                q = turn.get("question", "")
                a = turn.get("answer", "")
                if q:
                    new_messages.append({"role": "user", "content": q})
                if a:
                    new_messages.append({"role": "assistant", "content": a})

            new_messages.append(current_user_msg)
            return new_messages

        return messages
