"""
AI Advisor Module — General-purpose wealth management AI assistant.
Handles portfolio Q&A, analysis, document summarization, market commentary.
This is the primary module used by WealthSpectrum AI Advisor feature.
"""

from typing import Dict, Any
from loguru import logger

from modules.base_module import BaseModule


class AIAdvisorModule(BaseModule):
    """WealthSpectrum AI Advisor — the main AI module."""

    MODULE_NAME = "ai_advisor"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("AI Advisor module initialized")
