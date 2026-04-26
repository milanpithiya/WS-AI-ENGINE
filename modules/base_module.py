"""
Base Module — Common functionality shared by all AI modules.
All modules inherit from this class.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from llm.ollama_client import OllamaClient
from optimizer.portfolio_optimizer import PortfolioOptimizer
from prompts.prompt_engine import PromptEngine
from rag.retriever import RAGRetriever


class BaseModule:
    """Base class for all AI modules."""

    MODULE_NAME = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimizer = PortfolioOptimizer(config)
        self.prompt_engine = PromptEngine(config)
        self.rag = RAGRetriever(config)
        self._llm_client = None

    async def _get_llm(self) -> OllamaClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = OllamaClient(self.config)
        return self._llm_client

    async def _close_llm(self):
        """Close LLM client."""
        if self._llm_client:
            await self._llm_client.close()
            self._llm_client = None

    # ------------------------------------------------------------------
    # Core Chat Method
    # ------------------------------------------------------------------

    async def chat(
        self,
        question: str,
        model: Optional[str] = None,
        context: Optional[Any] = None,
        history: Optional[List[Dict[str, str]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        General chat — auto-detects template, optionally adds RAG context.
        """
        ctx = context.dict() if context and hasattr(context, "dict") else (context or {})

        # Build prompt with auto-detected template
        messages = self.prompt_engine.build_prompt_with_history(
            question=question,
            history=history,
            context=ctx,
        )

        # Add RAG context if enabled
        rag_context = self.rag.get_context(question)
        if rag_context:
            # Insert RAG context into system prompt
            system_msg = messages[0]["content"]
            messages[0]["content"] = system_msg + "\n\n" + rag_context

        # Call LLM
        llm = await self._get_llm()
        try:
            result = await llm.chat(messages, model=model, options=options)
            result["sources"] = ["rag"] if rag_context else []
            return result
        finally:
            await self._close_llm()

    # ------------------------------------------------------------------
    # Portfolio Analysis
    # ------------------------------------------------------------------

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
        Analyze portfolio with optimization pipeline.
        """
        ctx = context.dict() if context and hasattr(context, "dict") else (context or {})

        # Step 1: Optimize portfolio data
        optimized_text = ""
        sources = []
        if portfolio_data:
            optimized_text, sources = self.optimizer.optimize(
                portfolio_data=portfolio_data,
                question=question,
                tags=tags,
            )

        # Step 2: Add RAG context
        rag_context = self.rag.get_context(question)
        if rag_context:
            optimized_text = optimized_text + "\n\n" + rag_context
            sources.append("rag")

        # Step 3: Build prompt
        messages = self.prompt_engine.build_prompt_with_history(
            question=question,
            history=history,
            portfolio_data=optimized_text,
            context=ctx,
        )

        # Step 4: Call LLM
        llm = await self._get_llm()
        try:
            result = await llm.chat(messages, model=model)
            result["sources"] = sources
            return result
        finally:
            await self._close_llm()

    # ------------------------------------------------------------------
    # Document Summarization
    # ------------------------------------------------------------------

    async def summarize_document(
        self,
        question: Optional[str] = None,
        model: Optional[str] = None,
        document_text: Optional[str] = None,
        document_type: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Summarize a document."""
        ctx = context.dict() if context and hasattr(context, "dict") else (context or {})

        prompt_question = question or "Summarize this document concisely."
        doc_content = document_text or "No document content provided."

        # Truncate if too long (rough token estimate)
        max_chars = self.config.get("optimizer", {}).get("max_tokens_total", 6000) * 4
        if len(doc_content) > max_chars:
            doc_content = doc_content[:max_chars] + "\n\n[Document truncated due to length]"

        messages = self.prompt_engine.build_prompt(
            question=prompt_question,
            template_name="general_chat",
            portfolio_data="DOCUMENT CONTENT ({0}):\n{1}".format(
                document_type or "unknown", doc_content),
            context=ctx,
        )

        llm = await self._get_llm()
        try:
            return await llm.chat(messages, model=model)
        finally:
            await self._close_llm()

    # ------------------------------------------------------------------
    # Market Commentary
    # ------------------------------------------------------------------

    async def generate_market_commentary(
        self,
        model: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None,
        commentary_type: str = "daily",
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate market commentary."""
        length_map = {
            "daily": "200-300 words",
            "weekly": "400-500 words",
            "monthly": "600-800 words",
        }

        extra_vars = {
            "commentary_type": commentary_type,
            "length_guide": length_map.get(commentary_type, "300-400 words"),
            "market_data": str(market_data) if market_data else "Use your latest knowledge.",
        }

        messages = self.prompt_engine.build_prompt(
            question="Generate {0} market commentary.".format(commentary_type),
            template_name="market_commentary",
            extra_vars=extra_vars,
        )

        llm = await self._get_llm()
        try:
            return await llm.chat(messages, model=model)
        finally:
            await self._close_llm()
