"""
API Routes — All FastAPI endpoints for ws-ai-engine.
"""

import time
from typing import List

from fastapi import APIRouter, Request, HTTPException
from loguru import logger

from api.models import (
    AIResponse,
    ChatRequest,
    PortfolioAnalyzeRequest,
    DocumentSummarizeRequest,
    MarketCommentaryRequest,
    GoalPlanningRequest,
    RiskProfilerRequest,
    DocumentIndexRequest,
    HealthResponse,
    ModelInfo,
)
from modules.ai_advisor import AIAdvisorModule
from modules.portfolio_insight import PortfolioInsightModule
from modules.goal_planner import GoalPlannerModule
from modules.risk_profiler import RiskProfilerModule
from llm.ollama_client import OllamaClient

router = APIRouter()


# ===================================================================
# Helper: Get module by name
# ===================================================================

def _get_module(module_name: str, config: dict):
    """Factory to get the right AI module."""
    modules = {
        "ai_advisor": AIAdvisorModule,
        "portfolio_insight": PortfolioInsightModule,
        "goal_planner": GoalPlannerModule,
        "risk_profiler": RiskProfilerModule,
    }
    module_class = modules.get(module_name)
    if not module_class:
        raise HTTPException(status_code=400, detail="Unknown module: {0}".format(module_name))

    # Check if module is enabled
    module_cfg = config.get("modules", {}).get(module_name, {})
    if not module_cfg.get("enabled", False):
        raise HTTPException(status_code=400, detail="Module '{0}' is not enabled".format(module_name))

    return module_class(config)


# ===================================================================
# Health Check
# ===================================================================

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """Check service health, Ollama connectivity, and available models."""
    config = request.app.state.config
    ollama_cfg = config.get("ollama", {})

    client = OllamaClient(config)
    try:
        is_alive = await client.health_check()
        models = await client.list_models() if is_alive else []
        model_names = [m.get("name", "") for m in models]

        # Active modules
        active = [
            name for name, cfg in config.get("modules", {}).items()
            if cfg.get("enabled", False)
        ]

        return HealthResponse(
            status="healthy" if is_alive else "degraded",
            ollama_connected=is_alive,
            ollama_url=ollama_cfg.get("base_url", ""),
            available_models=model_names,
            rag_enabled=config.get("rag", {}).get("enabled", False),
            active_modules=active,
        )
    finally:
        await client.close()


# ===================================================================
# List Available Models
# ===================================================================

@router.get("/models", response_model=List[ModelInfo], tags=["System"])
async def list_models(request: Request):
    """List all models available in Ollama."""
    config = request.app.state.config
    client = OllamaClient(config)
    try:
        models = await client.list_models()
        return [
            ModelInfo(
                name=m.get("name", ""),
                size=m.get("size", ""),
                modified_at=m.get("modified_at", ""),
                family=m.get("details", {}).get("family", ""),
            )
            for m in models
        ]
    finally:
        await client.close()


# ===================================================================
# General Chat
# ===================================================================

@router.post("/chat", response_model=AIResponse, tags=["Chat"])
async def chat(request: Request, body: ChatRequest):
    """
    General-purpose chat endpoint.
    Routes to the appropriate module based on body.module.
    """
    config = request.app.state.config
    start = time.time()

    logger.info("Chat request | module={0} | question={1}",
                body.module, body.question[:80])

    try:
        module = _get_module(body.module, config)
        result = await module.chat(
            question=body.question,
            model=body.model,
            context=body.context,
            history=body.history,
            options=body.options,
        )

        elapsed = int((time.time() - start) * 1000)
        return AIResponse(
            status=True,
            msg="Success",
            data={"answer": result.get("answer", "")},
            model_used=result.get("model_used"),
            tokens_used=result.get("tokens_used"),
            processing_time_ms=elapsed,
            sources=result.get("sources"),
        )
    except Exception as e:
        logger.error("Chat error: {0}", str(e))
        elapsed = int((time.time() - start) * 1000)
        return AIResponse(
            status=False,
            msg=str(e),
            processing_time_ms=elapsed,
        )


# ===================================================================
# Portfolio Analysis
# ===================================================================

@router.post("/portfolio/analyze", response_model=AIResponse, tags=["Portfolio"])
async def portfolio_analyze(request: Request, body: PortfolioAnalyzeRequest):
    """
    Analyze portfolio data with AI.
    Java sends raw portfolio data (holdings, performance, allocation, etc.).
    Python optimizes data, builds prompt, calls Ollama.
    """
    config = request.app.state.config
    start = time.time()

    logger.info("Portfolio analyze | module={0} | tags={1} | question={2}",
                body.module, body.tags, body.question[:80])

    try:
        module = _get_module(body.module, config)
        result = await module.analyze_portfolio(
            question=body.question,
            model=body.model,
            context=body.context,
            portfolio_data=body.portfolio_data,
            tags=body.tags,
            history=body.history,
        )

        elapsed = int((time.time() - start) * 1000)
        return AIResponse(
            status=True,
            msg="Success",
            data={"answer": result.get("answer", "")},
            model_used=result.get("model_used"),
            tokens_used=result.get("tokens_used"),
            processing_time_ms=elapsed,
            sources=result.get("sources"),
        )
    except Exception as e:
        logger.error("Portfolio analyze error: {0}", str(e))
        elapsed = int((time.time() - start) * 1000)
        return AIResponse(
            status=False,
            msg=str(e),
            processing_time_ms=elapsed,
        )


# ===================================================================
# Document Summarize
# ===================================================================

@router.post("/document/summarize", response_model=AIResponse, tags=["Document"])
async def document_summarize(request: Request, body: DocumentSummarizeRequest):
    """Summarize a document using AI."""
    config = request.app.state.config
    start = time.time()

    logger.info("Document summarize | type={0}", body.document_type)

    try:
        module = _get_module("ai_advisor", config)
        result = await module.summarize_document(
            question=body.question,
            model=body.model,
            document_text=body.document_text,
            document_type=body.document_type,
            context=body.context,
        )

        elapsed = int((time.time() - start) * 1000)
        return AIResponse(
            status=True,
            msg="Success",
            data={"answer": result.get("answer", "")},
            model_used=result.get("model_used"),
            tokens_used=result.get("tokens_used"),
            processing_time_ms=elapsed,
        )
    except Exception as e:
        logger.error("Document summarize error: {0}", str(e))
        return AIResponse(status=False, msg=str(e))


# ===================================================================
# Market Commentary
# ===================================================================

@router.post("/market/commentary", response_model=AIResponse, tags=["Market"])
async def market_commentary(request: Request, body: MarketCommentaryRequest):
    """Generate AI-powered market commentary."""
    config = request.app.state.config
    start = time.time()

    logger.info("Market commentary | type={0}", body.commentary_type)

    try:
        module = _get_module("ai_advisor", config)
        result = await module.generate_market_commentary(
            model=body.model,
            market_data=body.market_data,
            commentary_type=body.commentary_type,
            context=body.context,
        )

        elapsed = int((time.time() - start) * 1000)
        return AIResponse(
            status=True,
            msg="Success",
            data={"answer": result.get("answer", "")},
            model_used=result.get("model_used"),
            processing_time_ms=elapsed,
        )
    except Exception as e:
        logger.error("Market commentary error: {0}", str(e))
        return AIResponse(status=False, msg=str(e))


# ===================================================================
# Goal Planning (Phase 2)
# ===================================================================

@router.post("/goal/plan", response_model=AIResponse, tags=["Goal Planning"])
async def goal_plan(request: Request, body: GoalPlanningRequest):
    """AI-powered goal-based financial planning."""
    config = request.app.state.config
    start = time.time()

    try:
        module = _get_module("goal_planner", config)
        result = await module.plan(
            question=body.question,
            model=body.model,
            context=body.context,
            goals=body.goals,
            current_portfolio=body.current_portfolio,
            income_details=body.income_details,
            risk_tolerance=body.risk_tolerance,
        )

        elapsed = int((time.time() - start) * 1000)
        return AIResponse(
            status=True,
            msg="Success",
            data={"answer": result.get("answer", "")},
            model_used=result.get("model_used"),
            processing_time_ms=elapsed,
        )
    except Exception as e:
        logger.error("Goal planning error: {0}", str(e))
        return AIResponse(status=False, msg=str(e))


# ===================================================================
# Risk Profiler (Phase 2)
# ===================================================================

@router.post("/risk/profile", response_model=AIResponse, tags=["Risk Profiler"])
async def risk_profile(request: Request, body: RiskProfilerRequest):
    """AI-powered risk profiling."""
    config = request.app.state.config
    start = time.time()

    try:
        module = _get_module("risk_profiler", config)
        result = await module.profile(
            question=body.question,
            model=body.model,
            context=body.context,
            portfolio_data=body.portfolio_data,
            questionnaire_answers=body.questionnaire_answers,
            transaction_history=body.transaction_history,
        )

        elapsed = int((time.time() - start) * 1000)
        return AIResponse(
            status=True,
            msg="Success",
            data={"answer": result.get("answer", "")},
            model_used=result.get("model_used"),
            processing_time_ms=elapsed,
        )
    except Exception as e:
        logger.error("Risk profiler error: {0}", str(e))
        return AIResponse(status=False, msg=str(e))


# ===================================================================
# RAG: Index Document
# ===================================================================

@router.post("/rag/index", response_model=AIResponse, tags=["RAG"])
async def rag_index_document(request: Request, body: DocumentIndexRequest):
    """Index a document for RAG retrieval."""
    config = request.app.state.config

    if not config.get("rag", {}).get("enabled", False):
        return AIResponse(status=False, msg="RAG is not enabled. Set rag.enabled=true in config.yaml")

    try:
        from rag.indexer import RAGIndexer
        indexer = RAGIndexer(config)
        indexer.index_document(
            document_id=body.document_id,
            document_type=body.document_type,
            title=body.title,
            content=body.content,
            metadata=body.metadata,
        )
        return AIResponse(status=True, msg="Document indexed successfully")
    except Exception as e:
        logger.error("RAG index error: {0}", str(e))
        return AIResponse(status=False, msg=str(e))
