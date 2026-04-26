"""
API Request/Response Models — Pydantic schemas.
Compatible with Python 3.8+ (uses typing imports).
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


# ===================================================================
# Common Models
# ===================================================================

class ClientContext(BaseModel):
    """Client context passed from WealthSpectrum Java."""
    client_name: Optional[str] = None
    group_id: Optional[int] = None
    scope: Optional[str] = None          # "C" = Client, "G" = Group
    scope_id: Optional[int] = None
    as_on_date: Optional[str] = None     # "2026-04-21"
    risk_profile: Optional[str] = None
    advisor_name: Optional[str] = None


class AIResponse(BaseModel):
    """Standard response format — mirrors WealthSpectrum SecureApiResult."""
    model_config = ConfigDict(protected_namespaces=())

    status: bool = True
    msg: str = "Success"
    data: Optional[Any] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[int] = None
    sources: Optional[List[str]] = None


# ===================================================================
# Chat Request (General purpose)
# ===================================================================

class ChatRequest(BaseModel):
    """General chat request — any module can use this."""
    session_id: Optional[str] = None
    module: str = "ai_advisor"           # ai_advisor | portfolio_insight | goal_planner | risk_profiler
    question: str
    model: Optional[str] = None          # Override default model
    context: Optional[ClientContext] = None
    history: Optional[List[Dict[str, str]]] = None   # Previous Q&A for conversation
    options: Optional[Dict[str, Any]] = None          # Model options override


# ===================================================================
# Portfolio Analysis Request
# ===================================================================

class PortfolioAnalyzeRequest(BaseModel):
    """Portfolio analysis — receives raw data from Java service layer."""
    session_id: Optional[str] = None
    module: str = "ai_advisor"
    question: str
    model: Optional[str] = None
    context: Optional[ClientContext] = None
    history: Optional[List[Dict[str, str]]] = None
    portfolio_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw portfolio data from WealthSpectrum. Keys can include: "
                    "holdings, performance, allocation, transactions, "
                    "capital_gain, cashflow, ips_review, look_through, diversification"
    )
    tags: Optional[List[str]] = None     # ["Holdings", "Performance", "Allocation"]


# ===================================================================
# Document Summarize Request
# ===================================================================

class DocumentSummarizeRequest(BaseModel):
    """Document summarization — receives text/base64 content."""
    session_id: Optional[str] = None
    question: Optional[str] = "Summarize this document."
    model: Optional[str] = None
    document_text: Optional[str] = None
    document_type: Optional[str] = None  # "pdf", "xlsx", "txt"
    context: Optional[ClientContext] = None


# ===================================================================
# Market Commentary Request
# ===================================================================

class MarketCommentaryRequest(BaseModel):
    """Market commentary generation."""
    session_id: Optional[str] = None
    model: Optional[str] = None
    market_data: Optional[Dict[str, Any]] = None
    commentary_type: str = "daily"       # daily | weekly | monthly
    context: Optional[ClientContext] = None


# ===================================================================
# Goal Planning Request (Phase 2)
# ===================================================================

class GoalPlanningRequest(BaseModel):
    """Goal-based financial planning AI."""
    session_id: Optional[str] = None
    question: str
    model: Optional[str] = None
    context: Optional[ClientContext] = None
    goals: Optional[List[Dict[str, Any]]] = None      # Client financial goals
    current_portfolio: Optional[Dict[str, Any]] = None
    income_details: Optional[Dict[str, Any]] = None
    risk_tolerance: Optional[str] = None               # Conservative | Moderate | Aggressive


# ===================================================================
# Risk Profiler Request (Phase 2)
# ===================================================================

class RiskProfilerRequest(BaseModel):
    """AI Risk Profiler — behavioral and portfolio risk."""
    session_id: Optional[str] = None
    question: str
    model: Optional[str] = None
    context: Optional[ClientContext] = None
    portfolio_data: Optional[Dict[str, Any]] = None
    questionnaire_answers: Optional[Dict[str, Any]] = None
    transaction_history: Optional[List[Dict[str, Any]]] = None


# ===================================================================
# RAG Document Index Request
# ===================================================================

class DocumentIndexRequest(BaseModel):
    """Index a document for RAG retrieval."""
    document_id: str
    document_type: str                    # "market_commentary", "ips", "factsheet", "policy"
    title: str
    content: str                          # Plain text content
    metadata: Optional[Dict[str, str]] = None


# ===================================================================
# Health & Model Info
# ===================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    ollama_connected: bool = False
    ollama_url: str = ""
    available_models: List[str] = []
    rag_enabled: bool = False
    active_modules: List[str] = []
    version: str = "1.0.0"


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None
    family: Optional[str] = None
