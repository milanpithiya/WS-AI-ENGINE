"""
ws-ai-engine — WealthSpectrum AI Engine
Central LLM service for all AI modules.

Usage:
    python main.py
    or
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import logging
import logging.handlers
import yaml
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes import router as api_router

# ---------------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = os.environ.get("WS_AI_CONFIG", "config.yaml")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config(CONFIG_PATH)

# ---------------------------------------------------------------------------
# Configure Logging — Rotating numbered log files
# Generates: wsAiEngine.log (current), wsAiEngine.log.1, wsAiEngine.log.2 ...
# Similar to WealthSpectrum Tomcat logging pattern
# ---------------------------------------------------------------------------
log_cfg = config.get("logging", {})

# Read config values (with defaults)
log_file_path = log_cfg.get("log_file_path", "./logs")
log_filename = log_cfg.get("log_filename", "wsAiEngine.log")
log_max_filesize = log_cfg.get("log_max_filesize", 50)        # MB
log_max_files = log_cfg.get("log_max_files", 5)
log_level = log_cfg.get("level", "INFO").upper()
console_enabled = log_cfg.get("console_enabled", True)

# Ensure log directory exists
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path, exist_ok=True)

# Full path: E:\Tomcat9251\logs\wsAiEngine.log
log_full_path = os.path.join(log_file_path, log_filename)

# Log format matching WealthSpectrum style
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Setup Logging — Loguru handles file rotation + console
# Generates: wsAiEngine.log (current) with rotation by size
# ---------------------------------------------------------------------------

# Remove default loguru stderr handler
logger.remove()

# Add loguru rotating file handler
logger.add(
    log_full_path,
    rotation="{0} MB".format(log_max_filesize),
    retention=log_max_files,
    level=log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
    encoding="utf-8",
    enqueue=True,       # Thread-safe
    backtrace=True,     # Full traceback on errors
    diagnose=False,     # No variable inspection in production
)

# Also log to console if enabled
if console_enabled:
    logger.add(
        sys.stdout,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
        colorize=True,
    )

# ---------------------------------------------------------------------------
# Intercept standard library logging → loguru
# This captures uvicorn, httpx, httpcore, etc. into our loguru handlers.
# ---------------------------------------------------------------------------
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# Replace root logger handlers with the intercept handler
intercept_handler = InterceptHandler()
root_logger = logging.getLogger()
root_logger.handlers = [intercept_handler]
root_logger.setLevel(getattr(logging, log_level, logging.INFO))

# Explicitly intercept uvicorn loggers (suppress their default handlers)
for _logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    _uvicorn_logger = logging.getLogger(_logger_name)
    _uvicorn_logger.handlers = [intercept_handler]
    _uvicorn_logger.propagate = False

# Suppress noisy third-party HTTP debug logs
for _logger_name in ("httpx", "httpcore", "httpcore.http11", "httpcore.connection"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

# Log the configuration on startup
logger.info("Logging configured:")
logger.info("  Log file: {0}", log_full_path)
logger.info("  Max file size: {0} MB", log_max_filesize)
logger.info("  Max backup files: {0}", log_max_files)
logger.info("  Log level: {0}", log_level)
logger.info("  Console output: {0}", console_enabled)

# ---------------------------------------------------------------------------
# Lifespan — Startup & Shutdown Events
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    ollama_url = config.get("ollama", {}).get("base_url", "http://localhost:11434")
    logger.info("=" * 60)
    logger.info("ws-ai-engine starting...")
    logger.info("Ollama URL: {0}", ollama_url)
    logger.info("Default Model: {0}", config.get("ollama", {}).get("default_model", "mistral"))
    logger.info("RAG Enabled: {0}", config.get("rag", {}).get("enabled", False))
    logger.info("API Docs: http://localhost:{0}/docs", config.get("server", {}).get("port", 8000))
    logger.info("=" * 60)

    # Verify Ollama connectivity
    from llm.ollama_client import OllamaClient
    client = OllamaClient(config)
    is_alive = await client.health_check()
    if is_alive:
        logger.info("Ollama connection: OK")
        models = await client.list_models()
        logger.info("Available models: {0}", [m.get("name", "") for m in models])
    else:
        logger.warning("Ollama connection: FAILED — check if Ollama is running at {0}", ollama_url)
    await client.close()

    yield

    # --- Shutdown ---
    logger.info("ws-ai-engine shutting down...")


# ---------------------------------------------------------------------------
# Create FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ws-ai-engine",
    description="WealthSpectrum AI Engine — Central LLM service for AI Advisor, "
                "Portfolio Insight, Goal Planner, Risk Profiler and more.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Middleware (allow WealthSpectrum Java server to call this)
# ---------------------------------------------------------------------------
cors_origins = config.get("server", {}).get("cors_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Store config in app state (accessible in routes via request.app.state)
# ---------------------------------------------------------------------------
app.state.config = config

# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: {0}", str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "status": False,
            "msg": "Internal server error. Please check logs.",
            "data": None,
        },
    )


# ---------------------------------------------------------------------------
# Include API Routes
# ---------------------------------------------------------------------------
app.include_router(api_router, prefix="/api/v1")

# ---------------------------------------------------------------------------
# Run with: python main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    server_cfg = config.get("server", {})
    uvicorn.run(
        "main:app",
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 8000),
        workers=server_cfg.get("workers", 4),
        reload=False,
        log_level=log_cfg.get("level", "info").lower(),
    )
