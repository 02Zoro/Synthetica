"""
FastAPI main application entry point for SABDE.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings
from app.api.routes import research, health
from app.services.vector_service import VectorService
from app.services.knowledge_graph_service import KnowledgeGraphService

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting SABDE application")
    
    # Initialize services
    app.state.vector_service = VectorService()
    app.state.knowledge_graph_service = KnowledgeGraphService()
    
    # Initialize vector database
    await app.state.vector_service.initialize()
    
    # Initialize knowledge graph
    await app.state.knowledge_graph_service.initialize()
    
    logger.info("SABDE application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SABDE application")
    await app.state.vector_service.close()
    await app.state.knowledge_graph_service.close()
    logger.info("SABDE application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="SABDE - State-of-the-Art Biomedical Discovery Engine",
    description="AI-powered scientific research assistant for hypothesis generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus monitoring removed for lightweight deployment

# Include API routes
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(research.router, prefix="/api/v1", tags=["research"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error("Unhandled exception", exc_info=exc, path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
