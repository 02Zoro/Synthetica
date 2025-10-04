"""
Enhanced SABDE application with ML Pipeline integration
"""
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from app.api.routes import health, research_ml
from app.services.vector_service import VectorService
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.ml_pipeline_service_real import ml_pipeline_service

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global services
vector_service = None
kg_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global vector_service, kg_service
    
    # Startup
    logger.info("Starting Synthetica ML-Enhanced Application")
    
    try:
        # Initialize services
        vector_service = VectorService()
        kg_service = KnowledgeGraphService()
        
        # Initialize ML pipeline if enabled
        ml_enabled = os.getenv("ML_PIPELINE_ENABLED", "true").lower() == "true"
        if ml_enabled:
            logger.info("Initializing ML Pipeline...")
            ml_initialized = await ml_pipeline_service.initialize()
            if ml_initialized:
                logger.info("ML Pipeline initialized successfully")
            else:
                logger.warning("ML Pipeline initialization failed - falling back to demo mode")
        else:
            logger.info("ML Pipeline disabled - using demo mode")
        
        logger.info("Synthetica ML-Enhanced Application startup complete")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Synthetica ML-Enhanced Application shutdown")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Synthetica ML-Enhanced - Scientific Research Assistant",
        description="AI-powered hypothesis generation with advanced ML pipeline",
        version="2.0.0-ml",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8000", "http://localhost:8001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(research_ml.router, prefix="/api/v1")
    
    @app.get("/")
    async def root():
        """Root endpoint with system information."""
        ml_status = ml_pipeline_service.get_status()
        
        return {
            "message": "Synthetica ML-Enhanced - Scientific Research Assistant",
            "version": "2.0.0-ml",
            "description": "AI-powered hypothesis generation with advanced ML pipeline",
            "features": [
                "Graph Neural Network hypothesis generation",
                "Knowledge graph construction",
                "Vector similarity search",
                "Multi-agent AI workflow",
                "Real-time hypothesis generation"
            ],
            "ml_pipeline_status": ml_status,
            "endpoints": {
                "health": "/api/v1/health",
                "research": "/api/v1/research/generate",
                "ml_status": "/api/v1/research/ml-status",
                "docs": "/docs"
            }
        }
    
    @app.get("/api/v1/status")
    async def system_status():
        """Get comprehensive system status."""
        return {
            "application": "Synthetica ML-Enhanced",
            "version": "2.0.0-ml",
            "status": "healthy",
            "ml_pipeline": ml_pipeline_service.get_status(),
            "services": {
                "vector_service": "available",
                "knowledge_graph": "available",
                "ml_pipeline": "available" if ml_pipeline_service.initialized else "demo_mode"
            }
        }
    
    return app

# Create the application
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8001"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    uvicorn.run(
        "app.main_ml:app",
        host=host,
        port=port,
        reload=debug,
        log_level=log_level.lower()
    )

