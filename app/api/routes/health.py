"""
Health check endpoints for monitoring system status.
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any
import structlog
from app.services.vector_service import VectorService
from app.services.knowledge_graph_service import KnowledgeGraphService

logger = structlog.get_logger()
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "sabde"}


@router.get("/health/detailed")
async def detailed_health_check(
    vector_service: VectorService = Depends(),
    kg_service: KnowledgeGraphService = Depends()
) -> Dict[str, Any]:
    """Detailed health check with service status."""
    try:
        # Check vector database
        vector_stats = await vector_service.get_collection_stats()
        
        # Check knowledge graph
        kg_stats = await kg_service.get_graph_stats()
        
        return {
            "status": "healthy",
            "services": {
                "vector_database": {
                    "status": "healthy",
                    "total_documents": vector_stats.get("total_documents", 0)
                },
                "knowledge_graph": {
                    "status": "healthy",
                    "total_nodes": kg_stats.get("total_nodes", 0),
                    "total_relationships": kg_stats.get("total_relationships", 0)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
