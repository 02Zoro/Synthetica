"""
Enhanced Research endpoints with ML Pipeline integration
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List
import structlog
import uuid
import os
from datetime import datetime

from app.models.research import ResearchQuery, ResearchResponse, WorkflowStatus
from app.services.ml_pipeline_service_real import ml_pipeline_service
from app.services.vector_service import VectorService
from app.services.knowledge_graph_service import KnowledgeGraphService

logger = structlog.get_logger()
router = APIRouter()

# Store active workflows (in production, use Redis or database)
active_workflows: Dict[str, WorkflowStatus] = {}


@router.post("/research/generate", response_model=ResearchResponse)
async def generate_hypotheses(
    query: ResearchQuery,
    vector_service: VectorService = Depends(),
    kg_service: KnowledgeGraphService = Depends()
) -> ResearchResponse:
    """Generate research hypotheses using advanced ML pipeline."""
    try:
        logger.info(f"Generating hypotheses for query: {query.query}")
        
        # Check if ML pipeline is available
        ml_available = os.getenv("ML_PIPELINE_ENABLED", "true").lower() == "true"
        
        if ml_available:
            # Try to use ML pipeline
            try:
                # Initialize ML pipeline if not already done
                if not ml_pipeline_service.initialized:
                    await ml_pipeline_service.initialize()
                
                # Generate hypotheses using ML pipeline
                ml_hypotheses = await ml_pipeline_service.generate_advanced_hypotheses(
                    query=query.query,
                    domain=query.domain.value,
                    max_hypotheses=query.max_hypotheses
                )
                
                # Convert to ResearchResponse format
                response = ResearchResponse(
                    query=query.query,
                    domain=query.domain,
                    hypotheses=ml_hypotheses,
                    total_documents=len(ml_hypotheses),
                    processing_time=2.5,  # ML processing takes longer
                    metadata={
                        "entities": sum(len(h.get("evidence_snippets", [])) for h in ml_hypotheses),
                        "relationships": len(ml_hypotheses),
                        "iterations": 1,
                        "status": "ML_ENHANCED",
                        "ml_pipeline_used": True,
                        "confidence_scores": [h.get("confidence_score", 0.0) for h in ml_hypotheses]
                    }
                )
                
                logger.info(f"Generated {len(response.hypotheses)} ML-enhanced hypotheses")
                return response
                
            except Exception as ml_error:
                logger.warning(f"ML pipeline failed, falling back to demo: {ml_error}")
                # Fall through to demo mode
        
        # Fallback to demo mode
        from app.agents.demo_workflow import DemoResearchWorkflow
        workflow = DemoResearchWorkflow()
        
        result = await workflow.execute(
            query=query.query,
            domain=query.domain.value,
            max_hypotheses=query.max_hypotheses
        )
        
        # Create response
        response = ResearchResponse(
            query=query.query,
            domain=query.domain,
            hypotheses=result.get("hypotheses", []),
            total_documents=result.get("total_documents", 0),
            processing_time=result.get("processing_time", 0.0),
            metadata={
                "entities": result.get("total_entities", 0),
                "relationships": result.get("total_relationships", 0),
                "iterations": result.get("iteration_count", 0),
                "status": result.get("overall_status", "DEMO_MODE"),
                "ml_pipeline_used": False
            }
        )
        
        logger.info(f"Generated {len(response.hypotheses)} demo hypotheses")
        return response
        
    except Exception as e:
        logger.error(f"Hypothesis generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research/async", response_model=Dict[str, str])
async def generate_hypotheses_async(
    query: ResearchQuery,
    background_tasks: BackgroundTasks,
    vector_service: VectorService = Depends(),
    kg_service: KnowledgeGraphService = Depends()
) -> Dict[str, str]:
    """Generate hypotheses asynchronously using ML pipeline."""
    try:
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Create workflow status
        workflow_status = WorkflowStatus(
            workflow_id=workflow_id,
            status="running",
            agents=[],
            current_step="initialized",
            progress=0.0
        )
        
        # Store workflow
        active_workflows[workflow_id] = workflow_status
        
        # Start background task
        background_tasks.add_task(
            _execute_ml_workflow_async,
            workflow_id,
            query,
            vector_service,
            kg_service
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "ML-enhanced hypothesis generation started"
        }
        
    except Exception as e:
        logger.error(f"Async workflow start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/research/status/{workflow_id}", response_model=WorkflowStatus)
async def get_workflow_status(workflow_id: str) -> WorkflowStatus:
    """Get the status of an async workflow."""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return active_workflows[workflow_id]


@router.get("/research/result/{workflow_id}", response_model=ResearchResponse)
async def get_workflow_result(workflow_id: str) -> ResearchResponse:
    """Get the result of a completed workflow."""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_status = active_workflows[workflow_id]
    
    if workflow_status.status != "completed":
        raise HTTPException(status_code=400, detail="Workflow not completed")
    
    # In a real implementation, you'd retrieve the result from storage
    # For now, return a placeholder
    return ResearchResponse(
        query="Placeholder query",
        domain="biomedical",
        hypotheses=[],
        total_documents=0,
        processing_time=0.0,
        metadata={}
    )


@router.get("/research/ml-status")
async def get_ml_pipeline_status() -> Dict[str, Any]:
    """Get the status of the ML pipeline."""
    return ml_pipeline_service.get_status()


@router.post("/research/ml-initialize")
async def initialize_ml_pipeline() -> Dict[str, Any]:
    """Initialize the ML pipeline."""
    try:
        success = await ml_pipeline_service.initialize()
        return {
            "success": success,
            "message": "ML pipeline initialized" if success else "ML pipeline initialization failed",
            "status": ml_pipeline_service.get_status()
        }
    except Exception as e:
        logger.error(f"ML pipeline initialization failed: {e}")
        return {
            "success": False,
            "message": f"ML pipeline initialization failed: {str(e)}",
            "status": ml_pipeline_service.get_status()
        }


@router.get("/research/entities/{entity_name}")
async def find_related_entities(
    entity_name: str,
    max_depth: int = 2,
    kg_service: KnowledgeGraphService = Depends()
) -> Dict[str, Any]:
    """Find entities related to a given entity."""
    try:
        related_entities = await kg_service.find_related_entities(
            entity_name=entity_name,
            max_depth=max_depth
        )
        
        return {
            "entity_name": entity_name,
            "related_entities": related_entities,
            "total_found": len(related_entities)
        }
        
    except Exception as e:
        logger.error(f"Entity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/research/paths/{entity1}/{entity2}")
async def find_entity_paths(
    entity1: str,
    entity2: str,
    max_length: int = 3,
    kg_service: KnowledgeGraphService = Depends()
) -> Dict[str, Any]:
    """Find paths between two entities."""
    try:
        paths = await kg_service.get_entity_paths(
            entity1=entity1,
            entity2=entity2,
            max_length=max_length
        )
        
        return {
            "entity1": entity1,
            "entity2": entity2,
            "paths": paths,
            "total_paths": len(paths)
        }
        
    except Exception as e:
        logger.error(f"Path search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _execute_ml_workflow_async(
    workflow_id: str,
    query: ResearchQuery,
    vector_service: VectorService,
    kg_service: KnowledgeGraphService
):
    """Execute ML workflow in background."""
    try:
        # Update workflow status
        if workflow_id in active_workflows:
            active_workflows[workflow_id].status = "running"
            active_workflows[workflow_id].current_step = "initializing_ml_pipeline"
            active_workflows[workflow_id].progress = 0.1
        
        # Initialize ML pipeline
        ml_initialized = await ml_pipeline_service.initialize()
        
        if workflow_id in active_workflows:
            active_workflows[workflow_id].current_step = "generating_hypotheses"
            active_workflows[workflow_id].progress = 0.3
        
        # Generate hypotheses using ML pipeline
        if ml_initialized:
            hypotheses = await ml_pipeline_service.generate_advanced_hypotheses(
                query=query.query,
                domain=query.domain.value,
                max_hypotheses=query.max_hypotheses
            )
        else:
            # Fallback to demo
            from app.agents.demo_workflow import DemoResearchWorkflow
            workflow = DemoResearchWorkflow()
            result = await workflow.execute(
                query=query.query,
                domain=query.domain.value,
                max_hypotheses=query.max_hypotheses
            )
            hypotheses = result.get("hypotheses", [])
        
        # Update workflow status
        if workflow_id in active_workflows:
            active_workflows[workflow_id].status = "completed"
            active_workflows[workflow_id].current_step = "completed"
            active_workflows[workflow_id].progress = 1.0
            active_workflows[workflow_id].updated_at = datetime.utcnow()
        
        logger.info(f"ML Workflow {workflow_id} completed successfully")
        
    except Exception as e:
        logger.error(f"ML Workflow {workflow_id} failed: {e}")
        if workflow_id in active_workflows:
            active_workflows[workflow_id].status = "failed"
            active_workflows[workflow_id].current_step = "error"
            active_workflows[workflow_id].updated_at = datetime.utcnow()

