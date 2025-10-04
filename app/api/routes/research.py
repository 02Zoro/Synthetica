"""
Research endpoints for hypothesis generation and analysis.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List
import structlog
import uuid
from datetime import datetime

from app.models.research import ResearchQuery, ResearchResponse, WorkflowStatus
from app.agents.workflow import ResearchWorkflow
from app.agents.demo_workflow import DemoResearchWorkflow
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
    """Generate research hypotheses from a query."""
    try:
        logger.info(f"Generating hypotheses for query: {query.query}")
        
        # Check if we're in demo mode (no API keys)
        import os
        demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
        
        if demo_mode:
            # Use demo workflow
            workflow = DemoResearchWorkflow()
        else:
            # Use real workflow
            workflow = ResearchWorkflow(vector_service)
        
        # Execute workflow
        result = await workflow.execute(
            query=query.query,
            domain=query.domain.value,
            max_hypotheses=query.max_hypotheses
        )
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
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
                "status": result.get("overall_status", "UNKNOWN")
            }
        )
        
        logger.info(f"Generated {len(response.hypotheses)} hypotheses")
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
    """Generate hypotheses asynchronously."""
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
            _execute_workflow_async,
            workflow_id,
            query,
            vector_service,
            kg_service
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Hypothesis generation started"
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


async def _execute_workflow_async(
    workflow_id: str,
    query: ResearchQuery,
    vector_service: VectorService,
    kg_service: KnowledgeGraphService
):
    """Execute workflow in background."""
    try:
        # Update workflow status
        if workflow_id in active_workflows:
            active_workflows[workflow_id].status = "running"
            active_workflows[workflow_id].current_step = "initialized"
            active_workflows[workflow_id].progress = 0.1
        
        # Initialize workflow
        workflow = ResearchWorkflow(vector_service)
        
        # Execute workflow
        result = await workflow.execute(
            query=query.query,
            domain=query.domain.value,
            max_hypotheses=query.max_hypotheses
        )
        
        # Update workflow status
        if workflow_id in active_workflows:
            active_workflows[workflow_id].status = "completed"
            active_workflows[workflow_id].current_step = "completed"
            active_workflows[workflow_id].progress = 1.0
            active_workflows[workflow_id].updated_at = datetime.utcnow()
        
        logger.info(f"Workflow {workflow_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")
        if workflow_id in active_workflows:
            active_workflows[workflow_id].status = "failed"
            active_workflows[workflow_id].current_step = "error"
            active_workflows[workflow_id].updated_at = datetime.utcnow()
