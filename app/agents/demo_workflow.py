"""
Demo workflow for SABDE without API keys.
"""
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
import structlog
from app.agents.demo_agents import DemoRAGAgent, DemoExtractionAgent, DemoSynthesisAgent, DemoCriticAgent
from app.models.research import Hypothesis

logger = structlog.get_logger()


class DemoResearchState(TypedDict):
    """Demo state object for the research workflow."""
    # Input
    query: str
    domain: str
    max_hypotheses: int
    max_documents: int
    
    # Intermediate results
    retrieved_documents: List[Any]
    entities: List[Any]
    relationships: List[Any]
    hypotheses: List[Hypothesis]
    critiques: List[Any]
    
    # Workflow control
    current_step: str
    iteration_count: int
    max_iterations: int
    overall_status: str
    
    # Metadata
    total_documents: int
    total_entities: int
    total_relationships: int
    total_hypotheses: int
    processing_time: float
    error_message: str


class DemoResearchWorkflow:
    """Demo workflow orchestrator for the research process."""
    
    def __init__(self):
        # Initialize demo agents
        self.rag_agent = DemoRAGAgent()
        self.extraction_agent = DemoExtractionAgent()
        self.synthesis_agent = DemoSynthesisAgent()
        self.critic_agent = DemoCriticAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the demo workflow graph."""
        workflow = StateGraph(DemoResearchState)
        
        # Add nodes
        workflow.add_node("rag", self._rag_node)
        workflow.add_node("extraction", self._extraction_node)
        workflow.add_node("synthesis", self._synthesis_node)
        workflow.add_node("critic", self._critic_node)
        
        # Add edges
        workflow.set_entry_point("rag")
        workflow.add_edge("rag", "extraction")
        workflow.add_edge("extraction", "synthesis")
        workflow.add_edge("synthesis", "critic")
        workflow.add_edge("critic", END)
        
        return workflow.compile()
    
    async def _rag_node(self, state: DemoResearchState) -> DemoResearchState:
        """Demo RAG node for document retrieval."""
        try:
            logger.info("Executing Demo RAG node")
            result = await self.rag_agent.execute(state)
            
            return {
                **state,
                "retrieved_documents": result.get("retrieved_documents", []),
                "total_documents": result.get("total_documents", 0),
                "current_step": "rag_completed"
            }
            
        except Exception as e:
            logger.error(f"Demo RAG node failed: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "rag_failed"
            }
    
    async def _extraction_node(self, state: DemoResearchState) -> DemoResearchState:
        """Demo extraction node for entity and relationship extraction."""
        try:
            logger.info("Executing Demo extraction node")
            result = await self.extraction_agent.execute(state)
            
            return {
                **state,
                "entities": result.get("entities", []),
                "relationships": result.get("relationships", []),
                "total_entities": result.get("total_entities", 0),
                "total_relationships": result.get("total_relationships", 0),
                "current_step": "extraction_completed"
            }
            
        except Exception as e:
            logger.error(f"Demo extraction node failed: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "extraction_failed"
            }
    
    async def _synthesis_node(self, state: DemoResearchState) -> DemoResearchState:
        """Demo synthesis node for hypothesis generation."""
        try:
            logger.info("Executing Demo synthesis node")
            result = await self.synthesis_agent.execute(state)
            
            return {
                **state,
                "hypotheses": result.get("hypotheses", []),
                "total_hypotheses": result.get("total_hypotheses", 0),
                "current_step": "synthesis_completed"
            }
            
        except Exception as e:
            logger.error(f"Demo synthesis node failed: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "synthesis_failed"
            }
    
    async def _critic_node(self, state: DemoResearchState) -> DemoResearchState:
        """Demo critic node for hypothesis validation."""
        try:
            logger.info("Executing Demo critic node")
            result = await self.critic_agent.execute(state)
            
            return {
                **state,
                "critiques": result.get("critiques", []),
                "overall_status": result.get("overall_status", "PASS"),
                "current_step": "critic_completed"
            }
            
        except Exception as e:
            logger.error(f"Demo critic node failed: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "critic_failed"
            }
    
    async def execute(self, query: str, domain: str = "biomedical", max_hypotheses: int = 5) -> Dict[str, Any]:
        """Execute the demo research workflow."""
        try:
            import time
            start_time = time.time()
            
            # Initialize state
            initial_state = DemoResearchState(
                query=query,
                domain=domain,
                max_hypotheses=max_hypotheses,
                max_documents=10,
                retrieved_documents=[],
                entities=[],
                relationships=[],
                hypotheses=[],
                critiques=[],
                current_step="initialized",
                iteration_count=0,
                max_iterations=1,  # Demo only runs once
                overall_status="PENDING",
                total_documents=0,
                total_entities=0,
                total_relationships=0,
                total_hypotheses=0,
                processing_time=0.0,
                error_message=""
            )
            
            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # Calculate processing time
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Demo workflow execution failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "domain": domain,
                "hypotheses": [],
                "processing_time": 0.0
            }
