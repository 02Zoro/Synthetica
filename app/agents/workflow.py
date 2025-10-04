"""
LangGraph workflow for orchestrating the multi-agent research process.
"""
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import structlog

from app.agents.rag_agent import RAGAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.synthesis_agent import SynthesisAgent
from app.agents.critic_agent import CriticAgent
from app.services.vector_service import VectorService
from app.models.research import Hypothesis

logger = structlog.get_logger()


class ResearchState(TypedDict):
    """State object for the research workflow."""
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


class ResearchWorkflow:
    """Main workflow orchestrator for the research process."""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        
        # Initialize agents
        self.rag_agent = RAGAgent(vector_service)
        self.extraction_agent = ExtractionAgent()
        self.synthesis_agent = SynthesisAgent()
        self.critic_agent = CriticAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("rag", self._rag_node)
        workflow.add_node("extraction", self._extraction_node)
        workflow.add_node("synthesis", self._synthesis_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("reflect", self._reflect_node)
        
        # Add edges
        workflow.set_entry_point("rag")
        workflow.add_edge("rag", "extraction")
        workflow.add_edge("extraction", "synthesis")
        workflow.add_edge("synthesis", "critic")
        
        # Add conditional edges for self-correction loop
        workflow.add_conditional_edges(
            "critic",
            self._should_continue,
            {
                "continue": "reflect",
                "end": END
            }
        )
        workflow.add_edge("reflect", "synthesis")
        
        return workflow.compile()
    
    async def _rag_node(self, state: ResearchState) -> ResearchState:
        """RAG node for document retrieval."""
        try:
            logger.info("Executing RAG node")
            result = await self.rag_agent.execute(state)
            
            return {
                **state,
                "retrieved_documents": result.get("retrieved_documents", []),
                "total_documents": result.get("total_documents", 0),
                "current_step": "rag_completed"
            }
            
        except Exception as e:
            logger.error(f"RAG node failed: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "rag_failed"
            }
    
    async def _extraction_node(self, state: ResearchState) -> ResearchState:
        """Extraction node for entity and relationship extraction."""
        try:
            logger.info("Executing extraction node")
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
            logger.error(f"Extraction node failed: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "extraction_failed"
            }
    
    async def _synthesis_node(self, state: ResearchState) -> ResearchState:
        """Synthesis node for hypothesis generation."""
        try:
            logger.info("Executing synthesis node")
            result = await self.synthesis_agent.execute(state)
            
            return {
                **state,
                "hypotheses": result.get("hypotheses", []),
                "total_hypotheses": result.get("total_hypotheses", 0),
                "current_step": "synthesis_completed"
            }
            
        except Exception as e:
            logger.error(f"Synthesis node failed: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "synthesis_failed"
            }
    
    async def _critic_node(self, state: ResearchState) -> ResearchState:
        """Critic node for hypothesis validation."""
        try:
            logger.info("Executing critic node")
            result = await self.critic_agent.execute(state)
            
            return {
                **state,
                "critiques": result.get("critiques", []),
                "overall_status": result.get("overall_status", "FAIL"),
                "current_step": "critic_completed"
            }
            
        except Exception as e:
            logger.error(f"Critic node failed: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "critic_failed"
            }
    
    async def _reflect_node(self, state: ResearchState) -> ResearchState:
        """Reflect node for incorporating feedback."""
        try:
            logger.info("Executing reflect node")
            
            # Update iteration count
            iteration_count = state.get("iteration_count", 0) + 1
            
            # Incorporate critic feedback into the state
            critiques = state.get("critiques", [])
            feedback_summary = self._summarize_feedback(critiques)
            
            return {
                **state,
                "iteration_count": iteration_count,
                "current_step": "reflect_completed",
                "feedback_summary": feedback_summary
            }
            
        except Exception as e:
            logger.error(f"Reflect node failed: {e}")
            return {
                **state,
                "error_message": str(e),
                "current_step": "reflect_failed"
            }
    
    def _should_continue(self, state: ResearchState) -> str:
        """Determine whether to continue the self-correction loop."""
        overall_status = state.get("overall_status", "FAIL")
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        # End if we have a pass or if we've reached max iterations
        if overall_status == "PASS" or iteration_count >= max_iterations:
            return "end"
        
        # Continue if we need revision and haven't hit max iterations
        if overall_status == "NEEDS_REVISION":
            return "continue"
        
        # End for any other status
        return "end"
    
    def _summarize_feedback(self, critiques: List[Dict[str, Any]]) -> str:
        """Summarize critic feedback for the reflect node."""
        if not critiques:
            return "No feedback available"
        
        feedback_parts = []
        for critique in critiques:
            status = critique.get("status", "UNKNOWN")
            feedback = critique.get("feedback", "")
            feedback_parts.append(f"Status: {status}\nFeedback: {feedback[:200]}...")
        
        return "\n\n".join(feedback_parts)
    
    async def execute(self, query: str, domain: str = "biomedical", max_hypotheses: int = 5) -> Dict[str, Any]:
        """Execute the complete research workflow."""
        try:
            # Initialize state
            initial_state = ResearchState(
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
                max_iterations=3,
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
            import time
            result["processing_time"] = time.time() - result.get("start_time", time.time())
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "domain": domain,
                "hypotheses": [],
                "processing_time": 0.0
            }
