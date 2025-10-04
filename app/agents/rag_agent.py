"""
RAG (Retrieval-Augmented Generation) agent for document retrieval.
"""
from typing import Dict, Any, List
import structlog
from app.agents.base_agent import BaseAgent
from app.services.vector_service import VectorService
from app.models.research import Document

logger = structlog.get_logger()


class RAGAgent(BaseAgent):
    """Agent responsible for retrieving relevant documents."""
    
    def __init__(self, vector_service: VectorService):
        super().__init__("RAG_Agent", "gpt-4o")
        self.vector_service = vector_service
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant documents for the query."""
        try:
            query = state.get("query", "")
            max_docs = state.get("max_documents", 10)
            domain = state.get("domain", "biomedical")
            
            logger.info(f"RAG Agent executing for query: {query}")
            
            # Search for relevant documents
            search_results = await self.vector_service.search_documents(
                query=query,
                n_results=max_docs
            )
            
            # Convert results to Document objects
            documents = []
            for result in search_results:
                doc = Document(
                    id=result["id"],
                    title=result["metadata"].get("title", "Unknown Title"),
                    abstract=result["text"],
                    authors=result["metadata"].get("authors", []),
                    journal=result["metadata"].get("journal"),
                    doi=result["metadata"].get("doi"),
                    url=result["metadata"].get("url"),
                    keywords=result["metadata"].get("keywords", []),
                    relevance_score=result["similarity_score"]
                )
                documents.append(doc)
            
            # Update state with retrieved documents
            result = {
                "retrieved_documents": documents,
                "total_documents": len(documents),
                "query": query,
                "domain": domain
            }
            
            self.log_execution(state, result)
            return result
            
        except Exception as e:
            logger.error(f"RAG Agent execution failed: {e}")
            return {
                "retrieved_documents": [],
                "total_documents": 0,
                "error": str(e)
            }
