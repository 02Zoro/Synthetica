"""
Vector database service for document storage and retrieval.
"""
import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import structlog
from app.core.config import settings
from app.models.research import Document

logger = structlog.get_logger()


class VectorService:
    """Service for vector database operations."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        
    async def initialize(self):
        """Initialize the vector database and embedding model."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="scientific_documents",
                metadata={"description": "Scientific papers and abstracts"}
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            
            logger.info("Vector service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize vector service", error=str(e))
            raise
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector database."""
        try:
            if not documents:
                return True
                
            # Prepare data for ChromaDB
            ids = [doc.id for doc in documents]
            texts = [f"{doc.title}\n\n{doc.abstract}" for doc in documents]
            metadatas = [
                {
                    "title": doc.title,
                    "authors": doc.authors,
                    "journal": doc.journal,
                    "doi": doc.doi,
                    "url": doc.url,
                    "keywords": doc.keywords,
                    "publication_date": doc.publication_date.isoformat() if doc.publication_date else None
                }
                for doc in documents
            ]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to vector database")
            return True
            
        except Exception as e:
            logger.error("Failed to add documents", error=str(e))
            return False
    
    async def search_documents(
        self, 
        query: str, 
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            # Format results
            documents = []
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (0-1)
                similarity_score = 1 - distance
                
                document = {
                    "id": doc_id,
                    "text": results['documents'][0][i],
                    "similarity_score": similarity_score,
                    "metadata": metadata
                }
                documents.append(document)
            
            logger.info(f"Found {len(documents)} relevant documents for query")
            return documents
            
        except Exception as e:
            logger.error("Failed to search documents", error=str(e))
            return []
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        try:
            results = self.collection.get(ids=[doc_id])
            if not results['ids']:
                return None
                
            return {
                "id": results['ids'][0],
                "text": results['documents'][0],
                "metadata": results['metadatas'][0]
            }
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}", error=str(e))
            return None
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector database."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}", error=str(e))
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name
            }
            
        except Exception as e:
            logger.error("Failed to get collection stats", error=str(e))
            return {"total_documents": 0, "collection_name": "unknown"}
    
    async def close(self):
        """Close the vector database connection."""
        try:
            if self.client:
                # ChromaDB doesn't require explicit closing
                pass
            logger.info("Vector service closed")
            
        except Exception as e:
            logger.error("Error closing vector service", error=str(e))
