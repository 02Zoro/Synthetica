#!/usr/bin/env python3
"""
Initialize the SABDE system with sample data.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.vector_service import VectorService
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.models.research import Document
from app.core.config import settings


async def load_sample_papers():
    """Load sample papers into the vector database."""
    print("Loading sample papers...")
    
    # Initialize vector service
    vector_service = VectorService()
    await vector_service.initialize()
    
    # Load sample papers
    data_path = Path(__file__).parent.parent / "data" / "sample_papers.json"
    with open(data_path, 'r') as f:
        papers_data = json.load(f)
    
    # Convert to Document objects
    documents = []
    for paper in papers_data:
        doc = Document(
            id=paper["id"],
            title=paper["title"],
            abstract=paper["abstract"],
            authors=paper["authors"],
            journal=paper["journal"],
            doi=paper["doi"],
            keywords=paper["keywords"],
            relevance_score=1.0
        )
        documents.append(doc)
    
    # Add documents to vector database
    success = await vector_service.add_documents(documents)
    if success:
        print(f"Successfully loaded {len(documents)} sample papers")
    else:
        print("Failed to load sample papers")
    
    await vector_service.close()


async def initialize_knowledge_graph():
    """Initialize the knowledge graph with sample entities and relationships."""
    print("Initializing knowledge graph...")
    
    # Initialize knowledge graph service
    kg_service = KnowledgeGraphService()
    await kg_service.initialize()
    
    # Sample entities
    from app.models.research import Entity, Relationship
    
    entities = [
        Entity(text="BRCA1", label="GENE", confidence=0.95, start_pos=0, end_pos=5),
        Entity(text="breast cancer", label="DISEASE", confidence=0.90, start_pos=6, end_pos=18),
        Entity(text="PARP inhibitors", label="DRUG", confidence=0.85, start_pos=19, end_pos=33),
        Entity(text="insulin resistance", label="DISEASE", confidence=0.88, start_pos=0, end_pos=17),
        Entity(text="type 2 diabetes", label="DISEASE", confidence=0.92, start_pos=18, end_pos=33),
        Entity(text="metformin", label="DRUG", confidence=0.90, start_pos=34, end_pos=43),
        Entity(text="Alzheimer's disease", label="DISEASE", confidence=0.95, start_pos=0, end_pos=18),
        Entity(text="amyloid-beta", label="PROTEIN", confidence=0.88, start_pos=19, end_pos=31),
        Entity(text="tau protein", label="PROTEIN", confidence=0.85, start_pos=32, end_pos=43)
    ]
    
    relationships = [
        Relationship(subject="BRCA1", predicate="ASSOCIATED_WITH", object="breast cancer", confidence=0.90, source_document="paper_001"),
        Relationship(subject="PARP inhibitors", predicate="TREATS", object="breast cancer", confidence=0.85, source_document="paper_001"),
        Relationship(subject="insulin resistance", predicate="CAUSES", object="type 2 diabetes", confidence=0.88, source_document="paper_002"),
        Relationship(subject="metformin", predicate="TREATS", object="type 2 diabetes", confidence=0.90, source_document="paper_002"),
        Relationship(subject="amyloid-beta", predicate="INTERACTS_WITH", object="tau protein", confidence=0.85, source_document="paper_003")
    ]
    
    # Add entities and relationships
    await kg_service.add_entities(entities, "sample_data")
    await kg_service.add_relationships(relationships)
    
    print("Knowledge graph initialized successfully")
    await kg_service.close()


async def main():
    """Main initialization function."""
    print("Initializing SABDE system...")
    
    try:
        # Load sample papers
        await load_sample_papers()
        
        # Initialize knowledge graph
        await initialize_knowledge_graph()
        
        print("SABDE system initialized successfully!")
        print("\nNext steps:")
        print("1. Start the FastAPI backend: uvicorn app.main:app --reload")
        print("2. Start the React frontend: cd frontend && npm start")
        print("3. Open http://localhost:3000 in your browser")
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
