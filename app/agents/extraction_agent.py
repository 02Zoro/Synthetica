"""
Entity and relationship extraction agent using BioBERT.
"""
from typing import Dict, Any, List
import structlog
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from app.agents.base_agent import BaseAgent
from app.models.research import Entity, Relationship

logger = structlog.get_logger()


class ExtractionAgent(BaseAgent):
    """Agent responsible for extracting entities and relationships from text."""
    
    def __init__(self):
        super().__init__("Extraction_Agent", "gpt-4o")
        self.ner_pipeline = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize BioBERT models for NER and RE."""
        try:
            # Initialize BioBERT for NER
            model_name = "dmis-lab/biobert-base-cased-v1.1"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            self.ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple"
            )
            
            logger.info("Extraction models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize extraction models: {e}")
            # Fallback to basic NER
            self.ner_pipeline = None
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities and relationships from documents."""
        try:
            documents = state.get("retrieved_documents", [])
            entities = []
            relationships = []
            
            logger.info(f"Extraction Agent processing {len(documents)} documents")
            
            for doc in documents:
                # Extract entities from document text
                doc_entities = await self._extract_entities(doc.abstract)
                doc_relationships = await self._extract_relationships(doc.abstract, doc_entities)
                
                # Add source document info
                for entity in doc_entities:
                    entities.append(entity)
                
                for rel in doc_relationships:
                    rel.source_document = doc.id
                    relationships.append(rel)
            
            result = {
                "entities": entities,
                "relationships": relationships,
                "total_entities": len(entities),
                "total_relationships": len(relationships)
            }
            
            self.log_execution(state, result)
            return result
            
        except Exception as e:
            logger.error(f"Extraction Agent execution failed: {e}")
            return {
                "entities": [],
                "relationships": [],
                "total_entities": 0,
                "total_relationships": 0,
                "error": str(e)
            }
    
    async def _extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text."""
        try:
            if not self.ner_pipeline:
                # Fallback: simple entity extraction
                return self._simple_entity_extraction(text)
            
            # Use BioBERT for NER
            results = self.ner_pipeline(text)
            entities = []
            
            for result in results:
                entity = Entity(
                    text=result["word"],
                    label=result["entity_group"],
                    confidence=result["score"],
                    start_pos=result["start"],
                    end_pos=result["end"]
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return self._simple_entity_extraction(text)
    
    def _simple_entity_extraction(self, text: str) -> List[Entity]:
        """Simple fallback entity extraction."""
        # This is a basic implementation - in production, you'd want more sophisticated extraction
        entities = []
        words = text.split()
        
        # Simple keyword-based extraction
        biomedical_keywords = [
            "protein", "gene", "disease", "drug", "compound", "pathway",
            "cell", "tissue", "organ", "molecule", "enzyme", "receptor"
        ]
        
        for i, word in enumerate(words):
            if word.lower() in biomedical_keywords:
                entity = Entity(
                    text=word,
                    label="BIOMEDICAL_TERM",
                    confidence=0.5,
                    start_pos=i,
                    end_pos=i + 1
                )
                entities.append(entity)
        
        return entities
    
    async def _extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities."""
        try:
            relationships = []
            
            # Simple relationship extraction based on proximity
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    # Check if entities are close in text
                    if abs(entity1.start_pos - entity2.start_pos) < 50:
                        # Determine relationship type based on entity types
                        rel_type = self._determine_relationship_type(entity1, entity2)
                        
                        relationship = Relationship(
                            subject=entity1.text,
                            predicate=rel_type,
                            object=entity2.text,
                            confidence=0.6,
                            source_document=""
                        )
                        relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []
    
    def _determine_relationship_type(self, entity1: Entity, entity2: Entity) -> str:
        """Determine the type of relationship between two entities."""
        # Simple rule-based relationship determination
        if entity1.label == "GENE" and entity2.label == "DISEASE":
            return "ASSOCIATED_WITH"
        elif entity1.label == "DRUG" and entity2.label == "DISEASE":
            return "TREATS"
        elif entity1.label == "PROTEIN" and entity2.label == "PROTEIN":
            return "INTERACTS_WITH"
        else:
            return "RELATED_TO"
