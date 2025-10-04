"""
Knowledge graph service for Neo4j operations.
"""
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import structlog
from app.core.config import settings
from app.models.research import Entity, Relationship

logger = structlog.get_logger()


class KnowledgeGraphService:
    """Service for Neo4j knowledge graph operations."""
    
    def __init__(self):
        self.driver = None
        
    async def initialize(self):
        """Initialize the Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            # Create constraints and indexes
            await self._create_constraints()
            await self._create_indexes()
            
            logger.info("Knowledge graph service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize knowledge graph service", error=str(e))
            raise
    
    async def _create_constraints(self):
        """Create Neo4j constraints for data integrity."""
        constraints = [
            "CREATE CONSTRAINT gene_id_unique IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE",
            "CREATE CONSTRAINT protein_id_unique IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT disease_id_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT compound_id_unique IF NOT EXISTS FOR (c:Compound) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT pathway_id_unique IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.id IS UNIQUE",
            "CREATE CONSTRAINT article_id_unique IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation failed (may already exist): {e}")
    
    async def _create_indexes(self):
        """Create Neo4j indexes for performance."""
        indexes = [
            "CREATE INDEX gene_name_index IF NOT EXISTS FOR (g:Gene) ON (g.name)",
            "CREATE INDEX protein_name_index IF NOT EXISTS FOR (p:Protein) ON (p.name)",
            "CREATE INDEX disease_name_index IF NOT EXISTS FOR (d:Disease) ON (d.name)",
            "CREATE INDEX compound_name_index IF NOT EXISTS FOR (c:Compound) ON (c.name)",
            "CREATE INDEX article_title_index IF NOT EXISTS FOR (a:Article) ON (a.title)"
        ]
        
        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.warning(f"Index creation failed (may already exist): {e}")
    
    async def add_entities(self, entities: List[Entity], source_document: str) -> bool:
        """Add entities to the knowledge graph."""
        try:
            with self.driver.session() as session:
                for entity in entities:
                    # Map entity labels to node labels
                    node_label = self._map_entity_label(entity.label)
                    
                    query = f"""
                    MERGE (e:{node_label} {{id: $entity_id, name: $entity_name}})
                    SET e.text = $entity_text,
                        e.confidence = $confidence,
                        e.start_pos = $start_pos,
                        e.end_pos = $end_pos,
                        e.last_updated = datetime()
                    
                    MERGE (doc:Article {{id: $source_document}})
                    MERGE (e)-[:EXTRACTED_FROM {{confidence: $confidence}}]->(doc)
                    """
                    
                    session.run(query, {
                        "entity_id": f"{node_label.lower()}_{entity.text.lower().replace(' ', '_')}",
                        "entity_name": entity.text,
                        "entity_text": entity.text,
                        "confidence": entity.confidence,
                        "start_pos": entity.start_pos,
                        "end_pos": entity.end_pos,
                        "source_document": source_document
                    })
            
            logger.info(f"Added {len(entities)} entities to knowledge graph")
            return True
            
        except Exception as e:
            logger.error("Failed to add entities", error=str(e))
            return False
    
    async def add_relationships(self, relationships: List[Relationship]) -> bool:
        """Add relationships to the knowledge graph."""
        try:
            with self.driver.session() as session:
                for rel in relationships:
                    # Map relationship types to edge types
                    edge_type = self._map_relationship_type(rel.predicate)
                    
                    query = f"""
                    MATCH (subj) WHERE subj.name = $subject_name
                    MATCH (obj) WHERE obj.name = $object_name
                    MERGE (subj)-[r:{edge_type}]->(obj)
                    SET r.confidence = $confidence,
                        r.source_document = $source_document,
                        r.last_updated = datetime()
                    """
                    
                    session.run(query, {
                        "subject_name": rel.subject,
                        "object_name": rel.object,
                        "confidence": rel.confidence,
                        "source_document": rel.source_document
                    })
            
            logger.info(f"Added {len(relationships)} relationships to knowledge graph")
            return True
            
        except Exception as e:
            logger.error("Failed to add relationships", error=str(e))
            return False
    
    async def find_related_entities(self, entity_name: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find entities related to a given entity."""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (start {{name: $entity_name}})
                MATCH path = (start)-[*1..{max_depth}]-(related)
                RETURN DISTINCT related, length(path) as distance
                ORDER BY distance, related.name
                LIMIT 50
                """
                
                result = session.run(query, entity_name=entity_name)
                entities = []
                
                for record in result:
                    entity = dict(record["related"])
                    entity["distance"] = record["distance"]
                    entities.append(entity)
                
                return entities
                
        except Exception as e:
            logger.error("Failed to find related entities", error=str(e))
            return []
    
    async def get_entity_paths(self, entity1: str, entity2: str, max_length: int = 3) -> List[Dict[str, Any]]:
        """Find paths between two entities."""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (a {{name: $entity1}}), (b {{name: $entity2}})
                MATCH path = shortestPath((a)-[*1..{max_length}]-(b))
                RETURN path, length(path) as path_length
                ORDER BY path_length
                LIMIT 10
                """
                
                result = session.run(query, entity1=entity1, entity2=entity2)
                paths = []
                
                for record in result:
                    path = record["path"]
                    path_length = record["path_length"]
                    
                    # Convert path to list of nodes and relationships
                    nodes = []
                    relationships = []
                    
                    for i, node in enumerate(path.nodes):
                        nodes.append(dict(node))
                    
                    for i, rel in enumerate(path.relationships):
                        relationships.append({
                            "type": rel.type,
                            "start": dict(rel.start_node),
                            "end": dict(rel.end_node)
                        })
                    
                    paths.append({
                        "nodes": nodes,
                        "relationships": relationships,
                        "length": path_length
                    })
                
                return paths
                
        except Exception as e:
            logger.error("Failed to get entity paths", error=str(e))
            return []
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        try:
            with self.driver.session() as session:
                # Get node counts by label
                node_counts = {}
                result = session.run("CALL db.labels()")
                labels = [record["label"] for record in result]
                
                for label in labels:
                    count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    node_counts[label] = count_result.single()["count"]
                
                # Get relationship counts by type
                rel_counts = {}
                result = session.run("CALL db.relationshipTypes()")
                rel_types = [record["relationshipType"] for record in result]
                
                for rel_type in rel_types:
                    count_result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                    rel_counts[rel_type] = count_result.single()["count"]
                
                return {
                    "node_counts": node_counts,
                    "relationship_counts": rel_counts,
                    "total_nodes": sum(node_counts.values()),
                    "total_relationships": sum(rel_counts.values())
                }
                
        except Exception as e:
            logger.error("Failed to get graph stats", error=str(e))
            return {"node_counts": {}, "relationship_counts": {}, "total_nodes": 0, "total_relationships": 0}
    
    def _map_entity_label(self, label: str) -> str:
        """Map entity labels to Neo4j node labels."""
        mapping = {
            "GENE": "Gene",
            "PROTEIN": "Protein",
            "DISEASE": "Disease",
            "DRUG": "Compound",
            "COMPOUND": "Compound",
            "PATHWAY": "Pathway",
            "CELL_LINE": "CellLine",
            "ORGANISM": "Organism"
        }
        return mapping.get(label.upper(), "Entity")
    
    def _map_relationship_type(self, predicate: str) -> str:
        """Map relationship predicates to Neo4j relationship types."""
        mapping = {
            "REGULATES": "REGULATES",
            "INTERACTS_WITH": "INTERACTS_WITH",
            "ASSOCIATED_WITH": "ASSOCIATED_WITH",
            "TREATS": "TREATS",
            "CAUSES": "CAUSES",
            "PART_OF": "PART_OF",
            "LOCATED_IN": "LOCATED_IN"
        }
        return mapping.get(predicate.upper(), "RELATED_TO")
    
    async def close(self):
        """Close the Neo4j connection."""
        try:
            if self.driver:
                self.driver.close()
            logger.info("Knowledge graph service closed")
            
        except Exception as e:
            logger.error("Error closing knowledge graph service", error=str(e))
