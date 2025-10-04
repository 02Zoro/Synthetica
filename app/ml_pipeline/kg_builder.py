import pandas as pd
import jsonlines
from neo4j import GraphDatabase
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import os
import time
import re
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
INPUT_FILE = "mvp_gene_abstracts.json"

# Neo4j Database Configuration (Assumes Docker default)
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

# STABLE BERT MODEL: Uses the most stable general BERT model for NER.
NLP_PIPELINE = "bert-base-cased" 

# FINAL ENTITY MAP: Expanded mapping to robustly capture general BERT tags (B-, I- prefixes).
# This maximizes the Node count, grouping all scientific concepts under 'Concept'.
ENTITY_MAP = {
    "PER": "Researcher", 
    "B-PER": "Researcher", 
    "I-PER": "Researcher",
    
    "ORG": "Concept", 
    "B-ORG": "Concept",
    "I-ORG": "Concept",
    
    "LOC": "Location",
    "B-LOC": "Location",
    "I-LOC": "Location",

    "MISC": "Concept",     # Catch-all for scientific terms
    "B-MISC": "Concept",
    "I-MISC": "Concept",
}

# Define simple trigger words for Rule-Based Relation Extraction (RE)
TRIGGER_PHRASES = {
    r'\b(regulates|inhibits|modulates|targets)\b': 'REGULATES',
    r'\b(is associated with|causes|leads to|results in)\b': 'ASSOCIATED_WITH',
    r'\b(treats|alleviates|cures|ameliorates)\b': 'TREATS',
}

# ==============================================================================
# 2. NEO4J CONNECTION AND TRANSACTION FUNCTIONS
# ==============================================================================

class Neo4jConnector:
    """Handles connection and Cypher transactions to Neo4j."""
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity() 

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            session.run(query, parameters)
    
    def create_constraint(self, label, property):
        """Ensures uniqueness for nodes, vital for performance."""
        query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property} IS UNIQUE"
        self.run_query(query)

    def upload_node(self, tx, label, name):
        """Creates or updates a node (MERGE)."""
        cypher = (f"MERGE (n:{label} {{name: $name}}) RETURN n")
        tx.run(cypher, name=name)

    def upload_relationship(self, tx, start_label, start_name, end_label, end_name, rel_type, source_pmid):
        """Finds two existing nodes and creates a relationship between them."""
        cypher = (
            f"MATCH (a:{start_label} {{name: $start_name}}) "
            f"MATCH (b:{end_label} {{name: $end_name}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"ON CREATE SET r.source_pmid = $source_pmid, r.count = 1 " 
            f"ON MATCH SET r.count = r.count + 1 " 
            f"RETURN r"
        )
        tx.run(cypher, start_name=start_name, end_name=end_name, rel_type=rel_type, source_pmid=source_pmid)


# ==============================================================================
# 3. EXTRACTION AND UPLOAD PIPELINE
# ==============================================================================

def extract_and_load():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found at {INPUT_FILE}.")
        return

    # --- 1. Load NLP Model ---
    print(f"--- 1. Loading NER Pipeline: {NLP_PIPELINE} ---")
    try:
        model = AutoModelForTokenClassification.from_pretrained(NLP_PIPELINE)
        tokenizer = AutoTokenizer.from_pretrained(NLP_PIPELINE)
        
        tokenizer.model_max_length = 512
        
        ner_pipeline = pipeline(
            "token-classification", 
            model=model, 
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        print("✅ Pipeline loaded successfully!")
    except Exception as e:
        print(f"❌ Fatal: Failed to load BERT model. Error: {e}")
        return

    # --- 2. Connect to Neo4j ---
    print(f"--- 2. Connecting to Neo4j at {URI} ---")
    try:
        db = Neo4jConnector(URI, USER, PASSWORD)
        for label in set(ENTITY_MAP.values()):
             # Only create constraints for labels that are defined (not None)
             if label: 
                 db.create_constraint(label, 'name')
        print("✅ Connection successful. Constraints set.")
    except Exception as e:
        print(f"❌ Fatal: Failed to connect to Neo4j. Is Docker running? Error: {e}")
        return
    
    # --- 3. EXTRACT NODES & EDGES ---
    nodes_created = 0
    edges_created = 0
    abstracts_processed = 0
    print("--- 3. Starting Extraction and Upload (Nodes & Edges) ---")

    with db.driver.session() as session, jsonlines.open(INPUT_FILE) as reader:
        for record in reader:
            abstract = record.get('Abstract', '')
            pmid = record.get('PMID', 'NO_PMID')
            
            if abstract == "No Abstract Available" or len(abstract) < 50:
                continue

            # Process the abstract for NER
            results = ner_pipeline(abstract)
            
            # 3a. NODE EXTRACTION 
            current_entities = []
            for ent in results:
                tag_full = ent['entity_group']
                
                name = ent['word'].strip()
                label = ENTITY_MAP.get(tag_full) # Use the full tag for lookup
                
                # Filter: Must have a valid label and be longer than 2 characters
                if label and len(name) > 2 and name not in ["a", "the", "and", "or", "in"]:
                    session.write_transaction(db.upload_node, label, name)
                    nodes_created += 1
                    current_entities.append({'name': name, 'label': label})

            
            # 3b. RELATION EXTRACTION (Rule-based)
            unique_entities = list({v['name']: v for v in current_entities}.values())
            
            for i in range(len(unique_entities)):
                for j in range(i + 1, len(unique_entities)):
                    entity_a = unique_entities[i]
                    entity_b = unique_entities[j]
                    
                    # Check for trigger phrases (the edge type)
                    for pattern, rel_type in TRIGGER_PHRASES.items():
                        if re.search(pattern, abstract.lower()):
                            # Create the relationship A->[REL]->B
                            session.write_transaction(
                                db.upload_relationship, 
                                entity_a['label'], 
                                entity_a['name'], 
                                entity_b['label'], 
                                entity_b['name'], 
                                rel_type, 
                                pmid
                            )
                            edges_created += 1
                            break # Move to next pair once one relation is found
            
            abstracts_processed += 1
            if abstracts_processed % 500 == 0:
                print(f"  Processed {abstracts_processed} abstracts. Nodes: {nodes_created}, Edges: {edges_created}")
    
    db.close()
    print(f"\n====================================================================")
    print(f"✅ PHASE 1 COMPLETE: {nodes_created} Nodes and {edges_created} Edges Loaded into the KG.")
    print("====================================================================")

# ==============================================================================
if __name__ == "__main__":
    # Ensure Neo4j Docker container is active before running!
    extract_and_load()