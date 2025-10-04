#!/usr/bin/env python3
"""
Script to populate Neo4j with real gene data from your ChromaDB
"""
import sqlite3
import os
import json
from neo4j import GraphDatabase
import pandas as pd

def populate_neo4j_with_real_data():
    """Populate Neo4j with real gene data from ChromaDB"""
    
    print("Populating Neo4j with real gene data...")
    print("=" * 60)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'synthetica_password'))
    
    try:
        with driver.session() as session:
            # Clear existing data
            print("Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Connect to ChromaDB to get real data
            print("Extracting real data from ChromaDB...")
            db_path = "app/ml_pipeline/chroma_db_gene_mvp_new/chroma.sqlite3"
            
            if not os.path.exists(db_path):
                print(f"ChromaDB not found at {db_path}")
                return False
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get sample documents with their metadata
            cursor.execute("""
                SELECT 
                    e.embedding_id,
                    em_doc.string_value as document,
                    em_source.string_value as source,
                    em_chunk.int_value as chunk_index
                FROM embeddings e
                LEFT JOIN embedding_metadata em_doc ON e.id = em_doc.id AND em_doc.key = 'chroma:document'
                LEFT JOIN embedding_metadata em_source ON e.id = em_source.id AND em_source.key = 'source'
                LEFT JOIN embedding_metadata em_chunk ON e.id = em_chunk.id AND em_chunk.key = 'chunk_index'
                WHERE em_doc.string_value IS NOT NULL
                LIMIT 100
            """)
            
            documents = cursor.fetchall()
            print(f"Found {len(documents)} documents to process")
            
            # Create nodes and relationships from real data
            for i, (doc_id, document, source, chunk_index) in enumerate(documents):
                if document:
                    # Create document node
                    session.run("""
                        CREATE (d:Document {
                            id: $doc_id,
                            content: $document,
                            source: $source,
                            chunk_index: $chunk_index
                        })
                    """, doc_id=doc_id, document=document, source=source, chunk_index=chunk_index)
                    
                    # Extract gene names and create gene nodes
                    gene_keywords = ['BRCA1', 'BRCA2', 'p53', 'TP53', 'EGFR', 'MYC', 'KRAS', 'APC', 'PTEN', 'RB1']
                    found_genes = []
                    
                    for gene in gene_keywords:
                        if gene.lower() in document.lower():
                            found_genes.append(gene)
                            
                            # Create gene node
                            session.run("""
                                MERGE (g:Gene {name: $gene_name})
                                ON CREATE SET g.type = 'Gene'
                            """, gene_name=gene)
                            
                            # Create relationship between document and gene
                            session.run("""
                                MATCH (d:Document {id: $doc_id})
                                MATCH (g:Gene {name: $gene_name})
                                CREATE (d)-[:MENTIONS]->(g)
                            """, doc_id=doc_id, gene_name=gene)
                    
                    # Extract disease terms
                    disease_keywords = ['cancer', 'tumor', 'stroke', 'Alzheimer', 'diabetes', 'hypertension']
                    for disease in disease_keywords:
                        if disease.lower() in document.lower():
                            # Create disease node
                            session.run("""
                                MERGE (disease:Disease {name: $disease_name})
                                ON CREATE SET disease.type = 'Disease'
                            """, disease_name=disease)
                            
                            # Create relationship
                            session.run("""
                                MATCH (d:Document {id: $doc_id})
                                MATCH (disease:Disease {name: $disease_name})
                                CREATE (d)-[:DISCUSSES]->(disease)
                            """, doc_id=doc_id, disease_name=disease)
                    
                    if i % 10 == 0:
                        print(f"Processed {i+1}/{len(documents)} documents")
            
            conn.close()
            
            # Get statistics
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
            stats = result.data()
            
            print("\nNeo4j populated with real data:")
            for stat in stats:
                print(f"  {stat['labels']}: {stat['count']} nodes")
            
            print("\n" + "=" * 60)
            print("Neo4j successfully populated with real gene data!")
            return True
            
    except Exception as e:
        print(f"Error populating Neo4j: {e}")
        return False
    finally:
        driver.close()

if __name__ == "__main__":
    populate_neo4j_with_real_data()
