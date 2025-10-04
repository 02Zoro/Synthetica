#!/usr/bin/env python3
"""
Script to populate Neo4j with ALL real gene data from ChromaDB
"""
import sqlite3
import os
import json
from neo4j import GraphDatabase
import pandas as pd
import re
from collections import Counter

def extract_genes_and_diseases(text):
    """Extract gene names and disease terms from text"""
    
    # Common gene names (expanded list)
    gene_patterns = [
        r'\bBRCA[12]\b', r'\bTP53\b', r'\bp53\b', r'\bEGFR\b', r'\bMYC\b', r'\bKRAS\b', 
        r'\bAPC\b', r'\bPTEN\b', r'\bRB1\b', r'\bVHL\b', r'\bCDKN2A\b', r'\bATM\b',
        r'\bCHEK2\b', r'\bPALB2\b', r'\bBRIP1\b', r'\bRAD51C\b', r'\bRAD51D\b',
        r'\bBARD1\b', r'\bMLH1\b', r'\bMSH2\b', r'\bMSH6\b', r'\bPMS2\b', r'\bEPCAM\b',
        r'\bCDH1\b', r'\bSTK11\b', r'\bSMAD4\b', r'\bBMPR1A\b', r'\bGREM1\b',
        r'\bPIK3CA\b', r'\bAKT1\b', r'\bMTOR\b', r'\bPTEN\b', r'\bTSC1\b', r'\bTSC2\b',
        r'\bNF1\b', r'\bNF2\b', r'\bVHL\b', r'\bRET\b', r'\bMET\b', r'\bALK\b',
        r'\bROS1\b', r'\bNTRK1\b', r'\bNTRK2\b', r'\bNTRK3\b', r'\bFGFR1\b', r'\bFGFR2\b',
        r'\bFGFR3\b', r'\bFGFR4\b', r'\bPDGFRA\b', r'\bPDGFRB\b', r'\bKIT\b', r'\bFLT3\b',
        r'\bJAK2\b', r'\bJAK3\b', r'\bSTAT3\b', r'\bSTAT5\b', r'\bBCL2\b', r'\bBCL6\b',
        r'\bMYC\b', r'\bCCND1\b', r'\bCDK4\b', r'\bCDK6\b', r'\bRB1\b', r'\bE2F1\b',
        r'\bMDM2\b', r'\bMDM4\b', r'\bARF\b', r'\bINK4A\b', r'\bARF\b', r'\bP14ARF\b',
        r'\bP16INK4A\b', r'\bP15INK4B\b', r'\bP18INK4C\b', r'\bP19INK4D\b'
    ]
    
    # Disease terms
    disease_patterns = [
        r'\bcancer\b', r'\btumor\b', r'\bneoplasm\b', r'\bcarcinoma\b', r'\bsarcoma\b',
        r'\bleukemia\b', r'\blymphoma\b', r'\bmelanoma\b', r'\bglioblastoma\b',
        r'\bstroke\b', r'\bischemia\b', r'\bAlzheimer\b', r'\bdementia\b',
        r'\bdiabetes\b', r'\bhypertension\b', r'\bobesity\b', r'\bmetabolic\b',
        r'\bcardiovascular\b', r'\bheart\b', r'\bcoronary\b', r'\bmyocardial\b',
        r'\bneurodegenerative\b', r'\bParkinson\b', r'\bHuntington\b', r'\bALS\b',
        r'\bmultiple sclerosis\b', r'\bMS\b', r'\bepilepsy\b', r'\bseizure\b',
        r'\bautism\b', r'\bADHD\b', r'\bdepression\b', r'\banxiety\b', r'\bPTSD\b',
        r'\bautoimmune\b', r'\brheumatoid\b', r'\blupus\b', r'\bCrohn\b', r'\bIBD\b',
        r'\binfectious\b', r'\bviral\b', r'\bbacterial\b', r'\bfungal\b', r'\bparasitic\b'
    ]
    
    genes = set()
    diseases = set()
    
    # Extract genes
    for pattern in gene_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        genes.update(matches)
    
    # Extract diseases
    for pattern in disease_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        diseases.update(matches)
    
    return list(genes), list(diseases)

def populate_neo4j_all_data():
    """Populate Neo4j with ALL real gene data from ChromaDB"""
    
    print("Populating Neo4j with ALL real gene data...")
    print("=" * 60)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'synthetica_password'))
    
    try:
        with driver.session() as session:
            # Clear existing data
            print("Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Connect to ChromaDB to get ALL data
            print("Extracting ALL data from ChromaDB...")
            db_path = "app/ml_pipeline/chroma_db_gene_mvp_new/chroma.sqlite3"
            
            if not os.path.exists(db_path):
                print(f"ChromaDB not found at {db_path}")
                return False
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get ALL documents with their metadata
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
            """)
            
            documents = cursor.fetchall()
            print(f"Found {len(documents)} documents to process")
            
            # Counters for statistics
            gene_counter = Counter()
            disease_counter = Counter()
            document_count = 0
            
            # Process documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                for doc_id, document, source, chunk_index in batch:
                    if document:
                        # Create document node
                        session.run("""
                            CREATE (d:Document {
                                id: $doc_id,
                                content: $document,
                                source: $source,
                                chunk_index: $chunk_index,
                                length: $length
                            })
                        """, 
                        doc_id=doc_id, 
                        document=document, 
                        source=source, 
                        chunk_index=chunk_index,
                        length=len(document)
                        )
                        
                        # Extract genes and diseases
                        genes, diseases = extract_genes_and_diseases(document)
                        
                        # Create gene nodes and relationships
                        for gene in genes:
                            gene_counter[gene.upper()] += 1
                            
                            session.run("""
                                MERGE (g:Gene {name: $gene_name})
                                ON CREATE SET g.type = 'Gene', g.count = 1
                                ON MATCH SET g.count = g.count + 1
                            """, gene_name=gene.upper())
                            
                            session.run("""
                                MATCH (d:Document {id: $doc_id})
                                MATCH (g:Gene {name: $gene_name})
                                CREATE (d)-[:MENTIONS]->(g)
                            """, doc_id=doc_id, gene_name=gene.upper())
                        
                        # Create disease nodes and relationships
                        for disease in diseases:
                            disease_counter[disease.lower()] += 1
                            
                            session.run("""
                                MERGE (disease:Disease {name: $disease_name})
                                ON CREATE SET disease.type = 'Disease', disease.count = 1
                                ON MATCH SET disease.count = disease.count + 1
                            """, disease_name=disease.lower())
                            
                            session.run("""
                                MATCH (d:Document {id: $doc_id})
                                MATCH (disease:Disease {name: $disease_name})
                                CREATE (d)-[:DISCUSSES]->(disease)
                            """, doc_id=doc_id, disease_name=disease.lower())
                        
                        document_count += 1
                
                print(f"Processed {min(i+batch_size, len(documents))}/{len(documents)} documents")
            
            conn.close()
            
            # Get final statistics
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
            stats = result.data()
            
            print("\nNeo4j populated with ALL real data:")
            for stat in stats:
                print(f"  {stat['labels']}: {stat['count']} nodes")
            
            print(f"\nTop 10 Genes found:")
            for gene, count in gene_counter.most_common(10):
                print(f"  {gene}: {count} mentions")
            
            print(f"\nTop 10 Diseases found:")
            for disease, count in disease_counter.most_common(10):
                print(f"  {disease}: {count} mentions")
            
            print("\n" + "=" * 60)
            print("Neo4j successfully populated with ALL real gene data!")
            return True
            
    except Exception as e:
        print(f"Error populating Neo4j: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        driver.close()

if __name__ == "__main__":
    populate_neo4j_all_data()
