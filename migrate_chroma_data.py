#!/usr/bin/env python3
"""
Script to migrate data from the corrupted ChromaDB to a new working database
"""
import sqlite3
import os
import json
import chromadb
from pathlib import Path
import hashlib

def migrate_chroma_data():
    """Migrate data from corrupted ChromaDB to a new working database"""
    
    print("Migrating ChromaDB gene data to new database...")
    print("=" * 60)
    
    # Paths
    old_db_path = "app/ml_pipeline/chroma_db_gene_mvp/chroma.sqlite3"
    new_db_path = "app/ml_pipeline/chroma_db_gene_mvp_new"
    
    try:
        # Connect to the old SQLite database
        conn = sqlite3.connect(old_db_path)
        cursor = conn.cursor()
        
        # Get all documents with their metadata
        print("Extracting documents from old database...")
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
            ORDER BY e.id
        """)
        
        documents = cursor.fetchall()
        print(f"Found {len(documents)} documents to migrate")
        
        # Create new ChromaDB database
        print("Creating new ChromaDB database...")
        client = chromadb.PersistentClient(path=new_db_path)
        
        # Create collection
        collection = client.create_collection(
            name="scientific_abstract_chunks",
            metadata={"description": "Scientific abstracts with gene-related data"}
        )
        
        # Process documents in batches
        batch_size = 100
        total_docs = len(documents)
        
        print(f"Migrating {total_docs} documents in batches of {batch_size}...")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            
            # Prepare batch data
            ids = []
            texts = []
            metadatas = []
            
            for doc_id, document, source, chunk_index in batch:
                if document:  # Only process documents with content
                    ids.append(doc_id)
                    texts.append(document)
                    
                    metadata = {}
                    if source:
                        metadata['source'] = source
                    if chunk_index is not None:
                        metadata['chunk_index'] = chunk_index
                    
                    metadatas.append(metadata)
            
            # Add batch to collection
            if ids:
                collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
            
            print(f"Processed {min(i+batch_size, total_docs)}/{total_docs} documents")
        
        conn.close()
        
        # Verify the new database
        print("\nVerifying new database...")
        new_collection = client.get_collection("scientific_abstract_chunks")
        count = new_collection.count()
        print(f"New database contains {count} documents")
        
        # Test a query
        print("Testing semantic search...")
        results = new_collection.query(
            query_texts=["gene expression"],
            n_results=2
        )
        
        print("Sample search results:")
        for i, (doc_id, doc_text) in enumerate(zip(results['ids'][0], results['documents'][0])):
            print(f"\n{i+1}. ID: {doc_id}")
            print(f"   Text: {doc_text[:200]}{'...' if len(doc_text) > 200 else ''}")
        
        print("\n" + "=" * 60)
        print("Migration successful!")
        print(f"New database created at: {new_db_path}")
        print(f"Total documents migrated: {count}")
        
        return True
        
    except Exception as e:
        print(f"Error migrating data: {e}")
        return False

if __name__ == "__main__":
    migrate_chroma_data()

