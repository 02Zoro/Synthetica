#!/usr/bin/env python3
"""
Script to extract and examine the ChromaDB gene data
"""
import sqlite3
import os
import json
from pathlib import Path

def extract_chroma_data():
    """Extract and examine the ChromaDB gene data"""
    
    # Path to the ChromaDB database
    db_path = "app/ml_pipeline/chroma_db_gene_mvp/chroma.sqlite3"
    
    print(f"Extracting ChromaDB gene data from: {db_path}")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get collection info
        cursor.execute("SELECT name, dimension FROM collections;")
        collections = cursor.fetchall()
        
        print("Collections found:")
        for collection in collections:
            print(f"  - {collection[0]} (dimension: {collection[1]})")
        
        # Get total document count
        cursor.execute("SELECT COUNT(*) FROM embeddings;")
        total_docs = cursor.fetchone()[0]
        print(f"\nTotal documents: {total_docs}")
        
        # Get sample documents with metadata
        print("\nSample documents:")
        cursor.execute("""
            SELECT e.embedding_id, em.string_value, em.key
            FROM embeddings e
            JOIN embedding_metadata em ON e.id = em.id
            WHERE em.key = 'chroma:document'
            LIMIT 5
        """)
        
        sample_docs = cursor.fetchall()
        for i, (doc_id, content, key) in enumerate(sample_docs):
            print(f"\n{i+1}. Document ID: {doc_id}")
            print(f"   Content: {content[:300]}{'...' if len(content) > 300 else ''}")
        
        # Get metadata statistics
        print("\nMetadata analysis:")
        cursor.execute("""
            SELECT key, COUNT(*) as count
            FROM embedding_metadata
            GROUP BY key
            ORDER BY count DESC
        """)
        
        metadata_stats = cursor.fetchall()
        for key, count in metadata_stats:
            print(f"  {key}: {count} entries")
        
        # Get sample metadata for a document
        print("\nSample metadata structure:")
        cursor.execute("""
            SELECT em.key, em.string_value, em.int_value, em.float_value
            FROM embedding_metadata em
            WHERE em.id = 1
        """)
        
        sample_metadata = cursor.fetchall()
        for key, str_val, int_val, float_val in sample_metadata:
            value = str_val or int_val or float_val
            print(f"  {key}: {value}")
        
        conn.close()
        
        print("\n" + "=" * 60)
        print("Data extraction complete!")
        print(f"\nThis ChromaDB contains {total_docs} scientific abstracts")
        print("with gene-related data that can be used for hypothesis generation!")
        
        return total_docs
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None

if __name__ == "__main__":
    extract_chroma_data()

