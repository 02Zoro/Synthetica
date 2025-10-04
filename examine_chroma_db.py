#!/usr/bin/env python3
"""
Script to examine the ChromaDB database and understand its contents
"""
import chromadb
import os
import json
from pathlib import Path

def examine_chroma_db():
    """Examine the ChromaDB database contents"""
    
    # Path to the ChromaDB database
    db_path = "app/ml_pipeline/chroma_db_gene_mvp"
    
    print(f"Examining ChromaDB database at: {db_path}")
    print("=" * 60)
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=db_path)
        
        # List all collections
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        
        for collection in collections:
            print(f"\nCollection: {collection.name}")
            print(f"   ID: {collection.id}")
            
            # Get collection info
            try:
                count = collection.count()
                print(f"   Document count: {count}")
                
                if count > 0:
                    # Get a sample of documents
                    sample_size = min(5, count)
                    sample = collection.get(limit=sample_size)
                    
                    print(f"\n   Sample documents (first {sample_size}):")
                    for i, (doc_id, doc_text) in enumerate(zip(sample['ids'], sample['documents'])):
                        print(f"   {i+1}. ID: {doc_id}")
                        print(f"      Text: {doc_text[:200]}{'...' if len(doc_text) > 200 else ''}")
                        
                        # Show metadata if available
                        if 'metadatas' in sample and sample['metadatas']:
                            metadata = sample['metadatas'][i]
                            print(f"      Metadata: {metadata}")
                        print()
                
                # Get embeddings info
                if count > 0:
                    sample_embeddings = collection.get(limit=1, include=['embeddings'])
                    if 'embeddings' in sample_embeddings and sample_embeddings['embeddings']:
                        embedding_dim = len(sample_embeddings['embeddings'][0])
                        print(f"   Embedding dimension: {embedding_dim}")
                
            except Exception as e:
                print(f"   Error accessing collection: {e}")
        
        print("\n" + "=" * 60)
        print("ChromaDB examination complete!")
        
        return collections
        
    except Exception as e:
        print(f"❌ Error examining ChromaDB: {e}")
        return None

if __name__ == "__main__":
    examine_chroma_db()
