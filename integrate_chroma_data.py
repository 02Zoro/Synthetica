#!/usr/bin/env python3
"""
Script to integrate ChromaDB gene data with the ML pipeline
"""
import sqlite3
import os
import json
import chromadb
from pathlib import Path

def integrate_chroma_data():
    """Integrate the existing ChromaDB data with the ML pipeline"""
    
    print("Integrating ChromaDB gene data with ML pipeline...")
    print("=" * 60)
    
    try:
        # Initialize ChromaDB client with the existing database
        client = chromadb.PersistentClient(path="app/ml_pipeline/chroma_db_gene_mvp")
        
        # Get the collection
        collection = client.get_collection("scientific_abstract_chunks")
        
        print(f"Collection: {collection.name}")
        print(f"Document count: {collection.count()}")
        
        # Test a simple query to see if the data is accessible
        print("\nTesting data access...")
        
        # Get a sample of documents
        sample = collection.get(limit=3)
        
        print("Sample documents:")
        for i, (doc_id, doc_text) in enumerate(zip(sample['ids'], sample['documents'])):
            print(f"\n{i+1}. ID: {doc_id}")
            print(f"   Text: {doc_text[:200]}{'...' if len(doc_text) > 200 else ''}")
            
            # Show metadata if available
            if 'metadatas' in sample and sample['metadatas'] and sample['metadatas'][i]:
                metadata = sample['metadatas'][i]
                print(f"   Metadata: {metadata}")
        
        # Test semantic search
        print("\nTesting semantic search...")
        query_results = collection.query(
            query_texts=["gene expression regulation"],
            n_results=2
        )
        
        print("Search results for 'gene expression regulation':")
        for i, (doc_id, doc_text, distance) in enumerate(zip(
            query_results['ids'][0], 
            query_results['documents'][0], 
            query_results['distances'][0]
        )):
            print(f"\n{i+1}. ID: {doc_id}")
            print(f"   Distance: {distance:.4f}")
            print(f"   Text: {doc_text[:200]}{'...' if len(doc_text) > 200 else ''}")
        
        print("\n" + "=" * 60)
        print("ChromaDB integration successful!")
        print("This data can now be used for hypothesis generation!")
        
        return True
        
    except Exception as e:
        print(f"Error integrating ChromaDB data: {e}")
        print("The database might be corrupted or incompatible.")
        return False

if __name__ == "__main__":
    integrate_chroma_data()

