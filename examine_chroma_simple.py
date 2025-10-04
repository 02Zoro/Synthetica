#!/usr/bin/env python3
"""
Simple script to examine ChromaDB database structure
"""
import sqlite3
import os
import json
from pathlib import Path

def examine_chroma_simple():
    """Examine the ChromaDB database using SQLite directly"""
    
    # Path to the ChromaDB database
    db_path = "app/ml_pipeline/chroma_db_gene_mvp/chroma.sqlite3"
    
    print(f"Examining ChromaDB SQLite database at: {db_path}")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        print("\n" + "=" * 40)
        
        # Examine each table
        for table_name in tables:
            table = table_name[0]
            print(f"\nTable: {table}")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            print("  Columns:")
            for col in columns:
                print(f"    {col[1]} ({col[2]})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"  Row count: {count}")
            
            # Get sample data
            if count > 0:
                cursor.execute(f"SELECT * FROM {table} LIMIT 3;")
                sample_rows = cursor.fetchall()
                print("  Sample data:")
                for i, row in enumerate(sample_rows):
                    print(f"    Row {i+1}: {row[:100]}{'...' if len(str(row)) > 100 else ''}")
        
        conn.close()
        print("\n" + "=" * 60)
        print("Database examination complete!")
        
    except Exception as e:
        print(f"Error examining database: {e}")

if __name__ == "__main__":
    examine_chroma_simple()

