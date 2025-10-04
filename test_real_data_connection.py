#!/usr/bin/env python3
"""
Test script to verify real data connection
"""
from neo4j import GraphDatabase
import pandas as pd

def test_real_data_connection():
    """Test connection to real Neo4j data"""
    
    print("Testing Real Data Connection...")
    print("=" * 50)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'synthetica_password'))
    
    try:
        with driver.session() as session:
            # Get total node count
            result = session.run('MATCH (n) RETURN count(n) as total_nodes')
            total_nodes = result.single()['total_nodes']
            
            # Get total relationship count
            result = session.run('MATCH ()-[r]->() RETURN count(r) as total_edges')
            total_edges = result.single()['total_edges']
            
            # Get node counts by type
            result = session.run('MATCH (n) RETURN labels(n) as labels, count(n) as count ORDER BY count DESC')
            node_types = result.data()
            
            # Get relationship counts by type
            result = session.run('MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC')
            edge_types = result.data()
            
            print(f'Neo4j Connection: SUCCESS')
            print(f'Total Nodes: {total_nodes}')
            print(f'Total Edges: {total_edges}')
            print()
            print('Node Types:')
            for node_type in node_types:
                print(f'  {node_type["labels"]}: {node_type["count"]}')
            print()
            print('Edge Types:')
            for edge_type in edge_types:
                print(f'  {edge_type["rel_type"]}: {edge_type["count"]}')
            
            # Test if we can get gene names
            result = session.run('MATCH (g:Gene) RETURN g.name as gene_name LIMIT 5')
            genes = [record['gene_name'] for record in result]
            print(f'\nSample Genes: {genes}')
            
            # Test if we can get disease names
            result = session.run('MATCH (d:Disease) RETURN d.name as disease_name LIMIT 5')
            diseases = [record['disease_name'] for record in result]
            print(f'Sample Diseases: {diseases}')
            
            print(f'\nReal Data Connection: SUCCESS')
            print(f'Your system has {total_nodes} nodes and {total_edges} edges')
            return True
            
    except Exception as e:
        print(f'Neo4j Connection Failed: {e}')
        return False
    finally:
        driver.close()

if __name__ == "__main__":
    test_real_data_connection()
