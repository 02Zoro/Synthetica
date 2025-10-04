#!/usr/bin/env python3
"""
Script to check Neo4j database statistics
"""
from neo4j import GraphDatabase

def check_neo4j_stats():
    """Check Neo4j database statistics"""
    
    print("Checking Neo4j database statistics...")
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
            
            print(f'Neo4j Database Statistics:')
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
            
            # Get sample data
            print()
            print('Sample Data:')
            result = session.run('MATCH (n) RETURN n LIMIT 3')
            samples = result.data()
            for i, sample in enumerate(samples):
                print(f'  Sample {i+1}: {sample}')
                
    except Exception as e:
        print(f'Error connecting to Neo4j: {e}')
    finally:
        driver.close()

if __name__ == "__main__":
    check_neo4j_stats()
