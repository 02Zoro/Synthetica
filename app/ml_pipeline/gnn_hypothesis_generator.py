import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit

# VDB & Utility Imports
from neo4j import GraphDatabase
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
import re
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Neo4j Database Configuration
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

# ChromaDB (Vector Database) Configuration - Must match indexing script
VDB_PATH = "./chroma_db_gene_mvp"
VDB_COLLECTION_NAME = "scientific_abstract_chunks"
VDB_MODEL_NAME = 'all-MiniLM-L6-v2' 

# Graph Schema (Must match the KG builder)
NODE_LABELS = ['Researcher', 'Concept', 'Location', 'Organization'] 
RELATION_TYPES = ['REGULATES', 'ASSOCIATED_WITH', 'TREATS']

# GNN Training Parameters
EMBEDDING_DIM = 32
HIDDEN_CHANNELS = 64
EPOCHS = 50

# ==============================================================================
# 2. DATA EXTRACTION AND TRANSFORMATION (UNCHANGED)
# ==============================================================================

def get_neo4j_data(driver):
    """Executes Cypher queries to extract nodes and relationships."""
    
    NODE_QUERY = """
    MATCH (n)
    WHERE size(labels(n)) > 0 AND NOT 'Resource' IN labels(n) AND NOT 'Temporal' IN labels(n)
    UNWIND labels(n) AS label
    WITH DISTINCT n.name AS name, head(labels(n)) AS label
    RETURN name, label
    """
    
    EDGE_QUERY = f"""
    MATCH (a)-[r]->(b)
    WHERE type(r) IN {RELATION_TYPES}
    RETURN a.name AS source_name, 
           type(r) AS rel_type, 
           b.name AS target_name
    """
    
    with driver.session() as session:
        node_records, _, _ = driver.execute_query(NODE_QUERY, database_="neo4j")
        df_nodes = pd.DataFrame([dict(r) for r in node_records])
        
        edge_records, _, _ = driver.execute_query(EDGE_QUERY, database_="neo4j")
        df_edges = pd.DataFrame([dict(r) for r in edge_records])

    return df_nodes, df_edges

def transform_to_pyg(df_nodes, df_edges):
    """Maps strings to integers and constructs PyTorch Geometric Data object."""
    
    all_names = pd.concat([df_nodes['name'], df_edges['source_name'], df_edges['target_name']]).unique()
    name_to_id = {name: i for i, name in enumerate(all_names)}
    num_nodes = len(all_names)
    
    label_to_int = {label: i for i, label in enumerate(NODE_LABELS)}
    num_labels = len(NODE_LABELS)

    x = torch.zeros((num_nodes, num_labels), dtype=torch.float)
    for index, row in df_nodes.iterrows():
        node_id = name_to_id.get(row['name'])
        label_int = label_to_int.get(row['label'])
        if node_id is not None and label_int is not None:
            x[node_id, label_int] = 1.0 
    
    rel_type_to_int = {rel_type: i for i, rel_type in enumerate(RELATION_TYPES)}
    
    source_nodes, target_nodes, edge_types = [], [], []

    for _, row in df_edges.iterrows():
        src_id = name_to_id.get(row['source_name'])
        tgt_id = name_to_id.get(row['target_name'])
        rel_int = rel_type_to_int.get(row['rel_type'])

        if src_id is not None and tgt_id is not None and rel_int is not None:
            source_nodes.append(src_id)
            target_nodes.append(tgt_id)
            edge_types.append(rel_int)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = F.one_hot(torch.tensor(edge_types), num_classes=len(RELATION_TYPES)).float()
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    
    data.name_to_id = name_to_id
    data.id_to_name = {v: k for k, v in name_to_id.items()}
    
    return data

# ==============================================================================
# 3. GNN MODEL DEFINITION
# ==============================================================================

class GNNEncoder(nn.Module):
    """GNN Encoder: Learns node embeddings (Z) using GCN layers."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x 

class LinkDecoder(nn.Module):
    """MLP Decoder: Predicts link probability from concatenated embeddings."""
    def __init__(self, in_channels):
        super(LinkDecoder, self).__init__()
        self.lin = nn.Linear(in_channels * 2, 1)

    def forward(self, z, edge_label_index):
        source_index = edge_label_index[0]
        target_index = edge_label_index[1]
        
        concat_z = torch.cat([z[source_index], z[target_index]], dim=-1)
        score = self.lin(concat_z)
        
        return score.squeeze(-1)

# ==============================================================================
# 4. TRAINING AND HYPOTHESIS GENERATION FUNCTIONS
# ==============================================================================

def train_gnn(data):
    """Runs the GNN training loop and returns the trained models."""
    
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1, is_undirected=False, neg_sampling_ratio=1.0
    )
    train_data, _, _ = transform(data) 
    
    encoder = GNNEncoder(data.num_node_features, HIDDEN_CHANNELS, EMBEDDING_DIM)
    decoder = LinkDecoder(in_channels=EMBEDDING_DIM)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training Loop
    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        
        z = encoder(train_data.x, train_data.edge_index)
        scores = decoder(z, train_data.edge_label_index)
        loss = criterion(scores, train_data.edge_label.float())
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:02d}, Loss: {loss.item():.4f}")

    return encoder, decoder


def get_vdb_client():
    """Initializes and returns the ChromaDB collection object."""
    try:
        db_client = PersistentClient(path=VDB_PATH)
        sbert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=VDB_MODEL_NAME
        )
        collection = db_client.get_collection(
            name=VDB_COLLECTION_NAME,
            embedding_function=sbert_ef
        )
        return collection
    except Exception as e:
        print(f"\n❌ VDB CONNECTION ERROR: Ensure indexing ran successfully. Error: {e}")
        return None

def ground_hypothesis(vdb_collection, source_entity, target_entity, max_results=3):
    """Performs a semantic search for context supporting the predicted link."""
    
    # Construct a high-precision query using both entities
    query_text = (
        f"Evidence that {source_entity} and {target_entity} are associated, "
        f"interact, or share a common regulatory pathway."
    )
    
    results = vdb_collection.query(
        query_texts=[query_text],
        n_results=max_results,
        include=['documents', 'distances']
    )
    
    snippets = []
    if results and results.get('documents') and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i]
            # Only include highly relevant results (Distance < 0.5 is high similarity)
            if distance < 0.5: 
                snippets.append(f"[Sim: {1 - distance:.3f}]: {doc.strip()}")
    
    return snippets


def generate_hypotheses(encoder, decoder, data, df_nodes, df_edges, vdb_collection, num_hypotheses=10):
    """
    Uses the trained GNN to predict novel links between unlinked nodes,
    and grounds the results using the Semantic Search VDB.
    """
    encoder.eval()
    decoder.eval()
    
    # 1. Generate final node embeddings (Z)
    with torch.no_grad():
        z = encoder(data.x, data.edge_index)

    # --- Pre-calculate Known Edges for Filtering ---
    known_edges = set()
    for src, tgt in data.edge_index.t().tolist():
        known_edges.add((src, tgt))

    # 2. Score a large sample of candidates (20,000 pairs)
    num_candidates = 20000 
    candidate_sources = torch.randint(0, data.num_nodes, (num_candidates,))
    candidate_targets = torch.randint(0, data.num_nodes, (num_candidates,))
    candidate_edge_index = torch.stack([candidate_sources, candidate_targets])
    
    scores = decoder(z, candidate_edge_index)
    
    # 3. Filter and Rank Hypotheses
    scored_candidates = []
    
    # Sort scores to prioritize candidates that are more likely to be true links
    top_scores, top_indices = torch.topk(scores, k=num_candidates)
    
    for i in range(num_candidates):
        score = top_scores[i].item()
        index = top_indices[i].item()
        
        src_id = candidate_edge_index[0, index].item()
        tgt_id = candidate_edge_index[1, index].item()
        
        source_name = data.id_to_name.get(src_id, f"Node_{src_id}")
        target_name = data.id_to_name.get(tgt_id, f"Node_{tgt_id}")
        
        # --- NOISE FILTERING ---
        if "##" in source_name or "##" in target_name or len(source_name) <= 2 or len(target_name) <= 2:
            continue

        # --- NOVELTY FILTERING ---
        if (src_id, tgt_id) in known_edges:
            continue 
        
        # --- GROUNDING STEP (CRITICAL) ---
        supporting_evidence = ground_hypothesis(
            vdb_collection, 
            source_name, 
            target_name
        )
        
        # Only proceed if strong evidence is found
        if not supporting_evidence:
            continue
            
        scored_candidates.append({
            "Score": torch.sigmoid(torch.tensor(score)).item(),
            "Source_Entity": source_name,
            "Target_Entity": target_name,
            "Relationship": "PREDICTED_ASSOCIATION",
            "Evidence_Snippets": supporting_evidence
        })
        
        if len(scored_candidates) >= num_hypotheses:
            break # Stop once we collect enough grounded hypotheses

    return scored_candidates


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # --- 1. Connect and Extract DataFrames ---
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        df_nodes, df_edges = get_neo4j_data(driver)
        driver.close()
    except Exception as e:
        print(f"\n❌ FATAL CONNECTION ERROR: Could not connect to Neo4j or execute query. Error: {e}")
        exit()
    
    if df_nodes.empty or df_edges.empty:
        print("\n❌ FATAL DATA ERROR: Node or Edge DataFrames are empty. Exiting.")
        exit()

    # --- 2. Transform to PyG Format ---
    graph_data = transform_to_pyg(df_nodes, df_edges)

    # --- 3. Run GNN Training ---
    print("\n--- Starting GNN Training ---")
    trained_encoder, trained_decoder = train_gnn(graph_data)
    
    # --- 4. Initialize VDB Client ---
    vdb_collection = get_vdb_client()
    if vdb_collection is None:
        exit()

    # --- 5. Generate and Report Grounded Hypotheses ---
    print("\n--- 5. Generating and Grounding Novel Hypotheses ---")
    
    hypotheses = generate_hypotheses(trained_encoder, trained_decoder, graph_data, df_nodes, df_edges, vdb_collection)
    
    print("\n✅ TOP 10 NOVEL HYPOTHESES (Grounded Predictions):")
    print("-" * 80)
    
    if not hypotheses:
        print("No sufficiently novel and grounded hypotheses were found. Try increasing sample size or checking VDB index quality.")
    else:
        for i, h in enumerate(hypotheses):
            print(f"[{i+1:02d}] {h['Source_Entity']:<25} --[PREDICTED_ASSOCIATION]--> {h['Target_Entity']:<25} (Prob: {h['Score']:.4f})")
            print("    EVIDENCE:")
            for snippet in h['Evidence_Snippets']:
                 print(f"    - {snippet}")

    print("-" * 80)
    print("\n==================================================================")
    print("✅ PROJECT MVP COMPLETE: AI Hypothesis Generation Pipeline Finished.")
    print("==================================================================")