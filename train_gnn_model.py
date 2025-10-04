#!/usr/bin/env python3
"""
Train GNN model on your real biomedical data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime

class GNNEncoder(nn.Module):
    """GNN Encoder for node embeddings"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class LinkDecoder(nn.Module):
    """MLP Decoder for link prediction"""
    def __init__(self, in_channels):
        super(LinkDecoder, self).__init__()
        self.lin1 = nn.Linear(in_channels * 2, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, z, edge_label_index):
        source_index = edge_label_index[0]
        target_index = edge_label_index[1]
        
        concat_z = torch.cat([z[source_index], z[target_index]], dim=-1)
        x = self.lin1(concat_z).relu()
        x = self.dropout(x)
        x = self.lin2(x).relu()
        x = self.dropout(x)
        x = self.lin3(x)
        
        return x.squeeze(-1)

def load_data_from_neo4j():
    """Load your real data from Neo4j"""
    print("Loading data from Neo4j...")
    
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'synthetica_password'))
    
    try:
        with driver.session() as session:
            # Get nodes
            result = session.run("MATCH (n) RETURN n, labels(n) as labels")
            nodes_data = []
            for record in result:
                node = record["n"]
                labels = record["labels"]
                
                if "Document" in labels:
                    nodes_data.append({
                        "name": node.get("id", "unknown"),
                        "labels": labels,
                        "type": "Document",
                        "content": node.get("content", "")[:100] + "..." if len(node.get("content", "")) > 100 else node.get("content", "")
                    })
                elif "Gene" in labels:
                    nodes_data.append({
                        "name": node.get("name", "unknown"),
                        "labels": labels,
                        "type": "Gene",
                        "count": node.get("count", 0)
                    })
                elif "Disease" in labels:
                    nodes_data.append({
                        "name": node.get("name", "unknown"),
                        "labels": labels,
                        "type": "Disease",
                        "count": node.get("count", 0)
                    })
            
            df_nodes = pd.DataFrame(nodes_data)
            
            # Get edges
            result = session.run("MATCH (a)-[r]->(b) RETURN a, type(r) as rel_type, b, labels(a) as source_labels, labels(b) as target_labels")
            edges_data = []
            for record in result:
                source_node = record["a"]
                target_node = record["b"]
                rel_type = record["rel_type"]
                source_labels = record["source_labels"]
                target_labels = record["target_labels"]
                
                # Get source name based on node type
                if "Document" in source_labels:
                    source_name = source_node.get("id", "unknown")
                else:
                    source_name = source_node.get("name", "unknown")
                    
                # Get target name based on node type
                if "Document" in target_labels:
                    target_name = target_node.get("id", "unknown")
                else:
                    target_name = target_node.get("name", "unknown")
                
                edges_data.append({
                    "source_name": source_name,
                    "rel_type": rel_type,
                    "target_name": target_name,
                    "source_type": source_labels[0] if source_labels else "Unknown",
                    "target_type": target_labels[0] if target_labels else "Unknown"
                })
            
            df_edges = pd.DataFrame(edges_data)
            
            print(f"Loaded {len(df_nodes)} nodes and {len(df_edges)} edges")
            return df_nodes, df_edges
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    finally:
        driver.close()

def create_graph_data(df_nodes, df_edges):
    """Create PyTorch Geometric data from your real data"""
    print("Creating graph data...")
    
    if df_nodes.empty or df_edges.empty:
        return None
    
    # Create node mappings
    unique_nodes = list(set(df_nodes['name'].tolist()))
    name_to_id = {name: i for i, name in enumerate(unique_nodes)}
    id_to_name = {i: name for name, i in name_to_id.items()}
    
    # Create node features (enhanced features)
    num_nodes = len(unique_nodes)
    
    # Create more sophisticated node features
    node_features = []
    for node_name in unique_nodes:
        node_data = df_nodes[df_nodes['name'] == node_name].iloc[0]
        
        # Create feature vector based on node type and properties
        if node_data['type'] == 'Gene':
            # Gene features: one-hot encoding + count
            features = [1, 0, 0, node_data.get('count', 0) / 100.0]  # Normalize count
        elif node_data['type'] == 'Disease':
            # Disease features: one-hot encoding + count
            features = [0, 1, 0, node_data.get('count', 0) / 100.0]  # Normalize count
        else:  # Document
            # Document features: one-hot encoding + content length
            content_len = len(node_data.get('content', ''))
            features = [0, 0, 1, min(content_len / 1000.0, 1.0)]  # Normalize content length
        
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create edge indices and attributes
    source_nodes = []
    target_nodes = []
    edge_types = []
    
    rel_types = df_edges['rel_type'].unique()
    rel_type_to_int = {rel: i for i, rel in enumerate(rel_types)}
    
    for _, row in df_edges.iterrows():
        src_id = name_to_id.get(row['source_name'])
        tgt_id = name_to_id.get(row['target_name'])
        rel_int = rel_type_to_int.get(row['rel_type'])

        if src_id is not None and tgt_id is not None and rel_int is not None:
            source_nodes.append(src_id)
            target_nodes.append(tgt_id)
            edge_types.append(rel_int)

    if not source_nodes:
        return None
        
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = F.one_hot(torch.tensor(edge_types), num_classes=len(rel_types)).float()
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    data.name_to_id = name_to_id
    data.id_to_name = id_to_name
    
    print(f"Created graph with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
    return data

def create_training_data(data):
    """Create training data for link prediction"""
    print("Creating training data...")
    
    # Get existing edges
    existing_edges = data.edge_index.t()
    
    # Create negative samples (non-existing edges)
    num_neg_samples = min(len(existing_edges) * 2, 10000)  # Limit negative samples
    neg_edges = []
    
    for _ in range(num_neg_samples):
        src = torch.randint(0, data.num_nodes, (1,)).item()
        tgt = torch.randint(0, data.num_nodes, (1,)).item()
        
        # Check if edge doesn't exist
        edge_exists = False
        for existing_edge in existing_edges:
            if (existing_edge[0] == src and existing_edge[1] == tgt) or \
               (existing_edge[0] == tgt and existing_edge[1] == src):
                edge_exists = True
                break
        
        if not edge_exists:
            neg_edges.append([src, tgt])
    
    # Combine positive and negative edges
    pos_edges = existing_edges.tolist()
    all_edges = pos_edges + neg_edges
    
    # Create labels (1 for positive, 0 for negative)
    labels = [1] * len(pos_edges) + [0] * len(neg_edges)
    
    # Convert to tensors
    edge_label_index = torch.tensor(all_edges).t()
    edge_label = torch.tensor(labels, dtype=torch.float)
    
    print(f"Created {len(pos_edges)} positive and {len(neg_edges)} negative samples")
    return edge_label_index, edge_label

def train_model(data, edge_label_index, edge_label, epochs=500):
    """Train the GNN model with extended training"""
    print(f"Training GNN model for {epochs} epochs...")
    
    # Create model with better architecture
    encoder = GNNEncoder(
        in_channels=data.x.size(1),
        hidden_channels=128,  # Increased hidden size
        out_channels=64       # Increased output size
    )
    
    decoder = LinkDecoder(in_channels=64)  # Updated for new output size
    
    # Better optimizer with learning rate scheduling
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.005,  # Lower learning rate for stability
        weight_decay=1e-3
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    # Training loop
    encoder.train()
    decoder.train()
    
    best_loss = float('inf')
    patience = 50  # Increased patience
    patience_counter = 0
    best_encoder = None
    best_decoder = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        z = encoder(data.x, data.edge_index)
        pred = decoder(z, edge_label_index)
        
        # Compute loss with regularization
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)
        
        # Add L2 regularization
        l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in encoder.parameters())
        loss += l2_reg
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss)
        
        # Early stopping with model saving
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save best model
            best_encoder = GNNEncoder(
                in_channels=data.x.size(1),
                hidden_channels=128,
                out_channels=64
            )
            best_decoder = LinkDecoder(in_channels=64)
            best_encoder.load_state_dict(encoder.state_dict())
            best_decoder.load_state_dict(decoder.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    
    # Return best model
    if best_encoder is not None and best_decoder is not None:
        return best_encoder, best_decoder
    else:
        return encoder, decoder

def save_model(encoder, decoder, data, model_dir="app/ml_pipeline/models"):
    """Save the trained model"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model weights
    torch.save(encoder.state_dict(), os.path.join(model_dir, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(model_dir, "decoder.pth"))
    
    # Save model metadata
    metadata = {
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.size(1),
        "node_features": data.x.size(1),
        "hidden_channels": 64,
        "out_channels": 32,
        "trained_at": datetime.now().isoformat(),
        "model_type": "GNN_LinkPrediction"
    }
    
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_dir}")

def main():
    """Main training function"""
    print("Starting GNN model training on your real data...")
    print("=" * 60)
    
    # Load data
    df_nodes, df_edges = load_data_from_neo4j()
    if df_nodes is None or df_edges is None:
        print("Failed to load data from Neo4j")
        return
    
    # Create graph data
    data = create_graph_data(df_nodes, df_edges)
    if data is None:
        print("Failed to create graph data")
        return
    
    # Create training data
    edge_label_index, edge_label = create_training_data(data)
    
    # Train model with extended training
    encoder, decoder = train_model(data, edge_label_index, edge_label, epochs=1000)
    
    # Save model
    save_model(encoder, decoder, data)
    
    print("=" * 60)
    print("GNN model training completed!")
    print("Your model is now trained on your real biomedical data.")
    print("The confidence scores should be much higher now!")

if __name__ == "__main__":
    main()
