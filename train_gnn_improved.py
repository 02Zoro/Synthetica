#!/usr/bin/env python3
"""
Improved GNN Training Script for Higher Confidence Scores
This script addresses the negative confidence score issue by:
1. Using focal loss instead of binary cross-entropy
2. Better data balancing
3. Improved model architecture
4. Better training strategy
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import json
from datetime import datetime
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedGNNEncoder(nn.Module):
    """Improved GNN Encoder with better architecture"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(ImprovedGNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        
    def forward(self, x, edge_index):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

class ImprovedLinkDecoder(nn.Module):
    """Improved Link Decoder with better architecture"""
    def __init__(self, in_channels, hidden_channels=64, dropout=0.3):
        super(ImprovedLinkDecoder, self).__init__()
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = nn.Linear(hidden_channels // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels // 2)
        
    def forward(self, z, edge_label_index):
        # Concatenate source and target node embeddings
        row, col = edge_label_index
        z = torch.cat([z[row], z[col]], dim=-1)
        
        z = F.relu(self.batch_norm1(self.lin1(z)))
        z = self.dropout(z)
        z = F.relu(self.batch_norm2(self.lin2(z)))
        z = self.dropout(z)
        z = self.lin3(z)
        return z.view(-1)

def load_data_from_neo4j():
    """Load data from Neo4j with improved query"""
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "synthetica_password"
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    with driver.session() as session:
        # Get nodes with better feature extraction
        nodes_query = """
        MATCH (n)
        RETURN n.id as id, n.name as name, n.content as content, 
               labels(n) as labels, n.count as count
        """
        nodes_result = session.run(nodes_query)
        nodes_data = []
        for record in nodes_result:
            nodes_data.append({
                'id': record['id'],
                'name': record['name'],
                'content': record['content'],
                'labels': record['labels'],
                'count': record['count']
            })
        
        # Get edges
        edges_query = """
        MATCH (a)-[r]->(b)
        RETURN a.name as source_name, b.name as target_name, 
               type(r) as rel_type, a.id as source_id, b.id as target_id
        """
        edges_result = session.run(edges_query)
        edges_data = []
        for record in edges_result:
            edges_data.append({
                'source_name': record['source_name'],
                'target_name': record['target_name'],
                'rel_type': record['rel_type'],
                'source_id': record['source_id'],
                'target_id': record['target_id']
            })
    
    driver.close()
    return pd.DataFrame(nodes_data), pd.DataFrame(edges_data)

def create_improved_graph_data(df_nodes, df_edges):
    """Create improved graph data with better features"""
    if df_nodes.empty or df_edges.empty:
        return None
    
    # Create node mappings
    unique_nodes = list(set(df_nodes['name'].tolist()))
    name_to_id = {name: i for i, name in enumerate(unique_nodes)}
    id_to_name = {i: name for name, i in name_to_id.items()}
    
    # Create better node features
    num_nodes = len(unique_nodes)
    
    # Feature 1: Node type (one-hot encoding)
    node_types = []
    for node in unique_nodes:
        node_rows = df_nodes[df_nodes['name'] == node]
        if len(node_rows) > 0:
            node_row = node_rows.iloc[0]
            labels = node_row.get('labels', [])
            if isinstance(labels, list) and 'Gene' in labels:
                node_types.append(0)  # Gene
            elif isinstance(labels, list) and 'Disease' in labels:
                node_types.append(1)  # Disease
            else:
                node_types.append(2)  # Document
        else:
            node_types.append(2)  # Default to Document
    
    # Feature 2: Node degree (will be calculated)
    node_degrees = [0] * num_nodes
    
    # Feature 3: Content length
    content_lengths = []
    for node in unique_nodes:
        node_rows = df_nodes[df_nodes['name'] == node]
        if len(node_rows) > 0:
            node_row = node_rows.iloc[0]
            content = node_row.get('content', '')
            content_lengths.append(len(str(content)) if content else 0)
        else:
            content_lengths.append(0)
    
    # Feature 4: Count (if available)
    counts = []
    for node in unique_nodes:
        node_rows = df_nodes[df_nodes['name'] == node]
        if len(node_rows) > 0:
            node_row = node_rows.iloc[0]
            counts.append(node_row.get('count', 0) if node_row.get('count') else 0)
        else:
            counts.append(0)
    
    # Normalize features
    max_content_len = max(content_lengths) if content_lengths and max(content_lengths) > 0 else 1
    max_count = max(counts) if counts and max(counts) > 0 else 1
    
    content_lengths = [x / max_content_len for x in content_lengths]
    counts = [x / max_count for x in counts]
    
    # Create edge indices and calculate degrees
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
            node_degrees[src_id] += 1
            node_degrees[tgt_id] += 1
    
    if not source_nodes:
        return None
    
    # Normalize degrees
    max_degree = max(node_degrees) if node_degrees else 1
    node_degrees = [x / max_degree for x in node_degrees]
    
    # Create feature matrix
    x = torch.tensor([
        [node_types[i], node_degrees[i], content_lengths[i], counts[i]]
        for i in range(num_nodes)
    ], dtype=torch.float)
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = F.one_hot(torch.tensor(edge_types), num_classes=len(rel_types)).float()
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    data.name_to_id = name_to_id
    data.id_to_name = id_to_name
    
    return data

def create_balanced_training_data(data, positive_ratio=0.7):
    """Create balanced training data with more positive examples"""
    num_nodes = data.num_nodes
    
    # Get all positive edges
    positive_edges = data.edge_index.t().tolist()
    num_positive = len(positive_edges)
    
    # Create negative edges (non-existent connections)
    negative_edges = []
    max_negative = int(num_positive * (1 - positive_ratio) / positive_ratio)
    
    existing_edges = set(tuple(edge) for edge in positive_edges)
    
    while len(negative_edges) < max_negative:
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        
        if src != dst and (src, dst) not in existing_edges and (dst, src) not in existing_edges:
            negative_edges.append([src, dst])
    
    # Combine positive and negative edges
    all_edges = positive_edges + negative_edges
    all_labels = [1] * num_positive + [0] * len(negative_edges)
    
    # Shuffle the data
    combined = list(zip(all_edges, all_labels))
    random.shuffle(combined)
    all_edges, all_labels = zip(*combined)
    
    edge_label_index = torch.tensor(all_edges, dtype=torch.long).t()
    edge_label = torch.tensor(all_labels, dtype=torch.float)
    
    return edge_label_index, edge_label

def train_improved_model(data, edge_label_index, edge_label, epochs=2000):
    """Train the improved model with better parameters"""
    print(f"Training improved GNN model for {epochs} epochs...")
    
    # Create improved models
    encoder = ImprovedGNNEncoder(
        in_channels=data.x.size(1),
        hidden_channels=256,  # Increased capacity
        out_channels=128,      # Increased capacity
        dropout=0.3
    )
    
    decoder = ImprovedLinkDecoder(
        in_channels=128,
        hidden_channels=128,
        dropout=0.3
    )
    
    # Use AdamW optimizer with better parameters
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001,  # Lower learning rate for stability
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Use focal loss for better handling of class imbalance
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=30
    )
    
    # Training loop
    encoder.train()
    decoder.train()
    
    best_loss = float('inf')
    patience = 100  # Increased patience
    patience_counter = 0
    best_encoder = None
    best_decoder = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        z = encoder(data.x, data.edge_index)
        pred = decoder(z, edge_label_index)
        
        # Compute loss with focal loss
        loss = criterion(pred, edge_label)
        
        # Add L2 regularization
        l2_reg = 0.0001 * sum(p.pow(2.0).sum() for p in encoder.parameters())
        l2_reg += 0.0001 * sum(p.pow(2.0).sum() for p in decoder.parameters())
        loss += l2_reg
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 
            max_norm=1.0
        )
        
        optimizer.step()
        scheduler.step(loss)
        
        # Early stopping with model saving
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save best model
            best_encoder = ImprovedGNNEncoder(
                in_channels=data.x.size(1),
                hidden_channels=256,
                out_channels=128,
                dropout=0.3
            )
            best_decoder = ImprovedLinkDecoder(
                in_channels=128,
                hidden_channels=128,
                dropout=0.3
            )
            best_encoder.load_state_dict(encoder.state_dict())
            best_decoder.load_state_dict(decoder.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    
    # Return best model
    if best_encoder is not None and best_decoder is not None:
        return best_encoder, best_decoder
    else:
        return encoder, decoder

def save_improved_model(encoder, decoder, data, model_dir="app/ml_pipeline/models"):
    """Save the improved trained model"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model weights
    torch.save(encoder.state_dict(), os.path.join(model_dir, "encoder_improved.pth"))
    torch.save(decoder.state_dict(), os.path.join(model_dir, "decoder_improved.pth"))
    
    # Save metadata
    metadata = {
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.size(1),
        "node_features": data.x.size(1),
        "hidden_channels": 256,
        "out_channels": 128,
        "trained_at": datetime.now().isoformat(),
        "model_type": "Improved_GNN_LinkPrediction",
        "architecture": "ImprovedGNNEncoder + ImprovedLinkDecoder",
        "loss_function": "FocalLoss",
        "optimizer": "AdamW",
        "dropout": 0.3
    }
    
    with open(os.path.join(model_dir, "metadata_improved.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Improved model saved to {model_dir}")

def main():
    """Main training function"""
    print("Starting IMPROVED GNN model training...")
    print("=" * 60)
    
    # Load data from Neo4j
    print("Loading data from Neo4j...")
    df_nodes, df_edges = load_data_from_neo4j()
    print(f"Loaded {len(df_nodes)} nodes and {len(df_edges)} edges")
    
    # Create improved graph data
    print("Creating improved graph data...")
    data = create_improved_graph_data(df_nodes, df_edges)
    if data is None:
        print("Failed to create graph data!")
        return
    
    print(f"Created graph with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
    
    # Create balanced training data
    print("Creating balanced training data...")
    edge_label_index, edge_label = create_balanced_training_data(data, positive_ratio=0.7)
    print(f"Created {edge_label.sum().item():.0f} positive and {(1-edge_label).sum().item():.0f} negative samples")
    
    # Train improved model
    encoder, decoder = train_improved_model(data, edge_label_index, edge_label, epochs=2000)
    
    # Save improved model
    save_improved_model(encoder, decoder, data)
    
    print("=" * 60)
    print("IMPROVED GNN model training completed!")
    print("This model should have much higher confidence scores!")
    print("The improved architecture and focal loss should address the negative confidence issue.")

if __name__ == "__main__":
    main()
