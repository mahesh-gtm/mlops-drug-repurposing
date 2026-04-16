import torch
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
from src.utils.config import CONFIG
import os
import json
from pathlib import Path
import pickle
import numpy as np

def build_heterogeneous_graph():
    print("🔨 Building realistic heterogeneous knowledge graph from real GDSC data...")

    # === ROBUST DIRECTORY CREATION (fixes FileExistsError) ===
    processed_path = Path(CONFIG["processed_data_path"])
    
    if processed_path.exists() and not processed_path.is_dir():
        print(f"⚠️  Found a file instead of directory at {processed_path}. Removing it.")
        processed_path.unlink()

    processed_path.mkdir(parents=True, exist_ok=True)

    # Load real GDSC data
    df = pd.read_csv(f"{CONFIG['raw_data_path']}gdsc2_original.csv")

    G = nx.Graph()

    drugs = sorted(df['DRUG_NAME'].unique()[:40])
    cell_lines = sorted(df['CELL_LINE'].unique()[:40])

    cancer_mapping = {
        "A549": "Lung Cancer",
        "MCF7": "Breast Cancer",
        "HT29": "Colorectal Cancer",
        "PC3": "Prostate Cancer",
        "U251": "Glioma",
    }
    diseases = sorted(set(cancer_mapping.values()))

    # Add nodes with attributes
    for i, drug in enumerate(drugs):
        G.add_node(drug, type="drug", type_id=0, index=i)
    for i, cell in enumerate(cell_lines):
        G.add_node(cell, type="cell_line", type_id=1, index=i)
    for i, disease in enumerate(diseases):
        G.add_node(disease, type="disease", type_id=2, index=i)

    # === PRIMARY EDGES: Drug-Cell Interactions ===
    drug_cell_edges = {}
    for _, row in df.iterrows():
        if row['DRUG_NAME'] in drugs and row['CELL_LINE'] in cell_lines:
            ic50 = float(row['IC50'])
            auc = float(row['AUC'])
            # Lower IC50 = more potent, Higher AUC = better
            edge_strength = (1.0 / (1.0 + ic50)) * auc
            
            G.add_edge(row['DRUG_NAME'], row['CELL_LINE'],
                       weight=edge_strength,
                       ic50=ic50,
                       auc=auc,
                       relation="tested_on")
            
            key = (row['DRUG_NAME'], row['CELL_LINE'])
            drug_cell_edges[key] = edge_strength

    # === SECONDARY EDGES: Cell Line-Disease Connections ===
    for cell, disease in cancer_mapping.items():
        if cell in cell_lines:
            drug_edges = [(d, c, w) for d, c, w in 
                         [(u, v, G[u][v]['weight']) for u, v in G.edges() if v == cell]]
            if drug_edges:
                avg_weight = sum(w for _, _, w in drug_edges) / len(drug_edges)
                G.add_edge(cell, disease, weight=avg_weight, relation="models_disease")

    # === TERTIARY EDGES: Drug-Drug Similarity ===
    # Connect drugs that work well on similar cell lines
    for i, drug1 in enumerate(drugs):
        cells1 = set([v for u, v in G.edges() if u == drug1 and G.nodes[v].get('type') == 'cell_line'])
        
        for drug2 in drugs[i+1:]:
            cells2 = set([v for u, v in G.edges() if u == drug2 and G.nodes[v].get('type') == 'cell_line'])
            
            # Jaccard similarity
            if cells1 or cells2:
                similarity = len(cells1 & cells2) / len(cells1 | cells2)
                if similarity > 0.3:  # Only keep significant similarities
                    G.add_edge(drug1, drug2, weight=similarity, relation="similar_to")

    # === QUATERNARY EDGES: Disease-Disease Similarity ===
    # Connect related diseases (e.g., lung and brain cancers share some mechanisms)
    disease_list = list(diseases)
    for i, disease1 in enumerate(disease_list):
        for disease2 in disease_list[i+1:]:
            # Diseases that share cell types are somewhat similar
            cells1 = set([u for u, v in cancer_mapping.items() if v == disease1])
            cells2 = set([u for u, v in cancer_mapping.items() if v == disease2])
            
            # Add weak connection between all diseases
            base_sim = 0.2 + (0.3 if cells1 & cells2 else 0)
            G.add_edge(disease1, disease2, weight=base_sim, relation="related_to")

    print(f"✅ Graph built: {len(drugs)} drugs, {len(cell_lines)} cell lines, {len(diseases)} diseases")
    print(f"   Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")

    # === Create PyTorch Geometric Graph with Proper Node Features ===
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    
    # Create node features: [one_hot_type (3-dim), node_attributes (8-dim)]
    # Type encoding: drug=[1,0,0], cell=[0,1,0], disease=[0,0,1]
    num_nodes = len(node_mapping)
    x = torch.zeros(num_nodes, 16)  # 3 for type + 13 for attributes
    
    # Assign node features
    for node, idx in node_mapping.items():
        node_data = G.nodes[node]
        node_type = node_data.get('type_id', 0)
        
        # One-hot encode type
        x[idx, node_type] = 1.0
        
        # Add node-specific features
        if node_data.get('type') == 'drug':
            # Drug index (normalized)
            x[idx, 3] = node_data.get('index', 0) / len(drugs)
        elif node_data.get('type') == 'cell_line':
            # Cell line index (normalized)
            x[idx, 3] = node_data.get('index', 0) / len(cell_lines)
        elif node_data.get('type') == 'disease':
            # Disease index (normalized)
            x[idx, 3] = node_data.get('index', 0) / len(diseases)
    
    # Create edge list with weights
    edge_index = []
    edge_weights_list = []
    for u, v, data in G.edges(data=True):
        u_idx = node_mapping[u]
        v_idx = node_mapping[v]
        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])  # Undirected
        weight = data.get('weight', 0.5)
        edge_weights_list.append(weight)
        edge_weights_list.append(weight)
    
    edge_index = torch.tensor(edge_index).t().contiguous() if edge_index else torch.tensor([[], []], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights_list, dtype=torch.float) if edge_weights_list else torch.tensor([], dtype=torch.float)

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
    
    # Save graph
    torch.save(data, processed_path / "graph.pt")
    
    # Save mappings for inference
    mappings = {
        'node_mapping': node_mapping,
        'drugs': drugs,
        'cell_lines': cell_lines,
        'diseases': diseases,
        'cancer_mapping': cancer_mapping,
    }
    with open(processed_path / "graph_mappings.pkl", 'wb') as f:
        pickle.dump(mappings, f)

    print(f"✅ Realistic KG built: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # === Save Graph Statistics for DVC Tracking ===
    stats = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_drugs": len(drugs),
        "num_cell_lines": len(cell_lines),
        "num_diseases": len(set(cancer_mapping.values())),
        "node_feature_dim": int(x.shape[1]),
        "graph_density": float(nx.density(G)),
        "avg_degree": float(2 * G.number_of_edges() / G.number_of_nodes()),
        "is_connected": nx.is_connected(G),
        "num_connected_components": nx.number_connected_components(G),
    }
    
    stats_path = processed_path / "graph_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"📊 Graph Statistics:")
    for key, value in stats.items():
        if isinstance(value, bool):
            print(f"   {key}: {value}")
        elif isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    return data

if __name__ == "__main__":
    build_heterogeneous_graph()