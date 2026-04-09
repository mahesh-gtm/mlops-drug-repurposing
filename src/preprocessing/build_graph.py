import torch
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
from src.utils.config import CONFIG
import os
from pathlib import Path

def build_heterogeneous_graph():
    print("🔨 Building realistic heterogeneous knowledge graph from real GDSC data...")

    # === ROBUST DIRECTORY CREATION (fixes FileExistsError) ===
    processed_path = Path(CONFIG["processed_data_path"])
    
    if processed_path.exists() and not processed_path.is_dir():
        print(f"⚠️  Found a file instead of directory at {processed_path}. Removing it.")
        processed_path.unlink()  # Delete the conflicting file

    processed_path.mkdir(parents=True, exist_ok=True)

    # Load real GDSC data
    df = pd.read_csv(f"{CONFIG['raw_data_path']}gdsc2_original.csv")

    G = nx.Graph()

    drugs = df['DRUG_NAME'].unique()[:40]
    cell_lines = df['CELL_LINE'].unique()[:40]

    cancer_mapping = {
        "A549": "Lung Cancer",
        "MCF7": "Breast Cancer",
        "HT29": "Colorectal Cancer",
        "PC3": "Prostate Cancer",
        "U251": "Glioma",
    }

    # Add nodes
    for drug in drugs:
        G.add_node(drug, type="drug")
    for cell in cell_lines:
        G.add_node(cell, type="cell_line")
    for cancer in set(cancer_mapping.values()):
        G.add_node(cancer, type="disease")

    # Add real edges from GDSC
    for _, row in df.iterrows():
        if row['DRUG_NAME'] in drugs and row['CELL_LINE'] in cell_lines:
            G.add_edge(row['DRUG_NAME'], row['CELL_LINE'],
                       weight=float(row['IC50']),
                       auc=float(row['AUC']),
                       relation="tested_on")

    # Add realistic disease connections
    for cell, cancer in cancer_mapping.items():
        if cell in cell_lines:
            for drug in drugs:
                if G.has_edge(drug, cell):
                    G.add_edge(drug, cancer, relation="associated_via_cell_line")

    # Convert to PyTorch Geometric
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()]).t().contiguous()
    x = torch.randn(len(G.nodes()), 32)

    data = Data(x=x, edge_index=edge_index, num_nodes=len(G.nodes()))
    torch.save(data, f"{CONFIG['processed_data_path']}graph.pt")

    print(f"✅ Realistic KG built: {data.num_nodes} nodes, {data.num_edges} edges")
    return data

if __name__ == "__main__":
    build_heterogeneous_graph()