import torch
import networkx as nx
from torch_geometric.data import Data
import pandas as pd
from src.utils.config import CONFIG
import os

def build_heterogeneous_graph():
    print("🔨 Building heterogeneous knowledge graph...")
    os.makedirs(CONFIG["processed_data_path"], exist_ok=True)
    
    # Synthetic small graph for exam (easy to run)
    G = nx.Graph()
    drugs = ["DRUG1", "DRUG2", "DRUG3", "DRUG4"]
    genes = ["GENE_A", "GENE_B", "GENE_C"]
    diseases = ["CANCER_LUNG", "CANCER_BREAST", "CANCER_GLIOMA"]
    
    # Add nodes
    for d in drugs: G.add_node(d, type="drug")
    for g in genes: G.add_node(g, type="gene")
    for dis in diseases: G.add_node(dis, type="disease")
    
    # Add edges (known + some new)
    G.add_edges_from([("DRUG1", "GENE_A"), ("GENE_A", "CANCER_LUNG"), ("DRUG1", "CANCER_LUNG")])
    G.add_edges_from([("DRUG2", "GENE_B"), ("GENE_B", "CANCER_BREAST")])
    
    # Convert to PyG Data object
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()]).t().contiguous()
    
    data = Data(edge_index=edge_index, num_nodes=len(G.nodes()))
    torch.save(data, f"{CONFIG['processed_data_path']}graph.pt")
    
    print(f"✅ Graph built with {data.num_nodes} nodes and {data.num_edges} edges.")
    return data

if __name__ == "__main__":
    build_heterogeneous_graph()