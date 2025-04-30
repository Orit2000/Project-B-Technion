def inspect_dataset(dataset, name="Train"):
    print(f"\n📦 {name} Dataset Summary")
    print(f"➤ Number of points: {len(dataset)}")
    print(f"➤ Feature shape: {dataset.features.shape}")
    print(f"➤ Label shape: {dataset.y.shape}")
    print(f"➤ Feature mean/std (first 5 dims):")
    print(f"   μ = {dataset.features.mean(0)[:5].numpy()}")
    print(f"   σ = {dataset.features.std(0)[:5].numpy()}")
    print(f"➤ Elevation min/max: {dataset.y.min().item():.2f} / {dataset.y.max().item():.2f}")

    # Coordinates info
    coords = dataset.coords.numpy()
    print(f"➤ Lat range: {coords[:, 0].min():.4f} - {coords[:, 0].max():.4f}")
    print(f"➤ Lon range: {coords[:, 1].min():.4f} - {coords[:, 1].max():.4f}")

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def show_sample_graph(model, index=0):
    graph = model.graph_inputs[index]
    G = to_networkx(graph, to_undirected=True)
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_size=50)
    plt.title(f"Sample Graph for Point #{index}")
    plt.show()

