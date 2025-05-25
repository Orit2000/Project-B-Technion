import torch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
import torch.serialization
#torch.serialization.add_safe_globals([Data])


# Load saved graphs (adjust path and k, keep_n accordingly)
graph_inputs = torch.load("cache/graph_inputs_n32_e035_1arc_v3_cropped_k50_keep_n0.00125.pt", weights_only=False ,map_location="cpu")

# Choose an index to visualize (e.g., 0, or a random sample)
idx = 2
graph = graph_inputs[idx]

# Convert to NetworkX graph (optional: to_undirected=True for clarity)
G = to_networkx(graph, edge_attrs=['edge_attr'], to_undirected=True)

# Draw the graph
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G, seed=42)  # Or use circular/spectral/etc.
nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', edge_color='gray')
edge_labels = nx.get_edge_attributes(G, 'edge_attr')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
plt.title(f"Graph structure for training point #{idx}")
plt.show()

print("Node features (x):", graph.x)
print("Edge indices (edge_index):", graph.edge_index)
print("Edge weights (edge_attr):", graph.edge_attr)
