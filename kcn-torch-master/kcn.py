import numpy as np
import sklearn
import sklearn.neighbors
import torch
import torch_geometric
import os, torch


class KCN(torch.nn.Module):
    """ Creates a KCN model with the given parameters."""

    def __init__(self, trainset, args) -> None:
        super(KCN, self).__init__()
        save_metadata(args, trainset.y_mean, trainset.y_std)
        self.trainset = trainset
        cache_path = f"cache/graph_inputs_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt"
        # set neighbor relationships within the training set
        self.n_neighbors = args.n_neighbors
        self.knn = sklearn.neighbors.NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.trainset.coords)
        distances, self.train_neighbors = self.knn.kneighbors(None, return_distance=True) # the shape of distances and train_neighbors are (N,5)
        os.makedirs("logs", exist_ok=True)
        with open("logs/output.txt", "w") as f:
            if args.length_scale == "auto": 
                self.length_scale = np.median(distances.flatten())
                print(f"Length scale is set to {self.length_scale}", file=f)
            else:
                if not isinstance(args.length_scale, float):
                    raise Exception(f"If the provided length scale is not 'auto', then it should be a float number: args.length_scale={args.length_scale}")
                self.length_scale = args.length_scale
            if os.path.exists(cache_path):
                print("Loading cached graphs...",file=f)
                self.graph_inputs = torch.load(cache_path,weights_only=False)
            else:
                with torch.no_grad():
                    self.graph_inputs = []
                    for i in range(self.trainset.coords.shape[0]):
                        #att_graph = self.form_input_graph(self.trainset.coords[i], self.trainset.features[i], self.train_neighbors[i],args.top_k)
                        att_graph = self.form_input_graph(self.trainset.coords[i], self.trainset.features[i], self.train_neighbors[i])
                        self.graph_inputs.append(att_graph)
                torch.save(self.graph_inputs, cache_path)    
            print(f"len of graph_inputs:{len(self.graph_inputs)}",file=f)
        # initialize model
        # input dimensions should be feature dimensions, a label dimension and an indicator dimension 
        input_dim = trainset.features.shape[1] + 2 #CHANGES TO 1, since features contains the labels
        output_dim = trainset.y.shape[1]

        self.gnn = GNN(input_dim, args)

        # the last linear layer
        self.linear = torch.nn.Linear(args.hidden_sizes[-1], output_dim, bias=False)

        # the last activation function
        if args.last_activation == 'relu':
            self.last_activation = torch.nn.ReLU()
        elif args.last_activation == 'sigmoid':
            self.last_activation = torch.nn.Sigmoid()
        elif args.last_activation == 'tanh':
            self.last_activation = torch.nn.Tanh()
        elif args.last_activation == 'softplus':
            self.last_activation = torch.nn.Softplus()
        elif args.last_activation == 'none':
            self.last_activation = lambda _: _ 
        else:
            raise Exception(f"No such choice of activation for the output: args.last_activation={args.last_activation}")


        self.collate_fn = torch_geometric.loader.dataloader.Collater(None, None)

        self.device = args.device
        self.gnn = self.gnn.to(self.device)

# %%
    def forward(self, coords, features, top_k, train_indices=None):
        if train_indices is not None:
            
            # if from training set, then read in pre-computed graphs
            batch_inputs = []
            for i in train_indices: 
                batch_inputs.append(self.graph_inputs[i])

            batch_inputs = self.collate_fn(batch_inputs) 


        else:

            # if new instances, then need to find neighbors and form input graphs
            neighbors = self.knn.kneighbors(coords, return_distance=False)

            with torch.no_grad():
                batch_inputs = []
                for i in range(len(coords)):
                    #att_graph = self.form_input_graph(coords[i], features[i], neighbors[i], top_k)
                    att_graph = self.form_input_graph(coords[i], features[i], neighbors[i])
                    batch_inputs.append(att_graph)
                    #show_sample_graph(model, index=0)
                batch_inputs = self.collate_fn(batch_inputs) 

        batch_inputs = batch_inputs.to(self.device)

        # run gnn on the graph input
        output = self.gnn(batch_inputs.x, batch_inputs.edge_index, batch_inputs.edge_attr)

        # take representations only corresponding to center nodes 
        output = torch.reshape(output, [-1, (self.n_neighbors + 1), output.shape[1]])
        center_output = output[:, 0]
        pred = self.last_activation(self.linear(center_output))

        return pred
# %%
    def form_input_graph_mine(self, coord, feature, neighbors,top_k):
    
        output_dim = self.trainset.y.shape[1] 

        # label inputs
        y = torch.concat([torch.zeros([1, output_dim]), self.trainset.y[neighbors]], axis=0) #[0, y_neighbor_1, y_neigbor_2,...], shape: 1x(output_dim*(1+len(neighbors)))
    
        # indicator
        indicator = torch.zeros([neighbors.shape[0] + 1])
        indicator[0] = 1.0 #[1,0,0,0,0,0], shape: 1*(neighbors.shape[0] + 1)
    
        # feature inputs 
        features = torch.concat([feature[None, :], self.trainset.features[neighbors]], axis=0) # A matrix! Each row is the vec feature for the current node. The first node is itself and the other are the neighbors.
        # Final features shape: [neighbors.shape[0] + 1, ],feature.shape[0]] 
        # form graph features
        graph_features = torch.concat([features, y, indicator[:, None]], axis=1) # A matrix shape (neighbors.shape[0] + 1, feartures.shape[0]+y.shape+1)
    

        # compute a weighted graph from an rbf kernel
        all_coords = torch.concat([coord[None, :], self.trainset.coords[neighbors]], axis=0)  # A matrix shape (neighbors.shape[0] + 1,2)

        # K(x, y) = exp(-gamma ||x-y||^2)
        kernel = sklearn.metrics.pairwise.rbf_kernel(all_coords.numpy(), gamma=1 / (2 * self.length_scale ** 2)) # A matrix shape ( # A matrix shape (neighbors.shape[0] + 1, (neighbors.shape[0] + 1)
        # This matrix has 1 on the diagonal and values under 1 in the other places
        ## the implementation here is the same as sklearn.metrics.pairwise.rbf_kernel
        #row_norm = torch.sum(torch.square(all_coords), dim=1)
        #dist = row_norm[:, None] - 2 * torch.matmul(all_coords, all_coords.t()) + row_norm[None, :]
        #kernel = torch.exp(-self.length_scale * dist)
        ## Orit's
        #adj = torch.from_numpy(kernel)
        adj = get_multihop_neighbors(torch.from_numpy(kernel), num_hops=3, top_k=top_k) #zeros out the weak edges to get a sparse matrix # ***PAYATTENTION***
        adj.fill_diagonal_(0.0)
        with open("logs/output.txt", "a") as f:
            print(f"adj shape:{adj.shape}",file=f)
            # one choice is to normalize the adjacency matrix 
            #curr_adj = normalize_adj(curr_adj + np.eye(curr_adj.shape[0]))
        
            # create a graph from it
            nz = adj.nonzero(as_tuple=True) #Finds the positions of non-zero entries in adj, i.e., where edges exist. nz = (row_indices, col_indices)
            #print(f"nz shape:{nz.shape}")
            edges = torch.stack(nz, dim=0) #Builds the edge_index tensor for PyG:
            '''
            In PyTorch Geometric, the graph's edge list is stored as:
            edge_index.shape = (2, num_edges)
            edge_index = [ [source_0, source_1, ..., source_N],
                        [target_0, target_1, ..., target_N] ]
            This is called the COO (coordinate) format for sparse graphs.
            
            sources = nz[0]  # row indices (i)
            targets = nz[1]  # col indices (j)
            edges = torch.stack([sources, targets], dim=0)
            '''
            print(f"edges shape:{edges.shape}",file=f)
            edge_weights = adj[nz] #Picks the values of the retained edges from adj, i.e., the actual weights of the edges in the graph.
            print(f"edge_weights shape:{edge_weights.shape}",file=f)
        
            # form the graph
            attributed_graph = torch_geometric.data.Data(x=graph_features, edge_index=edges, edge_attr=edge_weights, y=None)
            print(f"attributed_graph shape:{edge_weights.shape}",file=f)
        return attributed_graph 
# %%
    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
    
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
        adj_normalized = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    
        return adj_normalized
    
# %%
    def form_input_graph(self, coord, feature, neighbors):
        
            output_dim = self.trainset.y.shape[1]

            # label inputs
            y = torch.concat([torch.zeros([1, output_dim]), self.trainset.y[neighbors]], axis=0)
        
            # indicator
            indicator = torch.zeros([neighbors.shape[0] + 1])
            indicator[0] = 1.0
        
            # feature inputs 
            features = torch.concat([feature[None, :], self.trainset.features[neighbors]], axis=0)

            # form graph features
            graph_features = torch.concat([features, y, indicator[:, None]], axis=1)
        

            # compute a weighted graph from an rbf kernel
            all_coords = torch.concat([coord[None, :], self.trainset.coords[neighbors]], axis=0)

            # K(x, y) = exp(-gamma ||x-y||^2)
            kernel = sklearn.metrics.pairwise.rbf_kernel(all_coords.numpy(), gamma=1 / (2 * self.length_scale ** 2))
            ## the implementation here is the same as sklearn.metrics.pairwise.rbf_kernel
            #row_norm = torch.sum(torch.square(all_coords), dim=1)
            #dist = row_norm[:, None] - 2 * torch.matmul(all_coords, all_coords.t()) + row_norm[None, :]
            #kernel = torch.exp(-self.length_scale * dist)

            adj = torch.from_numpy(kernel)
            # one choice is to normalize the adjacency matrix 
            #curr_adj = normalize_adj(curr_adj + np.eye(curr_adj.shape[0]))
        
            # create a graph from it
            nz = adj.nonzero(as_tuple=True)
            edges = torch.stack(nz, dim=0)
            edge_weights = adj[nz]
        
            # form the graph
            attributed_graph = torch_geometric.data.Data(x=graph_features, edge_index=edges, edge_attr=edge_weights, y=None)
        
            return attributed_graph 

# %%
class GNN(torch.nn.Module):
    """ Creates a KCN model with the given parameters."""

    def __init__(self, input_dim, args) -> None:
        super().__init__()

        self.hidden_sizes = args.hidden_sizes
        self.dropout = args.dropout
        self.model_type = args.model

        if self.model_type == 'kcn':
            conv_layer = torch_geometric.nn.GCNConv (input_dim, self.hidden_sizes[0], bias=False, add_self_loops=False)
        elif self.model_type == 'kcn_gat':
            conv_layer = torch_geometric.nn.GATConv (input_dim, self.hidden_sizes[0])
        elif self.model_type == 'kcn_sage':
            conv_layer = torch_geometric.nn.SAGEConv(input_dim, self.hidden_sizes[0], aggr='max', normalize=True)
        else:
            raise Exception(f"No such model choice: args.model={args.model}")

        self.add_module("layer0", conv_layer)


        for ilayer in range(1, len(self.hidden_sizes)): #3 layers
            if self.model_type == 'kcn':
                conv_layer = torch_geometric.nn.GCNConv (self.hidden_sizes[ilayer - 1], self.hidden_sizes[ilayer], bias=False, add_self_loops=False)
            elif self.model_type == 'kcn_gat':
                conv_layer = torch_geometric.nn.GATConv (self.hidden_sizes[ilayer - 1], self.hidden_sizes[ilayer])
            elif self.model_type == 'kcn_sage':
                conv_layer = torch_geometric.nn.SAGEConv(self.hidden_sizes[ilayer - 1], self.hidden_sizes[ilayer], aggr='max', normalize=True)

            self.add_module("layer"+str(ilayer), conv_layer)

    def forward(self, x, edge_index, edge_weight):

        for conv_layer in self.children():

            if self.model_type == 'kcn':
                x = conv_layer(x, edge_index, edge_weight=edge_weight)

            elif self.model_type == 'kcn_gat':
                x, (edge_index, attention_weights) = conv_layer(x, edge_index, edge_attr=edge_weight, return_attention_weights=True)
                #edge_weight = ttention_weights

            elif self.model_type == 'kcn_sage':
                x = conv_layer(x, edge_index)

            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        return x
# %%
def get_multihop_neighbors(adj: torch.Tensor, num_hops: int = 3, top_k: int = 10):
    """
    Compute multi-hop neighbors up to a given number of hops.
    Args:
        adj: [N, N] tensor, weighted adjacency matrix
        num_hops: how many adjacency powers to consider
        top_k: maximum number of neighbors to keep per node
    Returns:
        new_adj: thresholded [N, N] tensor with multi-hop connections
    """
    N = adj.shape[0]
    A_power = adj.clone()
    combined_adj = adj.clone()

    for hop in range(2, num_hops + 1):
        A_power = torch.matmul(A_power, adj)
        combined_adj += A_power

    # Keep top-k connections per row
    for i in range(N):
        row = combined_adj[i]
        topk_vals, topk_idx = torch.topk(row, top_k) #This filters out weak/unimportant edges — keeps graph sparse and focused.
        mask = torch.zeros_like(row)
        mask[topk_idx] = topk_vals
        combined_adj[i] = mask

    return combined_adj
# %%
def show_sample_graph(model, index=0):
    graph = model.graph_inputs[index]
    G = to_networkx(graph, to_undirected=True)
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_size=50)
    plt.title(f"Sample Graph for Point #{index}")
    plt.show()
# %%
def save_metadata(args, y_mean, y_std):
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, f"{args.model}_{args.dataset}_metadata.pt")

    shared_info = {
    'args': vars(args),
    'y_mean': y_mean,
    'y_std': y_std,
    }
    torch.save(shared_info, save_path)
    print(f"Metadata saved to: {save_path}")

#%%
def save_checkpoint(model, args, y_mean, y_std, train_loss=None, val_loss=None):
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, "model.pt")  # fixed name for consistency

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if train_loss is not None and val_loss is not None:
        checkpoint['train_loss'] = train_loss
        checkpoint['val_loss'] = val_loss

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")

