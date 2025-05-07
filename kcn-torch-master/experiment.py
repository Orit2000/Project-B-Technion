import torch
import numpy as np
import kcn 
import data
import dt2_data
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import os

def run_kcn(args):
    """ Train and test a KCN model on a train-test split  

    Args
    ----
    args : argparse.Namespace object, which contains the following attributes:
        - 'model' : str, which is one of 'gcn', 'gcn_gat', 'gcn_sage'
        - 'n_neighbors' : int, number of neighbors
        - 'hidden1' : int, number of units in hidden layer 1
        - 'dropout' : float, the dropout rate in a dropout layer 
        - 'lr' : float, learning rate of the Adam optimizer
        - 'epochs' : int, number of training epochs
        - 'es_patience' : int, patience for early stopping
        - 'batch_size' : int, batch size
        - 'dataset' : str, path to the data file
        - 'last_activation' : str, activation for the last layer
        - 'weight_decay' : float, weight decay for the Adam optimizer
        - 'length_scale' : float, length scale for RBF kernel
        - 'loss_type' : str, which is one of 'squared_error', 'nll_error'
        - 'validation_size' : int, validation size
        - 'gcn_kriging' : bool, whether to use gcn kriging
        - 'sparse_input' : bool, whether to use sparse matrices
        - 'device' : torch.device, which is either 'cuda' or 'cpu'

    """
    # This function has the following three steps:
    # 1) loading data; 2) spliting the data into training and test subsets; 3) normalizing data 
    print(f"from run_kcn: {args.dataset}")
    
    if args.dataset == "bird_count":
        trainset, testset = data.load_bird_count_data(args)
    elif args.dataset == "n32_e035_1arc_v3":
        trainset, validset, testset = dt2_data.load_dt2_data(args)
    else: 
        raise Exception(f"The repo does not support this dataset yet: args.dataset={args.dataset}")

    print(f"The {args.dataset} dataset has {len(trainset)} training instances and {len(testset)} test instances.")

    num_train = len(trainset.y)
    # initialize a kcn model
    # 1) the entire training set including validation points are recorded by the model and will 
    # be looked up in neighbor searches
    # 2) the model will pre-compute neighbors for a training or validation instance to avoid repeated neighbor search
    # 3) if a data point appears in training set and validation set, its neighbors does not include itself
    model = kcn.KCN(trainset, args)
    model = model.to(args.device)

    loss_func = torch.nn.MSELoss(reduction='mean')
    #show_sample_graph(model, index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_train_error = []
    epoch_valid_error = []
    best_val_loss = float('inf')
    # the training loop
    model.train()


    for epoch in range(args.epochs):

        batch_train_error = [] 

        # use training indices directly because it will be used to retrieve pre-computed neighbors
        for i in tqdm(range(0, num_train, args.batch_size)):

            # fetch a batch of data  
            batch_ind = range(i, min(i + args.batch_size, num_train))
            batch_coords, batch_features, batch_y = model.trainset[batch_ind]

            # make predictions and compute the average loss
            pred = model(batch_coords, batch_features, args.top_k, batch_ind)
            loss = loss_func(pred, batch_y.to(args.device))

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record the training error
            batch_train_error.append(loss.item())

        train_error = sum(batch_train_error) / len(batch_train_error)
        epoch_train_error.append(train_error)
        '''
        # fetch the validation set
        valid_ind = range(num_train, num_total_train)
        valid_coords, valid_features, valid_y = model.trainset[valid_ind]
        '''
        # make predictions and calculate the error
        valid_pred = model(validset.coords, validset.features, args.top_k)
        valid_pred = valid_pred * trainset.y_std + trainset.y_mean
        valid_error = loss_func(valid_pred, validset.y.to(args.device))

        epoch_valid_error.append(valid_error.item())
        print(f"Epoch: {epoch},", f"train error: {train_error},", f"validation error: {valid_error}")

        # check whether to stop 
        if (epoch > args.es_patience) and \
                (np.mean(np.array(epoch_valid_error[-3:])) >
                 np.mean(np.array(epoch_valid_error[-(args.es_patience + 3):-3]))):
            print("\nEarly stopping at epoch {}".format(epoch))
            break
        
        #Savings
        torch.save(model.state_dict(), f"{args.save_path}/weights_epoch{epoch}.pt")

        # Optionally save best one
        if valid_error < best_val_loss:
            best_val_loss = valid_error
            torch.save(model.state_dict(), f"{args.save_path}/best_model_epoch{epoch}.pt")

    # test the model
    model.eval()
    with open("logs/loss_tracking_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.txt", "a") as f:
        f.write(f"{epoch},{train_error:.6f},{valid_error.item():.6f}\n")
    test_preds = model(testset.coords, testset.features, args.top_k) #MAKE SURE THIS IS OK
    test_preds = test_preds * trainset.y_std + trainset.y_mean
    test_error = loss_func(test_preds, testset.y.to(args.device))

    print(f"Test error is {test_error}")

    return test_error 

def show_sample_graph(model, index=0):
    graph = model.graph_inputs[index]
    G = to_networkx(graph, to_undirected=True)
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_size=50)
    plt.title(f"Sample Graph for Point #{index}")
    plt.show()
    
    