import torch
import numpy as np
import kcn 
import data
import dt2_data
from tqdm import tqdm
from geocp import GeoCPWrapper


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
        trainset, testset = dt2_data.load_dt2_data(args)
    else: 
        raise Exception(f"The repo does not support this dataset yet: args.dataset={args.dataset}")

    print(f"The {args.dataset} dataset has {len(trainset)} training instances and {len(testset)} test instances.")

    #num_total_train = len(trainset)
    #num_valid = args.validation_size
    #num_train = num_total_train - args.validation_size
    num_total_train = len(trainset)
    num_calib = args.validation_size
    num_train = num_total_train - num_calib
    print(f"num_total_train: {num_total_train}, num_calib: {num_calib}, num_train:{num_train}")

    # Split trainset into train + calibration
    train_coords, train_features, train_y = trainset.coords[:num_train], trainset.features[:num_train], trainset.y[:num_train]
    calib_coords, calib_features, calib_y = trainset.coords[num_train:], trainset.features[num_train:], trainset.y[num_train:]

    print(f"train_coords: {train_coords.shape}, train_features: {train_features.shape}, train_y:{train_y.shape}")
    print(f"calib_coords: {calib_coords.shape}, calib_features: {calib_features.shape}, calib_y:{calib_y.shape}")
    # initialize a kcn model
    # 1) the entire training set including validation points are recorded by the model and will 
    # be looked up in neighbor searches
    # 2) the model will pre-compute neighbors for a training or validation instance to avoid repeated neighbor search
    # 3) if a data point appears in training set and validation set, its neighbors does not include itself
    model = kcn.KCN(trainset, args)
    model = model.to(args.device)
    loss_func = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_train_error = []
    epoch_valid_error = []

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
            pred = model(batch_coords, batch_features, batch_ind)
            loss = loss_func(pred, batch_y.to(args.device))

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record the training error
            batch_train_error.append(loss.item())

        train_error = sum(batch_train_error) / len(batch_train_error)
        epoch_train_error.append(train_error)

        # fetch the validation set
        valid_ind = range(num_train, num_total_train)
        valid_coords, valid_features, valid_y = model.trainset[valid_ind]

        # make predictions and calculate the error
        valid_pred = model(valid_coords, valid_features, valid_ind)
        valid_error = loss_func(valid_pred, valid_y.to(args.device))

        epoch_valid_error.append(valid_error.item())

        print(f"Epoch: {epoch},", f"train error: {train_error},", f"validation error: {valid_error}")

        # check whether to stop 
        if (epoch > args.es_patience) and \
                (np.mean(np.array(epoch_valid_error[-3:])) >
                 np.mean(np.array(epoch_valid_error[-(args.es_patience + 3):-3]))):
            print("\nEarly stopping at epoch {}".format(epoch))
            break

    # test the model
    model.eval()
    geocp = GeoCPWrapper(model, calib_coords, calib_features, calib_y, eps=0.1, decay_beta=1.0, device=args.device)

    # 4) Evaluate on test set with GeoCP
    test_coords = testset.coords
    test_features = testset.features
    test_y = testset.y

    n_total = test_coords.shape[0]
    n_covered = 0
    interval_lengths = []

    for i in range(n_total):
        lower, upper = geocp.predict_interval(test_coords[i], test_features[i])
        true_y = test_y[i].item()

        if lower <= true_y <= upper:
            n_covered += 1
        interval_lengths.append(upper - lower)

    coverage_rate = n_covered / n_total
    avg_interval_length = np.mean(interval_lengths)

    print(f"GeoCP Coverage Rate: {coverage_rate:.3f}")
    print(f"Average Prediction Interval Length: {avg_interval_length:.3f}")

    return coverage_rate, avg_interval_length
