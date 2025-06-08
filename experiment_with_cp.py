import torch
import numpy as np
import kcn 
import data
import dt2_data
from tqdm import tqdm
from geocp import GeoCPWrapper
from data import SpatialDataset
import pickle

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
    
def MAE(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))/len(y_true)

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
    elif args.dataset == "n32_e035_1arc_v3_cropped":
       trainset, validset, testset = dt2_data.load_dt2_data(args)
    else: 
        raise Exception(f"The repo does not support this dataset yet: args.dataset={args.dataset}")
    print(f"The {args.dataset} dataset has {len(trainset)} training instances and {len(testset)} test instances.")

    # Calibration Set Creation
    num_total_train = len(trainset.y)
    num_calib = int(args.calib_percentage*num_total_train)
    num_train = num_total_train - num_calib
    print(f"num_total_train: {num_total_train}, num_calib: {num_calib}, num_train:{num_train}")
    # Split trainset into train + calibration
    train_coords, train_features, train_y = trainset.coords[:num_train], trainset.features[:num_train], trainset.y[:num_train]
    calib_coords, calib_features, calib_y = trainset.coords[num_train:], trainset.features[num_train:], trainset.y[num_train:]

    trainset =  SpatialDataset(
            coords=train_coords.numpy(),
            features=train_features.numpy(),
            y=train_y.numpy()
        )
    
    calibset =  SpatialDataset(
        coords=calib_coords.numpy(),
        features=calib_features.numpy(),
        y=calib_y.numpy()
    )

    torch.save(trainset, f"cache/trainset_divided_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
    torch.save(calibset, f"cache/calibset_divided_{args.dataset}_k{args.n_neighbors}_keep_n{args.keep_n}.pt")
    print(f"train_coords: {train_coords.shape}, train_features: {train_features.shape}, train_y:{train_y.shape}")
    print(f"calib_coords: {calib_coords.shape}, calib_features: {calib_features.shape}, calib_y:{calib_y.shape}")

    # Model's Trainging
    model = kcn.KCN(trainset, args)
    model = model.to(args.device)
    loss_func = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Model's relevant vecs
    epoch_train_loss = []
    epoch_train_error = []
    epoch_valid_error = []
    epoch_valid_loss = []
    epoch_train_mse = []
    epoch_train_mae = []
    epoch_valid_mse = []
    epoch_valid_mae = []
    coverage_rate_tot = []
    avg_interval_length_tot = []

    best_val_loss = float('inf')

    # the training loop
    model.train()
    for epoch in range(args.epochs):
        model.train()
        batch_train_error = [] 
        # use training indices directly because it will be used to retrieve pre-computed neighbors
        for i in tqdm(range(0, num_train, args.batch_size)):

            # fetch a batch of data  
            batch_ind = range(i, min(i + args.batch_size, num_train))
            batch_coords, batch_features, batch_y = model.trainset[batch_ind]

            # make predictions and compute the average loss
            pred = model(batch_coords, batch_y, batch_features,args, args.top_k, batch_ind)
            loss = loss_func(pred, batch_y.to(args.device))

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record the training error
            batch_train_error.append(loss.item())

        with torch.no_grad():
            train_loss = sum(batch_train_error) / len(batch_train_error)
            epoch_train_loss.append(train_loss)

        # Compute and store training predictions
        model.eval()
        with torch.no_grad():
            train_pred = model(trainset.coords, trainset.y, trainset.features, args,args.top_k)
            # Move y_std and y_mean to the same device
            y_std = trainset.y_std.cpu().numpy().squeeze()
            y_mean = trainset.y_mean.cpu().numpy().squeeze()
            train_pred = train_pred * y_std + y_mean

        model.eval()
        with torch.no_grad():
            # make predictions and calculate the error
            valid_pred = model(validset.coords, validset.y, validset.features, args,args.top_k)
            valid_loss = loss_func(valid_pred, validset.y.to(args.device))
            valid_pred = valid_pred * trainset.y_std.to(args.device) + trainset.y_mean.to(args.device)
            
            # Accuracies metrics:
            valid_error = validset.y.detach().cpu().numpy()* y_std + y_mean - valid_pred.detach().cpu().numpy()
            train_error = trainset.y.detach().cpu().numpy()* y_std + y_mean - train_pred.detach().cpu().numpy()
            
            valid_mse = MSE(validset.y.detach().cpu().numpy()* y_std + y_mean, valid_pred.detach().cpu().numpy())
            train_mse = MSE(trainset.y.detach().cpu().numpy()* y_std + y_mean, train_pred.detach().cpu().numpy())
            valid_mae = MAE(validset.y.detach().cpu().numpy()* y_std + y_mean, valid_pred.detach().cpu().numpy())
            train_mae = MAE(trainset.y.detach().cpu().numpy()* y_std + y_mean, train_pred.detach().cpu().numpy())

            # Results saving
            epoch_train_mae.append(train_mae)
            epoch_valid_loss.append(valid_loss)
            epoch_valid_mae.append(valid_mae)
            epoch_train_mse.append(train_mse)
            epoch_valid_mse.append(valid_mse)
            epoch_train_error.append(train_error)
            epoch_valid_error.append(valid_error)

            print(f"Epoch: {epoch},", f"train loss: {train_loss},", f"train mse: {train_mse}", f"train mae: {train_mae}", f"validation loss: {valid_loss}",f"valid mse: {valid_mse}", f"valid mae: {valid_mae}")
            print(f"Train Mean Err: {np.mean(train_error)},", f"Train STD Err: {np.std(train_error)},", f"Val Mean Err: {np.mean(valid_error)},", f"Val STD Err: {np.std(valid_error)}")

            if (epoch > args.es_patience) and \
                (np.mean(np.array(epoch_valid_error[-3:])) >
                np.mean(np.array(epoch_valid_error[-(args.es_patience + 3):-3]))):
                print("\nEarly stopping at epoch {}".format(epoch))
                break

            # Optionally save best one
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                torch.save(model.state_dict(), f"{args.save_path}/best_model_epoch{epoch}.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": range(args.epochs),
        "metrics": {
            "train_loss": epoch_train_loss,
            "train_error": epoch_train_error,
            "valid_loss": epoch_valid_loss,
            "valid_error": epoch_valid_error,
            "train_mse": epoch_train_mse,
            "valid_mse": epoch_valid_mse,
            "train_mae": epoch_train_mae,
            "valid_mae": epoch_valid_mae,
        }
    }
    torch.save(checkpoint, f"{args.save_path}/checkpoint_epoch{epoch}.pt")
    # CP BLOCK
    model.eval()
    with torch.no_grad():
        y_zeros = torch.zeros(testset.y.shape[0])
        print(f"Y_Zeros:{y_zeros}")
        test_preds_norm = model(testset.coords, y_zeros,testset.features, args, args.top_k) #MAKE SURE THIS IS OK
        test_loss = loss_func(test_preds_norm, testset.y.to(args.device))
        test_preds = test_preds_norm.to(args.device) * trainset.y_std.to(args.device) + trainset.y_mean.to(args.device)
        test_error = testset.y.detach().cpu().numpy() * y_std + y_mean - test_preds.detach().cpu().numpy()
        test_mse = MSE(testset.y.detach().cpu().numpy()* y_std + y_mean, test_preds.detach().cpu().numpy())
        test_mae = MAE(testset.y.detach().cpu().numpy()* y_std + y_mean, test_preds.detach().cpu().numpy())
        #test_mae = np.mean(np.abs(testset.y.detach().numpy() - test_pred.detach().numpy()))
        print(f"Test loss is {test_loss}")
        print(f"Test MSE is {test_mse}")
        print(f"Test MAE is {test_mae}")
        print(f"Test Mean error is {np.mean(test_error)},",f"Test STD error is {np.std(test_error)}" )

        geocp = GeoCPWrapper(model, calib_coords, calib_y, calib_features, args, eps=0.1, decay_beta=1.0, device=args.device)

        n_total = testset.y.shape[0]
        n_covered = 0
        interval_lengths = []
        CP_results = []
        for i in range(n_total):
            lower, upper = geocp.predict_interval(testset.coords[i], testset.y[i], testset.features[i], args)
            true_y = testset.y[i].item()

            if lower <= true_y <= upper:
                n_covered += 1
            interval_lengths.append(upper - lower)
            avg_interval_length = np.mean(interval_lengths)
            avg_interval_length_tot.append(avg_interval_length)

            CP_results.append({
                'lower': lower,
                'upper': upper
            })

        coverage_rate = n_covered / n_total

    print(f"GeoCP Coverage Rate: {coverage_rate:.3f}")
    print(f" Interval Length: {interval_lengths}")
    print(f"Average Prediction Interval Length: {avg_interval_length_tot}")
    print(f"Mean Average Prediction Interval Length: {np.mean(avg_interval_length_tot)}")
    
    #torch.save(CP_results, f"{args.save_path}/CP_results.pt")
    with open(f"{args.save_path}/CP_results.pkl", "wb") as f:
        pickle.dump(CP_results, f)

    return test_error, test_preds, testset, epoch_valid_loss, epoch_valid_error, epoch_valid_mse, epoch_valid_mae, epoch_train_loss, epoch_train_error, epoch_train_mse, epoch_train_mae, coverage_rate_tot, avg_interval_length_tot, test_preds_norm
    #return test_error, test_preds, testset, epoch_valid_loss, epoch_valid_error, epoch_valid_mse, epoch_valid_mae, epoch_train_loss, epoch_train_error, epoch_train_mse, epoch_train_mae, coverage_rate_tot, avg_interval_length_tot
