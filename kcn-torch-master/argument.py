import argparse
import torch

def parse_opt():
    '''custom_args = [
        "--dataset", "n32_e035_1arc_v3",
        "--device", "cuda",
        "--model", "kcn",
        "--n_neighbors", "5",
        "--num_hops", "3"
    ]'''
    # Settings
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default="5", help="The random seed")
    parser.add_argument('--keep_n', type=float, default="0.005", help="The random seed")
    parser.add_argument('--normalize_elev', type=bool, default=True, help="The random seed")
    #parser.add_argument('--dataset', type=str, default="bird_count", help="The dataset name: currently can only be 'bird_count'")
    #Orit
    parser.add_argument('--form_input_graph', type=str, default="original", help="The dataset name (either 'bird_count' or DT2 file name)")
    parser.add_argument('--dataset', type=str, default="n32_e035_1arc_v3", help="The dataset name (either 'bird_count' or DT2 file name)")
    parser.add_argument('--data_path', type=str, default="./datasets", help="The folder containing the data file. The default file is './data/{dataset}.pkl'")
    parser.add_argument('--use_default_test_set', type=bool, default=False, help='Use the default test set from the data')
    
    parser.add_argument('--model', type=str, default='kcn', help='One of three model types, kcn, kcn_gat, kcn_sage, which use GCN, GAT, and GraphSAGE respectively')
    parser.add_argument('--n_neighbors', type=int, default=100, help='Number of neighbors')
    parser.add_argument('--top_k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--length_scale', default="auto", help='Length scale for RBF kernel. If set to "auto", then it will be set to the median of neighbor distances')
    parser.add_argument('--hidden_sizes', type=list, default=[64, 128, 256], help='Number of units in hidden layers, also decide the number of layers')
    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--last_activation', type=str, default='none', help='Activation for the last layer')
    
    parser.add_argument('--loss_type', type=str, default='squared_error', help='Loss type') 
    parser.add_argument('--validation_size', type=float, default=0.2, help='Validation size') 
    
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--es_patience', type=int, default=20, help='Patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    parser.add_argument('--device', type=str, default="auto", help='Computation device.')
    parser.add_argument('--num_hops', type=int, default=3, help='Number of hops to include in the graph.')
    #args, unknowns = parser.parse_known_args()
    #args = parser.parse_args(custom_args)  # ← don't use sys.argv at all
    args, unknowns = parser.parse_known_args()
    args.save_path = f"saved_models/{args.model}_{args.dataset}/"
    if args.device == "auto":
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #args.device = 'cpu'
    else:
        args.device = torch.device(args.device)
        #args.device = 'cpu'
    print(f"device: {args.device}")

    return args
