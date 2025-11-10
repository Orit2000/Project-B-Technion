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

    parser.add_argument('--max_km', type=int, default=None, help="The random seed")
    parser.add_argument('--max_obs', type=int, default=5, help='Batch size')
    parser.add_argument('--include_elevation_in_features', type=bool, default=True, help='Batch size')
    parser.add_argument('--setsdistribtuion', type=str, default="equal", help="The random seed")
    parser.add_argument('--datasampling', type=str, default="uniform", help="The random seed")
    parser.add_argument('--new_spread', type=bool, default="False", help="The random seed")
    parser.add_argument('--random_seed', type=int, default="5", help="The random seed")
    parser.add_argument('--keep_n', type=float, default="0.005", help="The random seed")
    parser.add_argument('--normalize_elev', type=bool, default=True, help="The random seed")
    parser.add_argument('--calib_percentage', type=float, default=0.5, help="The random seed")
    #parser.add_argument('--dataset', type=str, default="bird_count", help="The dataset name: currently can only be 'bird_count'")
    #Oritn32_e035_1arc_v3_cropped
    parser.add_argument('--form_input_graph', type=str, default="original", help="The dataset name (either 'bird_count' or DT2 file name)")
    parser.add_argument('--dataset', type=str, default="n32_e035_1arc_v3_cropped", help="The dataset name (either 'bird_count' or DT2 file name)")
    #parser.add_argument('--data_path', type=str, default=".\Transformer_Map_Interp\datasets", help="The folder containing the data file. The default file is './data/{dataset}.pkl'")
    parser.add_argument('--data_path', type=str, default="./Transformer_Map_Interp/datasets", help="The folder containing the data file. The default file is './data/{dataset}.pkl'")
    parser.add_argument('--use_default_test_set', type=bool, default=False, help='Use the default test set from the data')
    
    parser.add_argument('--model', type=str, default='kcn', help='One of three model types, kcn, kcn_gat, kcn_sage, which use GCN, GAT, and GraphSAGE respectively')
    #parser.add_argument('--n_neighbors', type=int, default=50, help='Number of neighbors')
    #parser.add_argument('--top_k', type=int, default=5, help='Number of neighbors')
    #parser.add_argument('--length_scale', default="auto", help='Length scale for RBF kernel. If set to "auto", then it will be set to the median of neighbor distances')
    #parser.add_argument('--hidden_sizes', type=list, default=[64, 128, 256], help='Number of units in hidden layers, also decide the number of layers')
    #parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate (1 - keep probability).')
    #parser.add_argument('--last_activation', type=str, default='none', help='Activation for the last layer')
    
    parser.add_argument('--loss_type', type=str, default='squared_error', help='Loss type') 
    parser.add_argument('--validation_size', type=float, default=0.1, help='Validation size') 
    
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='Weight decay for the optimizer.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--es_patience', type=int, default=15, help='Patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    parser.add_argument('--device', type=str, default="auto", help='Computation device.')
    #parser.add_argument('--num_hops', type=int, default=3, help='Number of hops to include in the graph.')

    # SetFormer options
    parser.add_argument('--d_model', type=int, default=256) # from 256 to 64
    parser.add_argument('--n_layers', type=int, default=8) # stayed 
    parser.add_argument('--n_heads', type=int, default=4) # stayed 
    parser.add_argument('--dropout', type=float, default=0.0)
    #parser.add_argument('--sf_use_distance_bias', action='store_true')
    #parser.add_argument('--sf_rbf_centers', type=int, default=16)
    #parser.add_argument('--sf_rbf_gamma', type=float, default=10.0)
    #parser.add_argument('--sf_use_fourier_feats',  type=bool, default=False)
    #parser.add_argument('--sf_fourier_num_freqs', type=int, default=8)
    #parser.add_argument('--sf_use_obs_y_as_feature', type=bool, default=True)
    parser.add_argument('--use_posenc', type=bool, default=False)
    #parser.add_argument('--cls_init', type=str, default='xavier', choices=['xavier', 'zero', 'normal'])
    parser.add_argument('--ffn_dim', type=int, default=256) # from 256 to 128
    parser.add_argument(
    "--neighbor_ratio",
    type=float,
    default=1.0,
    help="Fraction (0–1] of extra (non-train) points used when neighbors_train_only=False. "
         "For example, 0.25 means use only 25% of val/test points as additional neighbors."
    )
    parser.add_argument(
    "--neighbors_train_only",
    type=bool, default=False,
    help="If set, neighbors for val/test/calib are taken only from the training set. "
         "If unset, neighbors are drawn from both train and the current split."
)
    #args, unknowns = parser.parse_known_args()
    #args = parser.parse_args(custom_args)  # ← don't use sys.argv at all
    parser.add_argument('--train_file', type=str, default="n32_e035_1arc_v3_cropped_train.tiff")
    parser.add_argument('--valid_file', type=str, default="n32_e035_1arc_v3_cropped_val.tiff")
    parser.add_argument('--test_file', type=str, default="n32_e035_1arc_v3_cropped_test.tiff")
    parser.add_argument('--calib_file', type=str, default="n32_e035_1arc_v3__cropped_test.tiff")
    parser.add_argument('--keep_n_dict', type=str, default="train:0.005,valid:0.005,test:0.005,calib:0.000001") #Orit - I changes this!! from 0.005 to 0.05

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
