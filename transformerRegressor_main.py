import numpy as np
import torch
from transformerRegressor_run_experiment import run_transformer
from Transformer_Map_Interp.argument_transformer import parse_opt
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["TENSORBOARD_NO_TF"] = "1"      # tell TB to use its 'notf' path
if __name__ == "__main__":
    
    args = parse_opt()
    # Select Transformer model & typical hyperparams
    args.model = "transformer"
    # Example overrides (adjust as you like):
    # args.dataset = "n32_e035_1arc_v3_cropped"
    # args.n_neighbors = 32
    # args.d_model = 128
    # args.nhead = 8
    # args.num_layers = 4
    # args.ffn_dim = 256
    args.dropout = 0.0
    #the largs.weight_decay = 0.0
    # args.cls_init = "xavier"
    # args.use_posenc = False
    # args.batch_size = 64
    args.max_km = 0.75
    args.epochs = 30
    args.new_spread = False
    
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    save_dir = getattr(args, "save_path",
                   f"saved_models/transformer_{getattr(args, 'dataset','dataset')}")
    print(save_dir)
    tb_dir = os.path.join(save_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset}")
    #print(f"Neighbors k: {args.n_neighbors}")

    test_mse, test_mae = run_transformer(args, tb_writer=writer)
    print(f"Done. Test MSE={test_mse:.4f}, MAE={test_mae:.4f}")
    #writer.close()
