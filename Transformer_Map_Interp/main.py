# Main for the CP part
import numpy as np
import torch
from argument_transformer import parse_opt 
#from run_transformer import run_transformer
from new_run_trandformer import run_transformer

if __name__ == "__main__":

    args = parse_opt()
    args.dataset = "n32_e035_1arc_v3_cropped"
    args.new_spread = True
    print(args)

    # set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # run experiment on one train-test split
    run_transformer(args)