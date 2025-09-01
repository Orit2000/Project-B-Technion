import torch
import numpy as np
from torch.utils.data import DataLoader
from models.setformer import SetFormer
import torch.nn as nn

from datasets.TransformerDataset import TransformerDataset
from datasets.dt2_data import load_dt2_data



def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def run_transformer(args):
    print(f"Loading dataset: {args.dataset}")
    trainset, validset, testset, calibset = load_dt2_data(args)
    print(f"Train: {len(trainset)}, Val: {len(validset)}, Test: {len(testset)}")

    # Dataloaders
    train_loader = DataLoader(
        TransformerDataset(trainset, args, max_obs_per_sample=args.max_obs),
        batch_size=args.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        TransformerDataset(validset, args, max_obs_per_sample=args.max_obs),
        batch_size=args.batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_feat_dim = trainset.features.shape[1]

    model = SetFormer(
        in_feat_dim=in_feat_dim,
        d_model=args.sf_d_model,
        depth=args.sf_depth,
        n_heads=args.sf_heads,
        p_drop=args.sf_drop,
        use_distance_bias=args.sf_use_distance_bias,
        rbf_centers=args.sf_rbf_centers,
        rbf_gamma=args.sf_rbf_gamma,
        use_fourier_feats=args.sf_use_fourier_feats,
        fourier_num_freqs=args.sf_fourier_num_freqs,
        use_obs_y_as_feature=args.sf_use_obs_y_as_feature,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss(reduction='mean')

    def evaluate(loader):
        model.eval()
        se_sum, ae_sum, n_points = 0.0, 0.0, 0
        with torch.no_grad():
            for coords, feats, y, obs, qry, pad in loader:
                coords, y, obs, qry, pad = coords.to(device), y.to(device), obs.to(device), qry.to(device), pad.to(device)
                feats = feats.to(device) if feats is not None else None
                pred = model(coords, feats, y=y, obs_mask=obs, query_mask=qry, key_padding_mask=pad)
                q_mask = qry & (~pad)
                se_sum += ((pred[q_mask] - y[q_mask]) ** 2).sum().item()
                ae_sum += (pred[q_mask] - y[q_mask]).abs().sum().item()
                n_points += q_mask.sum().item()
        return (se_sum / max(1, n_points)) ** 0.5, ae_sum / max(1, n_points)

    for epoch in range(args.epochs):
        model.train()
        for coords, feats, y, obs, qry, pad in train_loader:
            coords, y, obs, qry, pad = coords.to(device), y.to(device), obs.to(device), qry.to(device), pad.to(device)
            feats = feats.to(device) if feats is not None else None
            pred = model(coords, None, y=y, obs_mask=obs, query_mask=qry, key_padding_mask=pad)
            q_mask = qry & (~pad)
            loss = mse_loss(pred[q_mask], y[q_mask])
            opt.zero_grad(); loss.backward(); opt.step()

        rmse, mae = evaluate(valid_loader)
        print(f"Epoch {epoch:02d}: Valid RMSE={rmse:.4f} | MAE={mae:.4f}")

