# models/setformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# Utilities
# ------------------------

def fourier_features(x, num_freqs=8):
    """
    x: [B, N, 2] in [0,1] range
    Returns: [B, N, 4*num_freqs]
    """
    B, N, D = x.shape
    assert D == 2
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device).float() * math.pi
    # [num_freqs] -> [1,1,num_freqs]
    freqs = freqs.view(1, 1, -1)
    # expand coords to [B,N,num_freqs,2]
    xf = x.unsqueeze(-2) * freqs.unsqueeze(-1)
    # sin/cos for each dim
    s = torch.sin(xf)
    c = torch.cos(xf)
    # concat on the last dim: [..., 2] -> [..., 2*num_freqs]
    emb = torch.cat([s, c], dim=-1).reshape(B, N, -1)
    return emb

class DistanceBias(nn.Module):
    """Turns pairwise distances into an attention logit bias via RBF + linear head per attention head."""
    def __init__(self, rbf_centers=16, rbf_gamma=10.0, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.gamma = rbf_gamma
        self.register_buffer("centers", torch.linspace(0, 1, rbf_centers)[None, None, :])  # [1,1,C]
        self.proj = nn.Linear(rbf_centers, n_heads)

    def forward(self, coords):  # coords: [B, N, 2] normalized
        B, N, _ = coords.shape
        diff = coords[:, :, None, :] - coords[:, None, :, :]  # [B, N, N, 2]
        dist = torch.sqrt(torch.clamp((diff ** 2).sum(-1), min=1e-12))  # [B, N, N]
        rbf = torch.exp(-self.gamma * (dist[..., None] - self.centers) ** 2)  # [B,N,N,C]
        bias_per_head = self.proj(rbf)  # [B,N,N,H]
        return bias_per_head.permute(0, 3, 1, 2).reshape(B * self.n_heads, N, N)

class NonCausalDecoderBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, mlp_ratio=4.0, p_drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(p_drop),
        )

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        # attn_mask can be [B*H, N, N]; nn.MultiheadAttention will broadcast per batch*head.
        y = self.attn(
            query=self.ln1(x), key=self.ln1(x), value=self.ln1(x),
            attn_mask=attn_bias, key_padding_mask=key_padding_mask
        )[0] # For attn_bias = None we will get a non casual decoder
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x

class SetFormer(nn.Module):
    """
    Decoder-only, non-causal Transformer for set inputs.

    Inputs per batch:
      - coords: [B,N,2] in [0,1] (lat, lon normalized)
      - feats:  [B,N,F] additional features (optional, can be zero-dim)
      - obs_mask:   [B,N] boolean. True = position has ground-truth y available (observed).
      - query_mask: [B,N] boolean. True = position is a query location we evaluate loss on.

    If use_obs_y_as_feature=True, then we concatenate y to feats **only for observed tokens** during training.
    This mimics kriging-like conditioning without leaking target at query sites.
    """
    def __init__(self,
                 in_feat_dim=0,
                 d_model=256,
                 depth=4,
                 n_heads=4,
                 p_drop=0.1,
                 use_distance_bias=True,
                 rbf_centers=16,
                 rbf_gamma=10.0,
                 use_fourier_feats=False,
                 fourier_num_freqs=8,
                 use_obs_y_as_feature=False):
        super().__init__()
        self.use_distance_bias = use_distance_bias
        self.use_fourier_feats = use_fourier_feats
        self.use_obs_y_as_feature = use_obs_y_as_feature
        self.fourier_num_freqs = fourier_num_freqs

        # feature input dimension = F (+ optional Fourier features of coords) (+ optional y_obs)
        ff_dim = in_feat_dim
        if self.use_fourier_feats:
            ff_dim += 4 * fourier_num_freqs
        # y_obs is concatenated only at runtime for observed tokens
        self.embed = nn.Sequential(
            nn.Linear(3 , d_model),  # 2 for coords  + ff_dim id using features
            nn.GELU(),
            nn.Dropout(p_drop),
        )

        if self.use_distance_bias:
            self.bias = DistanceBias(rbf_centers=rbf_centers, rbf_gamma=rbf_gamma, n_heads=n_heads)
        else:
            self.bias = None

        self.blocks = nn.ModuleList([
            NonCausalDecoderBlock(d_model=d_model, n_heads=n_heads, p_drop=p_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, coords, feats, y=None, obs_mask=None, query_mask=None, key_padding_mask=None):
        # coords: [B,N,2] in [0,1]
        # feats:  [B,N,F] or None
        B, N, _ = coords.shape
        if feats is None:
            feats = coords.new_zeros(B, N, 0)

        x_parts = [coords]
        # Optional Fourier features of coords
        if self.use_fourier_feats:
            x_parts.append(fourier_features(coords, num_freqs=self.fourier_num_freqs))
        # User-supplied features
        #No Feature Now
        #if feats.size(-1) > 0:
        #    x_parts.append(feats)

        x = torch.cat(x_parts, dim=-1)  # [B,N,2+F(+fourier)]

        # Optional: concatenate y **only at observed positions** during training
        if self.use_obs_y_as_feature and (y is not None) and (obs_mask is not None):
            y_feat = torch.zeros_like(x[..., :1])  # [B,N,1]
            y_feat[obs_mask] = y[obs_mask]
            x = torch.cat([x, y_feat], dim=-1)

        h = self.embed(x)
        attn_bias = self.bias(coords) if self.bias is not None else None
        for blk in self.blocks:
            h = blk(h, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        h = self.norm(h)
        pred = self.head(h).squeeze(-1)  # [B,N]
        return pred
