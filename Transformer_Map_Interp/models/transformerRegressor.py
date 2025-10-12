# =============================
# NEW FILE: transformer_model.py
# =============================
import torch
import torch.nn as nn


class TransformerCLSRegressor(nn.Module):
    """
    Decoder-only, no positional encodings.
    - Memory tokens (observed neighbors): shape (B, S, in_dim)
    - Target is a single learnable CLS token; model predicts a scalar elevation (normalized) for the CLS point.
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        cls_init: str = "xavier",  # one of: 'xavier', 'zero', 'normal'
        use_posenc: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.use_posenc = use_posenc

        # Project raw token features to model dimension
        self.in_proj = nn.Linear(in_dim, d_model)

        # Single CLS token (learnable) used as the decoder target sequence (len=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # OPTIONAL: learnable positional encodings for memory tokens (off by default)
        self.pos_emb = None

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, 1)

        self.reset_parameters(cls_init)

    def reset_parameters(self, how: str):
        if how == "zero":
            nn.init.zeros_(self.cls_token)
        elif how == "normal":
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:  # xavier
            nn.init.xavier_uniform_(self.cls_token)

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, mem_tokens: torch.Tensor, mem_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            mem_tokens: (B, S, in_dim) observed neighbor tokens (Δlat, Δlon, y_known, is_obs)
            mem_key_padding_mask: (B, S) True for PAD positions that should be masked
        Returns:
            y_hat: (B,) normalized elevation prediction for the CLS point
        """
        B, S, _ = mem_tokens.shape

        mem = self.in_proj(mem_tokens)  # (B, S, D)

        if self.use_posenc:
            if (self.pos_emb is None) or (self.pos_emb.size(1) < S):
                # allocate or grow a learnable pos_emb for memory tokens
                self.pos_emb = nn.Parameter(torch.zeros(1, S, self.d_model, device=mem.device))
                nn.init.trunc_normal_(self.pos_emb, std=0.02)
            mem = mem + self.pos_emb[:, :S, :]

        # expand CLS token for the batch; target sequence length = 1
        tgt = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # Run decoder: self-attn on tgt (degenerate; len=1), then cross-attn over mem
        dec = self.decoder(
            tgt=tgt,
            memory=mem,
            memory_key_padding_mask=mem_key_padding_mask,
        )  # (B, 1, D)

        y_hat = self.out(dec[:, 0, :]).squeeze(-1)  # (B,)
        return y_hat
    
