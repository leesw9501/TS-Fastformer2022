import torch
from torch import nn
import torch.nn.functional as F
import math


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout):
        super().__init__()
        self.conv1 = SamePadConv(in_channels=hidden_dim, out_channels=pf_dim, dilation = 2, kernel_size=3)
        self.conv2 = SamePadConv(in_channels=pf_dim, out_channels=hidden_dim, dilation = 2, kernel_size=3)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(nn.functional.gelu(self.conv1(x.transpose(-1,1))))
        x = self.conv2(x).transpose(-1,1)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = n_heads, 
                                                    dropout=dropout, batch_first=True, device=device)        
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = n_heads, 
                                                    dropout=dropout, batch_first=True, device=device)
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_dim)
        
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim=hidden_dim, pf_dim=pf_dim,  dropout=dropout)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src):
        # trg: [batch_size, trg_len, hidden_dim]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        # past short-term multihead attention
        _trg, _ = self.self_attention(trg, trg, trg)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg: [batch_size, trg_len, hidden_dim]

        # past long short-term cross multihead attention
        _trg, attention = self.cross_attention(trg, enc_src, enc_src)

        # dropout, residual connection and layer norm
        trg = self.cross_attn_layer_norm(trg + self.dropout(_trg))
        # trg: [batch_size, trg_len, hidden_dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        #print('asssd', _trg.shape)
        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        return trg, attention


class PastAttentionDecoder(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        

        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])


    def forward(self, trg, enc_src):

        for layer in self.layers:
            trg, attention = layer(trg, enc_src)
        return trg, attention
