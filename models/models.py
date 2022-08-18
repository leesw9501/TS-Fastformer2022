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
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout):
        super().__init__()
        self.conv1 = SamePadConv(in_channels=hidden_dim, out_channels=pf_dim, dilation = 2, kernel_size=3)
        self.conv2 = SamePadConv(in_channels=pf_dim, out_channels=hidden_dim, dilation = 2, kernel_size=3)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        x = self.dropout(nn.functional.gelu(self.conv1(x.transpose(-1,1))))
        # x: [batch_size, seq_len, pf_dim]
        x = self.conv2(x).transpose(-1,1)
        # x: [batch_size, seq_len, hidden_dim]

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
        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]
        return trg, attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LearningRepresentation(nn.Module):
    def __init__(self, c_in, d_model):
        super(LearningRepresentation, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x


class DataRepresentation(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataRepresentation, self).__init__()

        self.learning_representation = LearningRepresentation(c_in=c_in, d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        x = self.learning_representation(x) + self.positional_encoding(x)

        return self.dropout(x)