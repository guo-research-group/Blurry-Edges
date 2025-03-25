import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, stride, device = None):
        super().__init__()
        d_model_half = int(d_model / 2)
        position = torch.linspace(0, (max_len-1)*stride, max_len)
        pe = torch.zeros((max_len, max_len, d_model))
        div_term = torch.exp(torch.arange(0, d_model_half, 2) * (-2 * math.log(10000.0) / d_model)).unsqueeze(0).unsqueeze(0)
        pe[:, :, 0:d_model_half:2] = torch.sin(position.unsqueeze(1).unsqueeze(1) * div_term)
        pe[:, :, 1:d_model_half:2] = torch.cos(position.unsqueeze(1).unsqueeze(1) * div_term)
        pe[:, :, d_model_half:d_model:2] = torch.sin(position.unsqueeze(0).unsqueeze(2) * div_term)
        pe[:, :, d_model_half+1:d_model:2] = torch.cos(position.unsqueeze(0).unsqueeze(2) * div_term)
        self.pe = pe.flatten(start_dim=0, end_dim=1).unsqueeze(0).to(device)
    def forward(self, x):
        x += self.pe[:, :x.size(1), :]
        return x

class GlobalStage(nn.Module):
    def __init__(self, max_len=64, stride=2, in_parameter_size=38, out_parameter_size=12, d_model=128, nhead=8, num_encoder_layers=8, \
                    dim_feedforward=256, layer_norm_eps=1e-5, batch_first=True, bias=True, device=None):
        super().__init__()
        self.in_src_projection = nn.Linear(in_features=in_parameter_size, out_features=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, stride=stride, device=device)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1,
                                                    activation=F.relu, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=False,
                                                    bias=bias, device=device)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.generator = nn.Linear(in_features=d_model, out_features=out_parameter_size)
    def forward(self, src):
        src_emb = self.positional_encoding(self.in_src_projection(src))
        outs = self.encoder(src_emb)
        outs = self.generator(outs)
        return outs