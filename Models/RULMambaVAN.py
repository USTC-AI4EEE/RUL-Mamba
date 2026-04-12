import math
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from pytorch_forecasting.models import BaseModel

from Models.MambaSimple import MambaBlock as Mamba

class MyGLU(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear_1 = nn.Linear(input_size, output_size)
        self.linear_2 = nn.Linear(input_size, output_size)
        self.glu = nn.GLU()

    def forward(self, x):
        a = self.linear_1(x)
        b = self.linear_2(x)
        return self.glu(torch.cat([a, b], dim=-1))

class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, external=False):
        super().__init__()

        self.shortcut = nn.Identity()

        if external:
            body_input_size = 2 * input_size
        else:
            body_input_size = input_size

        self.body = nn.Sequential(
            nn.Linear(body_input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout),
            MyGLU(input_size, input_size))

        self.norm = nn.LayerNorm(input_size)

    def forward(self, x, e=None):
        s = self.shortcut(x)

        if e is not None:
            x = torch.cat([x, e], dim=-1)

        x = self.body(x)
        y = self.norm(s + x)
        return y

class VarEncoder(nn.Module):
    def __init__(self, n_var, dim):
        super().__init__()
        self.n_var = n_var
        self.dim = dim
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(1, dim))
                                     for _ in range(n_var)])

    def forward(self, x):
        y = torch.zeros([x.shape[0], x.shape[1], self.n_var, self.dim], device=x.device)
        for i in range(self.n_var):
            y[:, :, i, :] = self.layers[i](x[:, :, i].unsqueeze(-1))
        return y

class DecoderBlock(nn.Module):
    def __init__(self,  d_inner,dt_rank,d_model,d_ff, dropout):
        super().__init__()

        self.shortcut = nn.Identity()

        self.mamba_dec = Mamba(d_inner=d_inner, dt_rank=dt_rank, d_model=d_model, d_ff=d_ff, d_conv=4)

        self.grn = GRN(d_model, 2 * d_model, dropout, external=True)

    def forward(self, x, c):
        if x is not None:
            x = torch.cat([x, c], dim=-2)
            s = self.shortcut(x)
            y = self.mamba_dec(x)
            y = self.grn(y, s)[:, 1:, :]
        else:
            s = self.shortcut(c)
            y = self.mamba_dec(c)
            y = self.grn(y, s)
     
        return y
    
class AttentionSE2D(nn.Module):
    def __init__(self, inchannel, ratio=2):
        super(AttentionSE2D, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
            b, c, h, w = x.size()
            y = self.gap(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU(alpha=1.0)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate_norm = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(output_size, 2 * output_size),
            nn.GLU(dim=-1),
            nn.LayerNorm(output_size)
        )

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.fc2(x)
        x = self.gate_norm(x)
        return x

class VariableAttentionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: list,
        hidden_size: int,
        dropout: float = 0.1,
        calc_sum=True
    ):
        super().__init__()
        self.calc_sum = calc_sum
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.dropout = dropout

        self.single_variable_grns = nn.ModuleList()
        self.prescalers = nn.ModuleList()
        for input_size in input_sizes:
            self.single_variable_grns.append(
                GatedResidualNetwork(
                    input_size,
                    min(input_size, hidden_size),
                    output_size=hidden_size,
                    dropout=dropout
                )
            )
            self.prescalers.append(nn.Linear(1, input_size))

        self.flattened_grn = GatedResidualNetwork(
            sum(input_sizes),
            min(hidden_size, len(input_sizes)),
            len(input_sizes),
            dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        var_outputs = []
        weight_inputs = []
        for i in range(len(self.input_sizes)):
            variable_embedding = x[..., i:i + 1]
            variable_embedding = self.prescalers[i](variable_embedding)
            weight_inputs.append(variable_embedding)
            var_outputs.append(self.single_variable_grns[i](variable_embedding))

        var_outputs = torch.stack(var_outputs, dim=-1)
        flat_embedding = torch.cat(weight_inputs, dim=-1)
        sparse_weights = self.flattened_grn(flat_embedding)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

        outputs = var_outputs * sparse_weights
        if self.calc_sum:
            outputs = outputs.sum(dim=-1)

        return outputs, sparse_weights

class RULMambaVAN(nn.Module):
    def __init__(self, enc_in, d_model, n_dec_layer, dropout, expand=2):
        super().__init__()

        self.emb_enc = VarEncoder(enc_in, d_model)

        self.encoder_VSN = VariableAttentionNetwork(input_sizes=[d_model]*enc_in,hidden_size=d_model,dropout=0.1,calc_sum=False)    
        
        self.se_attention = AttentionSE2D(d_model)

        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)
        self.d_ff = 2*d_model

        self.mamba_enc = Mamba(d_inner=self.d_inner, dt_rank=self.dt_rank, d_model=d_model, d_ff=self.d_ff)

        self.dec = nn.ModuleList([DecoderBlock(d_inner=self.d_inner, dt_rank=self.dt_rank, d_model=d_model, d_ff=self.d_ff,dropout=dropout) for _ in range(n_dec_layer)])

        self.proj = nn.Linear(d_model, 1)

    def forward(self, x_enc, x_dec):

        x_enc = x_enc.unsqueeze(-2)
        x_enc, _ = self.encoder_VSN(x_enc)
        x_enc = x_enc.squeeze(2)
        x_enc = rearrange(x_enc,'b l d m -> b d l m')
        x_enc = self.se_attention(x_enc)
        x_enc = rearrange(x_enc,'b d l m -> b l m d')
        x_enc = torch.sum(x_enc, dim=-2)

        enc_out = self.mamba_enc(x_enc)

        context = enc_out[:, -1:, :]

        for i in range(len(self.dec)):
            dec_out = self.dec[i](x_dec, context)

        out = self.proj(dec_out)

        return out


class RULMambaVANNetModel(BaseModel):
    def __init__(self, seq_len=24, pred_len=24, enc_in=11, c_out=1, d_model=16, n_dec_layer=2, dropout=0.01,
                 expand=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(**kwargs)

        self.network = RULMambaVAN(enc_in=enc_in, d_model=d_model, n_dec_layer=n_dec_layer, dropout=dropout, expand=expand)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_enc = x["encoder_cont"][:,:,:-1]

        prediction = self.network(x_enc=x_enc, x_dec=None)
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])

        return self.to_network_output(prediction=prediction)

if __name__=='__main__':
    N,L,C=128,10,1
    x_enc=torch.ones((N,L,C))
    x_mark_enc=None
    x_dec=None
    x_mark_dec=None
    model=RULMambaVAN(enc_in=1, d_model=16, n_dec_layer=2, dropout=0.01)
    out = model(x_enc=x_enc, x_dec=x_dec)
    print(out.shape)
