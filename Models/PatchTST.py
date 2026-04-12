from typing import Dict

import torch
from torch import nn
from pytorch_forecasting.models import BaseModel

from Models.Layers.Embed import PatchEmbedding
from Models.Layers.Self_Attention_Family import AttentionLayer, FullAttention
from Models.Layers.Transformer_Enc_Dec import Encoder, EncoderLayer
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, patch_len=16, stride=8, task_name='short_term_forecast', seq_len=96, pred_len=96, enc_in=7, c_out=1,
                 e_layers=2, n_heads=8, factor=3, d_model=16, d_ff=32, dropout=0.1, activation='gelu',
                 output_attention=False, **kwargs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        padding = stride

        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        )

        self.head_nf = d_model * \
                       int((seq_len - patch_len) / stride + 2)

        self.head = FlattenHead(enc_in, self.head_nf, pred_len,
                                head_dropout=dropout)

        self.projection_final = nn.Linear(pred_len*enc_in, pred_len*c_out, bias=True)
    
    def forward(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out=dec_out[:, -self.pred_len:, :]
        dec_out=self.projection_final(dec_out.reshape(dec_out.shape[0], -1))
        return dec_out

class PatchTSTNetModel(BaseModel):
    def __init__(self, patch_len=6, stride=3, task_name='short_term_forecast', seq_len=24, pred_len=1, enc_in=7, c_out=1,
                 e_layers=2, n_heads=8, factor=3, d_model=16, d_ff=32, dropout=0.1, activation='gelu',
                 output_attention=False, **kwargs):
        self.save_hyperparameters()
        super().__init__(**kwargs)
        self.network = PatchTST(
            patch_len=patch_len, stride=stride, task_name=task_name, seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
            c_out=c_out, e_layers=e_layers, n_heads=n_heads, factor=factor, d_model=d_model, d_ff=d_ff,
            dropout=dropout, activation=activation, output_attention=output_attention
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_enc = x["encoder_cont"][:,:,:-1]
        prediction = self.network(x_enc)
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])

        return self.to_network_output(prediction=prediction)

if __name__=='__main__':
    N,L,C=100,96,15
    label_len = 16
    c_out = 1
    pred_len=16
    x_enc=torch.ones((N,L,C))
    model=PatchTST(seq_len=L, pred_len=pred_len, enc_in=C, c_out=1)
    out = model(x_enc=x_enc)
    print(out.shape)
