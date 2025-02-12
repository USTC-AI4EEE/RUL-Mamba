import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

#  from .layers.Embed import DataEmbedding
# import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars



class ResidualBlock(nn.Module):
    def __init__(self, d_inner, dt_rank, d_model=16, d_ff=32,d_conv=4, top_k=5,):
        super(ResidualBlock, self).__init__()
        
        self.mixer = MambaBlock(d_inner, dt_rank, d_model=d_model, d_ff=d_ff,d_conv=d_conv, top_k=top_k)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

class MambaBlock(nn.Module):
    def __init__(self, d_inner, dt_rank, d_model=16, d_ff=32,d_conv=4, top_k=5,):
        super(MambaBlock, self).__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels = self.d_inner,
            out_channels = self.d_inner,
            bias = True,
            kernel_size = d_conv,
            padding = d_conv - 1,
            groups = self.d_inner
        )

        # takes in x and outputs the input-specific delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_ff * 2, bias=False)

        # projects delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_ff + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Figure 3 in Section 3.4 in the paper
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x) # [B, L, 2 * d_inner]
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output


    def ssm(self, x):
        """
        Algorithm 2 in Section 3.2 in the paper
        """
        
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float()) # [d_in, n]
        D = self.D.float() # [d_in]

        x_dbl = self.x_proj(x) # [B, L, d_rank + 2 * d_ff]
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1) # delta: [B, L, d_rank]; B, C: [B, L, n]
        delta = F.softplus(self.dt_proj(delta)) # [B, L, d_in]
        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n")) # A is discretized using zero-order hold (ZOH) discretization
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n") # B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors: "A is the more important term and the performance doesn't change much with the simplification on B"

        # selective scan, sequential instead of parallel
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1) # [B, L, d_in]
        y = y + u * D

        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class MambaSimple(nn.Module):
    """
    Mamba, linear-time sequence modeling with selective state spaces O(L)
    Paper link: https://arxiv.org/abs/2312.00752
    Implementation refernce: https://github.com/johnma2006/mamba-minimal/
    """

    def __init__(self, task_name='short_term_forecast',seq_len=96, label_len=48, pred_len=96, enc_in=7, dec_in=7, c_out=1, e_layers=2, d_layers=1, n_heads=8,factor=3,
                 d_model=16, d_ff=32, des='Exp', expand=2, d_conv=4, top_k=5, embed='timeF',freq='h', dropout=0.1,num_kernels=6,
                 moving_avg=25,channel_independence=1, decomp_method='moving_avg', use_norm=1,
                 version='fourier', mode_select='random', modes=32, activation='gelu',seasonal_patterns='Monthly',
                 inverse=False, mask_rate=0.25, anomaly_ratio=0.25,output_attention=False,down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
                seg_len=48, num_workers=0, itr=1, train_epochs=100, batch_size=32, patience=3, learning_rate=0.0001, loss='MSE',
                lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', p_hidden_dims=[128, 128],
                p_hidden_layers=2, use_dtw=False, augmentation_ratio=0, seed=2, jitter=False, scaling=False, permutation=False,
                randompermutation=False, magwarp=False, timewarp=False, windowslice=False, windowwarp=False, rotation=False,
                spawner=False, dtwwarp=False, shapedtwwarp=False, wdba=False, discdtw=False, discsdtw=False, extra_tag='',
                 **kwargs):
        super(MambaSimple, self).__init__()
        self.task_name = task_name
        self.pred_len = pred_len

        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        self.embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)

        self.layers = nn.ModuleList([ResidualBlock(self.d_inner, self.dt_rank,d_model=d_model, d_ff=d_ff,d_conv=d_conv, top_k=top_k) for _ in range(e_layers)])
        self.norm = RMSNorm(d_model)

        self.out_layer = nn.Linear(d_model, c_out, bias=False)
        self.projection_final = nn.Linear(pred_len*enc_in, pred_len*c_out, bias=True)
    # def short_term_forecast(self, x_enc, x_mark_enc):
    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    # def long_term_forecast(self, x_enc, x_mark_enc):
    #     x = self.embedding(x_enc, x_mark_enc)
    #     for layer in self.layers:
    #         x = layer(x)

    #     x = self.norm(x)
    #     x_out = self.out_layer(x)
    #     return x_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            x_out=x_out[:, -self.pred_len:, :]
            x_out = self.projection_final(x_out.view(x_out.shape[0], -1))
            return x_out

        # other tasks not implemented

from pytorch_forecasting.models import BaseModel
from typing import Dict

class MambaSimpleNetModel(BaseModel):
    def __init__(self,seq_len=24, label_len=0, pred_len=1, enc_in=7, dec_in=7, c_out=1, e_layers=2, d_layers=1, factor=3,
                 d_model=16, d_ff=32, des='Exp', itr=1, top_k=5,embed='timeF',freq='h', dropout=0.1,num_kernels=6, **kwargs):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = MambaSimple(
            seq_len=seq_len, label_len=label_len, pred_len=pred_len, enc_in=enc_in, dec_in=dec_in, c_out=c_out, e_layers=e_layers, d_layers=d_layers, factor=factor,
            d_model=d_model, d_ff=d_ff, des=des, itr=itr, top_k=top_k, embed=embed, freq=freq, dropout=dropout, num_kernels=num_kernels
        )


    # 修改，锂电池预测
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_enc = x["encoder_cont"][:,:,:-1]
        # 输出
        prediction = self.network(x_enc, x_mark_enc=None)
        # 输出rescale， rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])

        # 返回一个字典，包含输出结果（prediction）
        return self.to_network_output(prediction=prediction)


if __name__=='__main__':
    N,L,C=100,24,11
    label_len = 0
    c_out = 1
    pred_len=1
    x_enc=torch.ones((N,L,C))
    x_mark_enc=torch.ones((N, L, 4))
    x_dec = torch.ones((N, pred_len, C))
    x_mark_dec=torch.ones((N, pred_len, 4))
    model=MambaSimple(seq_len=L, enc_in=C, dec_in=C, label_len = label_len, pred_len=pred_len, c_out=1)              # pred_len 被限制了
    out=model(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
    print(out.shape)