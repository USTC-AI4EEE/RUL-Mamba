import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from pytorch_forecasting.models import BaseModel
from typing import Dict

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


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res



def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len,pred_len,top_k,d_model,d_ff,num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, seq_len=24, label_len=0, pred_len=1, enc_in=7, dec_in=7, c_out=1, e_layers=2, d_layers=1, factor=3, d_model=16, d_ff=32, des='Exp', itr=1, top_k=5,embed='timeF',freq='h', dropout=0.1,num_kernels=6  ):
        super(TimesNet, self).__init__()
        self.task_name = 'TimesNet'
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([TimesBlock(seq_len,pred_len,top_k,d_model,d_ff,num_kernels)
                                    for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)

        self.predict_linear = nn.Linear(self.seq_len, self.pred_len+ self.seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc=None):
        '''
        
        :param x_enc:        torch.Size([32, 96, 7])
        :param x_mark_enc:   torch.Size([32, 96, 4])
        :param x_dec:        torch.Size([32, 144, 7])
        :param x_mark_dec:   torch.Size([32, 144, 4])
        :return: 
        '''
        # Normalization from Non-stationary Transformer   减均值除方差
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        target_mean = means[:, :, -1]
        target_stdev = stdev[:, :, -1]

        # embedding
        '''
        self.enc_embedding: 
        DataEmbedding(
                  (value_embedding): TokenEmbedding(
                    (tokenConv): Conv1d(7, 16, kernel_size=(3,), stride=(1,), padding=(1,), bias=False, padding_mode=circular)
                  )
                  (position_embedding): PositionalEmbedding()
                  (temporal_embedding): TimeFeatureEmbedding(
                    (embed): Linear(in_features=4, out_features=16, bias=False)
                  )
                  (dropout): Dropout(p=0.1, inplace=False)
                )
        '''
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]    x_mark_enc=None
        '''
        self.predict_linear:
        Linear(in_features=96, out_features=192, bias=True)
        enc_out :[100,96,16] -> [100,16,96]-> [100,16,192]->[100,192,16]
        
        '''
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        # TimesNet
        '''
        self.model[i]: 
        TimesBlock(
      (conv): Sequential(
        (0): Inception_Block_V1(
          (kernels): ModuleList(
            (0): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (3): Conv2d(16, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
            (4): Conv2d(16, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
            (5): Conv2d(16, 32, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5))
          )
        )
        (1): GELU(approximate=none)
        (2): Inception_Block_V1(
          (kernels): ModuleList(
            (0): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): Conv2d(32, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (3): Conv2d(32, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
            (4): Conv2d(32, 16, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
            (5): Conv2d(32, 16, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5))
          )
        )
        self.layer_norm: 
        LayerNorm((16,), eps=1e-05, elementwise_affine=True)
        '''
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        '''
        self.projection:
        Linear(in_features=16, out_features=7, bias=True)
        enc_out: [100, 97, 16]->  [100, 97, 1] 
        '''
        dec_out = self.projection(enc_out)      # torch.Size([100, 34, 16])-> torch.Size([100, 34, 1])
        dec_out=dec_out*target_stdev.unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)+target_mean.unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        return dec_out[:, -self.pred_len:, :]

class TimesNetModel(BaseModel):
    def __init__(self,seq_len=24, label_len=0, pred_len=1, enc_in=7, dec_in=7, c_out=1, e_layers=2, d_layers=1, factor=3, d_model=16, d_ff=32, des='Exp', itr=1, top_k=5,embed='timeF',freq='h', dropout=0.1,num_kernels=6, **kwargs):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = TimesNet(
            seq_len=seq_len, label_len=label_len, pred_len=pred_len, enc_in=enc_in, dec_in=dec_in, c_out=c_out, e_layers=e_layers, d_layers=d_layers, factor=factor,
            d_model=d_model, d_ff=d_ff, des=des, itr=itr, top_k=top_k, embed=embed, freq=freq, dropout=dropout, num_kernels=num_kernels
        )
    # 修改，锂电池预测
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_enc = x["encoder_cont"][:,:,:-1]  # torch.Size([100, 10, 9])
        # 输出
        prediction = self.network(x_enc)
        # 输出rescale， rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])

        # 返回一个字典，包含输出结果（prediction）
        return self.to_network_output(prediction=prediction)



if __name__=='__main__':
    N,L,C=100,96,5
    input=torch.ones((N,L,C))
    model=TimesNet(seq_len=L, enc_in=C, pred_len=1)
    out=model(input)
    print(out.shape)