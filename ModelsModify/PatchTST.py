import torch
from torch import nn
from ModelsModify.layers.Transformer_EncDec import Encoder, EncoderLayer
from ModelsModify.layers.SelfAttention_Family import FullAttention, AttentionLayer
from ModelsModify.layers.Embed import PatchEmbedding
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

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self,patch_len=16, stride=8,task_name='short_term_forecast',seq_len=96, label_len=48, pred_len=96, enc_in=7, dec_in=7, c_out=1, e_layers=2, d_layers=1, n_heads=8,factor=3,
                 d_model=16, d_ff=32, des='Exp', expand=2, d_conv=4, top_k=5, embed='timeF',freq='h', dropout=0.1,num_kernels=6,
                 moving_avg=25,channel_independence=1, decomp_method='moving_avg', use_norm=1,
                 version='fourier', mode_select='random', modes=32, activation='gelu',seasonal_patterns='Monthly',
                 inverse=False, mask_rate=0.25, anomaly_ratio=0.25,output_attention=False,down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
                seg_len=48, num_workers=0, itr=1, train_epochs=100, batch_size=32, patience=3, learning_rate=0.0001, loss='MSE',
                lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', p_hidden_dims=[128, 128],
                p_hidden_layers=2, use_dtw=False, augmentation_ratio=0, seed=2, jitter=False, scaling=False, permutation=False,
                randompermutation=False, magwarp=False, timewarp=False, windowslice=False, windowwarp=False, rotation=False,
                spawner=False, dtwwarp=False, shapedtwwarp=False, wdba=False, discdtw=False, discsdtw=False, extra_tag='',**kwargs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)

        # Encoder
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

        # Prediction Head
        self.head_nf = d_model * \
                       int((seq_len - patch_len) / stride + 2)

        self.head = FlattenHead(enc_in, self.head_nf, pred_len,
                                head_dropout=dropout)

        self.projection_final = nn.Linear(pred_len*enc_in, pred_len*c_out, bias=True)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out=dec_out[:, -self.pred_len:, :]  # [B, L, D]
        dec_out=self.projection_final(dec_out.reshape(dec_out.shape[0], -1))
        return dec_out


from pytorch_forecasting.models import BaseModel
from typing import Dict

class PatchTSTNetModel(BaseModel):
    def __init__(self,patch_len=6, stride=3,seq_len=24, pred_len=1, enc_in=7, c_out=1, e_layers=2, d_layers=1, factor=3,
                 d_model=16, d_ff=32, des='Exp', itr=1, top_k=5,embed='timeF',freq='h', dropout=0.1,num_kernels=6, **kwargs):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = PatchTST(
            patch_len=patch_len,stride=stride,seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, c_out=c_out, e_layers=e_layers, d_layers=d_layers, factor=factor,
            d_model=d_model, d_ff=d_ff, des=des, itr=itr, top_k=top_k, embed=embed, freq=freq, dropout=dropout, num_kernels=num_kernels
        )

    # 修改，锂电池预测
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_enc = x["encoder_cont"][:,:,:-1]
        # 输出
        prediction = self.network(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)
        # 输出rescale， rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])

        # 返回一个字典，包含输出结果（prediction）
        return self.to_network_output(prediction=prediction)


if __name__=='__main__':
    N,L,C=100,96,15
    label_len = 16
    c_out = 1
    pred_len=16
    x_enc=torch.ones((N,L,C))
    # x_mark_enc=torch.ones((N, L, 4))
    # x_dec = torch.ones((N, pred_len, C))
    # x_mark_dec=torch.ones((N, pred_len, 4))
    model=PatchTST(seq_len=L, enc_in=C, dec_in=C, label_len = label_len, pred_len=pred_len, c_out=1)              # pred_len 被限制了
    out=model(x_enc=x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)
    print(out.shape)