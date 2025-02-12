import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from ModelsModify.layers.AMS import AMS
from ModelsModify.layers.Layer import WeightGenerator, CustomLinear
from ModelsModify.layers.RevIN import RevIN
from functools import reduce
from operator import mul

from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric as LightningMetric

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, EncoderNormalizer, MultiNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nn import LSTM, MultiEmbedding


from pytorch_forecasting.utils import autocorrelation, create_mask, detach, integer_histogram, padded_stack, to_list

from pytorch_forecasting.models import BaseModel
from typing import Dict


import torch.nn.functional as F
from einops import rearrange

class PathFormer(nn.Module):
    def __init__(self,
                 layer_nums=3, 
                 num_nodes=15, 
                 pred_len=1, 
                 seq_len=24, 
                 k=3, 
                 num_experts_list=[4, 4, 4], 
                 patch_size_list=[8,6,4,2], #能整除seq_len,且长度等于num_experts
                 d_model=16, 
                 d_ff=64, 
                 residual_connection=True, 
                 revin=1,
                 gpu=0
                 ):
        super(PathFormer, self).__init__()
        self.layer_nums = layer_nums  # 设置pathway的层数
        self.num_nodes = num_nodes
        self.pre_len = pred_len
        self.seq_len = seq_len
        self.k = k
        self.num_experts_list = num_experts_list
        self.patch_size_list = patch_size_list
        self.d_model = d_model
        self.d_ff = d_ff
        self.residual_connection = residual_connection
        self.revin = revin
        self.gpu = gpu
        
        if self.revin:
            self.revin_layer = RevIN(num_features=self.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(self.gpu))

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list, noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, residual_connection=self.residual_connection))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:

        #balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))


        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, _ = layer(out)
            #balance_loss += aux_loss

        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')
        #print("Out shape:",out.shape)
        out = out[:,:,-1].view(out.shape[0],self.pre_len)
        
        return out
        
class PathFormerModel(BaseModel):
    def __init__(self,
                 enc_in:int,
                 seq_len:int,
                 pred_len:int,
                 k:int,
                 patch_size_list:list,
                 **kwargs):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()

        super().__init__(**kwargs)
        self.network = PathFormer(
            num_nodes=self.hparams.enc_in,
            seq_len=self.hparams.seq_len,
            pred_len=self.hparams.pred_len,
            k=self.hparams.k, 
            patch_size_list=self.hparams.patch_size_list, 
        )

    # 修改，锂电池预测
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_enc = x["encoder_cont"][:,:,:-1]
        # 输出
        prediction = self.network(x_enc)
        # 输出rescale， rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])

        # 返回一个字典，包含输出结果（prediction）
        return self.to_network_output(prediction=prediction)