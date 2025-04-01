## RUL-Mamba: Mamba-Based Remaining Useful Life Prediction for Lithium-Ion Batteries

> **Authors:**
Jiahui Huang, Lei Liu, Hongwei Zhao, Tianqi Li, Bin Li.

This repo contains the code and data from our paper published in Journal of Energy Storage [RUL-Mamba: Mamba-Based Remaining Useful Life Prediction for Lithium-Ion Batteries](Paper: https://www.sciencedirect.com/science/article/pii/S2352152X25010898?dgcid=author).

## 1. Abstract

Lithium-ion batteries play a crucial role in the fields of renewable energy and electric vehicles. Accurately predicting their Remaining Useful Life (RUL) is essential for ensuring safe and reliable operation. However, achieving precise RUL predictions poses significant challenges due to the complexities of degradation mechanisms and the impact of operational noise, particularly the capacity regeneration phenomenon. To address these issues, we propose a lithium-ion battery RUL prediction model named RUL-Mamba, which is based on the Mamba-Feature Attention Network (FAN)-Gated Residual Network (GRN). This model employs an encoder-decoder architecture that effectively integrates the Mamba module, FAN network, and GRN network. Mamba demonstrates superior temporal representation capabilities alongside efficient inference properties. The constructed FAN network leverages a feature attention mechanism to efficiently extract key features at each time step, enabling the Mamba block in the encoder to effectively capture information related to capacity regeneration from historical capacity sequences. The designed GRN network adaptively processes the decoded features output by the Mamba blocks in the decoder through a gating mechanism, accurately modeling the nonlinear mapping relationship between the decoded feature vector and the prediction target. Compared to state-of-the-art (SOTA) time series forecasting models on three battery degradation datasets from NASA, Oxford and Tongji University, the proposed model not only achieves SOTA predictive performance across various prediction starting points, with a maximum accuracy improvement of 42.5% over existing models, but also offers advantages such as efficient training, fast inference and being less influenced by the prediction starting point.

## 2.Requirements

The version of python is 3.10.13 .
```bash
numpy==1.21.6
numba==0.55.1
matplotlib==3.3.4
scipy==1.8.0
statsmodels==0.13.5
pytorch-lightning==1.9.5
pytorch-forecasting==0.10.3
sympy==1.12.1
reformer_pytorch==1.4.4
openpyxl==3.1.5
einops==0.8.0
```

## 3.Datasets

The TJU dataset is already placed in the datasets folder. The URL of TJU dataset is as follows:

TJU dataset: https://github.com/wang-fujin/PINN4SOH/tree/main/data/TJU%20data/Dataset_3_NCM_NCA_battery.

## 4.Usage

- an example for train and evaluate a new model：

```bash
python RUL_Prediction_RULMambaVAN.py
```

- You can get the following output:
    
```bash
['encoder_cont']: torch.Size([128, 64, 18])
['decoder_cont']: torch.Size([128, 1, 18])
y: torch.Size([1])
model name:RULMambaVAN

selected battery name:CY25_1, start point:200

train dataset:1472,val:184,test:751
Input Feature num:18 ,name:['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy', 'current mean', 'current std', 'current kurtosis', 'current skewness', 'CV Q', 'CV charge time', 'current slope', 'current entropy', 'Capacity', 'target']
time_varying_unknown_reals num:1,decoder num:17 ,name:['target']
model name:RULMambaVAN

Number of parameters in network(参数数量(Params)): 44.7k
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [2]

| Name            | Type        | Params
------------------------------------------------
0 | loss            | MAE         | 0     
1 | logging_metrics | ModuleList  | 0     
2 | network         | RULMambaVAN | 44.7 K
------------------------------------------------
44.7 K    Trainable params
0         Non-trainable params
44.7 K    Total params
0.179     Total estimated model params size (MB)
Epoch 32: 100%|████████████████████████████████████████████████████████████████████| 13/13 [00:02<00:00,  5.55it/s, loss=0.00093, train_loss_step=0.000929, val_loss=0.000709, train_loss_epoch=0.000945]
best_model_path: results_TJU_RUL_prediction_sl_64/CY25_1/RULMambaVAN/SP200/Exp1/RULMambaVANNetModel/checkpoints/epoch=32-step=330-v1.ckpt                                                                
best_model_path:results_TJU_RUL_prediction_sl_64/CY25_1/RULMambaVAN/SP200/Exp1/RULMambaVANNetModel/checkpoints/epoch=32-step=330-v1.ckpt <_io.TextIOWrapper name='results_TJU_RUL_prediction_sl_64/CY25_1/RULMambaVAN/SP200/Exp1/log_Feas_18_18_in_l_64_out_l_1_Pcap.txt' mode='w' encoding='UTF-8'>
第1次实验运行的Epoch数量: 33  

第1次实验运行的训练时间: 79.21130394935608  

第1次实验运行的推理时间: 0.5797321796417236  

第1次实验的结果: 
MAE:0.0015, RMSE:0.0022, r2:0.9998, RUL_real:579, RUL_pred:581, AE:2, RE:0.0035  

第1次实验运行的Epoch数量: 33  

第1次实验运行的训练时间: 79.21130394935608  

第1次实验运行的推理时间: 0.5797321796417236  

第1次实验的结果: 
MAE:0.0015, RMSE:0.0022, r2:0.9998, RUL_real:579, RUL_pred:581, AE:2, RE:0.0035  

第1次实验模型的保存路径:results_TJU_RUL_prediction_sl_64/CY25_1/RULMambaVAN/SP200/Exp1/RULMambaVANNetModel/checkpoints/epoch=32-step=330-v1.ckpt
```

## 5.Acknowledgments

Work&Code is inspired by https://github.com/USTC-AI4EEE/PatchFormer.

## 6.Citation

If you find our work useful in your research, please consider citing:

```latex
@article{HUANG2025116376,
    title = {RUL-Mamba: Mamba-based remaining useful life prediction for lithium-ion batteries},
    journal = {Journal of Energy Storage},
    volume = {120},
    pages = {116376},
    year = {2025},
    issn = {2352-152X},
    doi = {https://doi.org/10.1016/j.est.2025.116376},
    url = {https://www.sciencedirect.com/science/article/pii/S2352152X25010898},
    author = {Jiahui Huang and Lei Liu and Hongwei Zhao and Tianqi Li and Bin Li},
}
```

If you have any problems, contact me via liulei13@ustc.edu.cn.