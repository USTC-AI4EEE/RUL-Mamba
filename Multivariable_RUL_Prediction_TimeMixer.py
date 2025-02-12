# -*- coding: utf-8 -*-
import os
from assistant import get_gpus_memory_info
id,_ = get_gpus_memory_info()
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from shutil import copyfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder,EncoderNormalizer,MultiNormalizer,TorchNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
import scipy.io
from sklearn.preprocessing import MinMaxScaler
# from pytorch_forecasting.data import GroupNormalizer
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('agg')
from datetime import datetime
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='TimeMixer',help='Model name.')
parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length') 
parser.add_argument('--Battery_list', type=list, default=['CY25_1', 'CY25_2', 'CY25_3'], help='Battery data')
parser.add_argument('--data_dir', type=str, default='data/TJU data/Dataset_3_NCM_NCA_battery/', help='path of the data file')
parser.add_argument('--Rated_Capacity', type=float, default=2.5, help='Rate Capacity')
parser.add_argument('--test_name', type=str, default='CY25_1', help='Battery data used for test')
parser.add_argument('--start_point_list', type=int, default=[200,300,400], help='The cycle when prediction gets started.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--root_dir', type=str, default='TJU_RUL_prediction_sl_64', help='root path of the store file')
parser.add_argument('--count', type=int, default=10, help='The number of independent experiment.')
parser.add_argument('--batch_size', type=int, default=128, help='The batch size.')
parser.add_argument('--max_epochs', type=int, default=200, help='max train epochs') 
args = parser.parse_args()

def rul_value_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
            
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    rul_real = true_re + 1
    rul_pred = pred_re + 1
    ae_error = abs(true_re - pred_re)        
    re_score = abs(true_re - pred_re)/true_re
    if re_score > 1: re_score = 1
        
    return rul_real,rul_pred,ae_error,re_score

# 打印日志到文件
def print_log(print_string, log_file, visible=True):
    if visible:
        print("{}".format(print_string))
    # 写入日志文件
    log_file.write('{}\n'.format(print_string))
    # 刷新缓存区，将数据写入
    log_file.flush()

from DataPreProcess import MultiVariateBatteryDataProcess,BatteryDataProcess

#------------------------------------------------- step 1: 数据准备 ----------------------------------------
# -------------------------------数据分析和数据预处理【至关重要】---------------------------------------
BatteryData = np.load('data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy', allow_pickle=True)
BatteryData = BatteryData.item()

_,_,df_all = MultiVariateBatteryDataProcess(BatteryData,args.test_name,args.start_point_list[0],args)
real_data = df_all['target'].values*args.Rated_Capacity
all_pred_data_list = []

root_dir = 'results_{}/{}/{}/'.format(args.root_dir,args.test_name,args.model)
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
save_figure_dir = os.path.join(root_dir,'figures')
if not os.path.exists(save_figure_dir):
    os.makedirs(save_figure_dir)

for start_point in args.start_point_list:
    df_train,df_test,df_all = MultiVariateBatteryDataProcess(BatteryData,args.test_name,start_point,args)
    mask_len =len(df_train)     # 训练集按照80%，20%划分训练集和验证集
    # tf_test =len(df_test)
    time_varying_known_reals=['voltage mean','voltage std','voltage kurtosis','voltage skewness','CC Q','CC charge time','voltage slope','voltage entropy',
                                      'current mean','current std','current kurtosis','current skewness','CV Q','CV charge time','current slope','current entropy','Capacity']
    time_varying_unknown_reals = ['target']
    # --------------------- 历史长度和预测长度 -------------------------
    max_prediction_length = args.pred_len
    max_encoder_length = args.seq_len #24
    # ---(5)构建数据集---
    training = TimeSeriesDataSet(
        df_train[0:int(0.8*mask_len)],
        time_idx="time_idx",
        target="target",
        group_ids=['group_id'],  
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length, 
        max_prediction_length=max_prediction_length,  
        # 特征（除了target外的）
        time_varying_known_reals=time_varying_known_reals,  # 【随时间变化且在未来已知的连续变量】
        # 未来未知
        time_varying_unknown_reals=time_varying_unknown_reals,  # 【随时间变化且在未来未知的连续变量】
        #  # use softplus and normalize by group  【如何计算】   Softplus函数可以看作是ReLU函数的平滑。Softplus(x)=log(1+e^x)
        target_normalizer=EncoderNormalizer(),#MultiNormalizer([TorchNormalizer(),TorchNormalizer()]),   # EncoderNormalizer(), TorchNormalizer()
        add_encoder_length=False
    )
    # 从数据集中获取数据加载器    # train ( bool , optional ) -- 如果数据加载器用于训练或预测，如果为真，将打乱并丢弃最后一批
    train_dataloader = training.to_dataloader(train=True, batch_size=args.batch_size,shuffle=True,num_workers=0,drop_last=True)
    # ---  保存文件名，设置特征数量【必须】 -------
    for x, (y, weight) in iter(train_dataloader):
        print("['encoder_cont']:", x['encoder_cont'].shape)  # [batch_size, max_encoder_length, 5]
        print("['decoder_cont']:", x['decoder_cont'].shape)  # [batch_size, max_prediction_length, 15]
        en_feats_num = x['encoder_cat'].shape[-1] + x['encoder_cont'].shape[-1]
        de_feats_num = x['decoder_cat'].shape[-1] + x['decoder_cont'].shape[-1]
        print('y:', y[0].shape)     #
        # print('weight:',weight.shape)
        break
    validing = TimeSeriesDataSet(
        df_train[int(0.8 * mask_len):],
        time_idx="time_idx",  # 【时间索引字段，确定样本的顺序】
        target='target', 
        group_ids=['group_id'],  # 【无】
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length, 
        max_prediction_length=max_prediction_length,  
        # 特征（除了target外的）
        time_varying_known_reals=time_varying_known_reals,  # 【随时间变化且在未来已知的连续变量】
        # 未来未知
        time_varying_unknown_reals=time_varying_unknown_reals,  # 【随时间变化且在未来未知的连续变量】
        target_normalizer=EncoderNormalizer(),
        add_encoder_length=False
    )
    val_dataloader = validing.to_dataloader(train=False,batch_size=args.batch_size,shuffle=False,num_workers=0,drop_last=False)
    testing= TimeSeriesDataSet(
        df_test,
        time_idx="time_idx",  # 【时间索引字段，确定样本的顺序】
        target='target', 
        group_ids=['group_id'],  # 【无】
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,  
        max_prediction_length=max_prediction_length, 
        # 特征（除了target外的）
        time_varying_known_reals=time_varying_known_reals,  # 【随时间变化且在未来已知的连续变量】
        # 未来未知
        time_varying_unknown_reals=time_varying_unknown_reals,  # 【随时间变化且在未来未知的连续变量】
        #  # use softplus and normalize by group  【如何计算】   Softplus函数可以看作是ReLU函数的平滑。Softplus(x)=log(1+e^x)
        target_normalizer=EncoderNormalizer(),
        add_encoder_length=False
    )
    test_dataloader = testing.to_dataloader(train=False,batch_size=args.batch_size,shuffle=False,num_workers=0,drop_last=False)


    #mask_len==
    sp_root_dir = os.path.join(root_dir,'SP{}/'.format(start_point))
    if not os.path.exists(sp_root_dir):
        os.makedirs(sp_root_dir)

    # 记录统计日志
    after_name = 'in_l_{}_out_l_{}_Pcap'.format(max_encoder_length, max_prediction_length)
    stat_log_path = os.path.join(sp_root_dir,
                            'log_stat_Feas_{}_{}_{}.txt'.format(len(training.reals), en_feats_num, after_name))
    stat_log = open(stat_log_path, 'w', encoding='UTF-8')
    print_log('model name:{}\n'.format(args.model),stat_log)
    print_log('selected battery name:{}, start point:{}\n'.format(args.test_name,start_point),stat_log)
    # 10次随机实验的指标平均值:MAE,RMSE,R^2,RE,RUL_real,RUL_pred,AE==|RUL_real-RUL_pred|
    MAE_avg = 0.
    RMSE_avg = 0.
    r2_avg = 0.
    RE_avg= 0.
    AE_avg = 0.
    RUL_real_avg = 0.
    RUL_pred_avg = 0.
    epoch_avg=0.
    train_time_avg = 0.
    infer_time_avg = 0.
    count=0
    stat_pred_data_list = []
    from assistant import set_seed
    while count<args.count:
        count+=1
        args.seed = set_seed(count)
        # --------------------------------------step 2: 网络构建--------------------------------------------------------
        # ----------------------------------------- 创建基线模型 ----------------------------------------------
        from ModelsModify.TimeMixer import TimeMixerNetModel
        model = TimeMixerNetModel.from_dataset(
            training,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=len(time_varying_known_reals),
            learning_rate=0.001,
            loss=SMAPE(),  
        )
        # -----------------------------------------  保存文件名，设置特征数量【必须】 -----------------------------------
        after_name = 'in_l_{}_out_l_{}_Pcap'.format(max_encoder_length, max_prediction_length)
        import time, datetime

        save_dir = sp_root_dir+'Exp{}/'.format(count)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        curr_time = time.strftime('%Y-%m-%d %X', time.localtime()).replace(':', '-')

        log_path = os.path.join(save_dir,
                                'log_Feas_{}_{}_{}.txt'.format(len(training.reals), en_feats_num, after_name))
        log = open(log_path, 'w', encoding='UTF-8')


        # 保存训练结果
        study_model_path = os.path.join(save_dir, str(model).split('\n')[0][:-1])
        if not os.path.exists(study_model_path):
            os.makedirs(study_model_path)

        print_log(('train dataset:{},val:{},test:{}'.format(int(0.8 * mask_len),int(0.1 * mask_len), len(df_test))), log)
        print_log(('Input Feature num:{} ,name:{}'.format(len(training.reals), training.reals)), log)

        print_log(('time_varying_unknown_reals num:{},decoder num:{} ,name:{}'.format(len(training.time_varying_unknown_reals),
                                                                                    de_feats_num - len(training.time_varying_unknown_reals),
                                                                                    training.time_varying_unknown_reals)),log)

        print_log('model name:{}\n'.format(args.model),log)
        print_log((f"Number of parameters in network(参数数量(Params)): {model.size()/1e3:.1f}k"),log)


        # print_log(('model structure:\n{}'.format(model)),log)
        # --------------------------------------step 3: 训练--------------------------------------------------------
        # 提前停止的回call函数
        early_stop_callback = EarlyStopping(monitor='val_loss',min_delta=1e-5,patience=10,verbose=False,mode='min')
        # EarlyStopping 被配置为监控验证集上的损失（val_loss），并且在连续10个epoch（patience）中没有改进（即损失没有下降超过最小变化阈值 min_delta）时
        # 停止训练。verbose=False 表示在停止训练时不会打印任何信息，mode='min' 表示我们关注的性能指标是最小化的，即损失。

        # 训练器设置
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            gpus=1,
            # weights_summary="top",
            gradient_clip_val=0.2,
            # limit_train_batches=30,  # coment in for training, running valiation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            # callbacks=[lr_logger, early_stop_callback],
            callbacks=[early_stop_callback],
            logger=False,   # 禁用内置的日志记录器
            default_root_dir=study_model_path,
        )
        #'''
        # 训练网络        ckpt_path: Path/URL of the checkpoint from which training is resumed.
        start_time = time.time()
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        end_time = time.time()
        train_time = end_time - start_time


        # *** 加载最优模型(本循环内) ***
        best_model_path = trainer.checkpoint_callback.best_model_path
        print('best_model_path:',best_model_path)
        print(('best_model_path:{}'.format(best_model_path)),log)
        # 加载最佳模型参数
        device=torch.device('cuda')
        best_model = TimeMixerNetModel.load_from_checkpoint(best_model_path).to(device=device)

        # 预测方式,不实时更新
        start_time = time.time()
        predictions = best_model.predict(test_dataloader,batch_size=256)
        end_time = time.time()
        infer_time = end_time - start_time
        predictions = predictions.detach().cpu().numpy().reshape(-1)
        actuals_df = df_all.loc[df_all['Cycle']>=start_point,['Cycle','target']]
        actuals = actuals_df['target'].values
        y_true = actuals*args.Rated_Capacity
        y_pred = predictions*args.Rated_Capacity

        stat_pred_data_list.append(y_pred)
        mask = y_true >= 0.
        # ----------------------------  统计常用指标 ------------------
        from sklearn.metrics import r2_score
        # 指标:MAE,RMSE,R^2,RE,RUL_real,RUL_pred,AE==RUL_err,
        # 统计指标：AMAE,ARMSE,AR^2,ARE,AAE
        # NMAE
        NMAE = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        # NRMSE
        NRMSE = np.sqrt(np.mean(np.square(y_true[mask] - y_pred[mask])))
        # R^2
        r2 = r2_score(y_true[mask], y_pred[mask])
        # RUL_real,RUL_pred,AE==RUL_err,RE
        RUL_real,RUL_pred,AE,RE = rul_value_error(y_true[mask],y_pred[mask], threshold=args.Rated_Capacity*0.7)
        epoch_avg += trainer.current_epoch
        print_log("第{}次实验运行的Epoch数量: {}  \n".format(count,trainer.current_epoch),log)
        print_log("第{}次实验运行的训练时间: {}  \n".format(count,train_time),log)
        print_log("第{}次实验运行的推理时间: {}  \n".format(count,infer_time),log)
        print_log(('第{}次实验的结果: \nMAE:{:.4f}, RMSE:{:.4f}, r2:{:.4f}, RUL_real:{}, RUL_pred:{}, AE:{}, RE:{:.4f}  \n'.format(
                count,NMAE,NRMSE,r2,RUL_real,RUL_pred,AE,RE)), log)
        print_log("第{}次实验运行的Epoch数量: {}  \n".format(count,trainer.current_epoch),stat_log)
        print_log("第{}次实验运行的训练时间: {}  \n".format(count,train_time),stat_log)
        print_log("第{}次实验运行的推理时间: {}  \n".format(count,infer_time),stat_log)
        print_log(('第{}次实验的结果: \nMAE:{:.4f}, RMSE:{:.4f}, r2:{:.4f}, RUL_real:{}, RUL_pred:{}, AE:{}, RE:{:.4f}  \n'.format(
                count,NMAE,NRMSE,r2,RUL_real,RUL_pred,AE,RE)), stat_log)
        print_log(('第{}次实验模型的保存路径:{}'.format(count,best_model_path)),stat_log)
        #'''
        # 记录统计结果之和
        MAE_avg += NMAE
        RMSE_avg += NRMSE
        r2_avg += r2
        RE_avg += RE
        AE_avg += AE
        RUL_real_avg += RUL_real
        RUL_pred_avg += RUL_pred
        train_time_avg += train_time
        infer_time_avg += infer_time
       
        del model
        log.close()

    all_pred_data_list.append(stat_pred_data_list)
    # 记录结果之和
    MAE_avg /= args.count
    RMSE_avg /= args.count
    r2_avg /= args.count
    RE_avg /= args.count
    AE_avg /= args.count
    RUL_real_avg /= args.count
    RUL_pred_avg /= args.count
    epoch_avg /= args.count
    train_time_avg /= args.count
    infer_time_avg /= args.count
    print_log("{}次实验实际运行的Epoch数量的平均值: {}  \n".format(args.count,epoch_avg),stat_log)
    print_log("{}次实验实际运行的训练时间的平均值: {} 秒  \n".format(args.count,train_time_avg),stat_log)
    print_log("{}次实验实际运行的推理时间的平均值: {} 秒  \n".format(args.count,infer_time_avg),stat_log)
    print_log(('{}次实验各项指标的统计平均值: \nMAE:{:.4f}, RMSE:{:.4f}, r2:{:.4f}, RUL_real:{}, RUL_pred:{}, AE:{}, RE:{:.4f}  \n'.format(
            args.count,MAE_avg,RMSE_avg,r2_avg,RUL_real_avg,RUL_pred_avg,AE_avg,RE_avg)),stat_log)
    setting = '{}_sl{}_ll{}_pl{}_bs{}_ct{}_tn{}_me{}'.format(
            args.model,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.batch_size,
            args.count,
            int(args.test_name[-1:]),
            args.max_epochs)
    print_log('args setting : {}\n'.format(setting),stat_log)
    stat_log.close()          

results = dict()
results['SP200'] = all_pred_data_list[0]
results['SP300'] = all_pred_data_list[1]
results['SP400'] = all_pred_data_list[2]
if not os.path.exists('results'):
    os.makedirs('results')
torch.save(results, 'results/RUL_{}_{}.pth'.format(args.test_name,args.model))


from Helper_Plot import *
if args.test_name == 'CY25_1':
    for i in range(args.count):
        single_model_draw_test_CY25_1_plt(real_data,all_pred_data_list[0][i],all_pred_data_list[1][i],all_pred_data_list[2][i],
            save_filename='best_model_{}'.format(i)+'_RUL_Prediction',save_figure_dir=save_figure_dir,Rated_Capacity=args.Rated_Capacity,model=args.model)