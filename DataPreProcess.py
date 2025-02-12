import numpy as np
import pandas as pd
import argparse
import glob
import os
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from assistant import get_gpus_memory_info
from sklearn.model_selection import train_test_split

id,_ = get_gpus_memory_info()
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
def write_to_txt(txt_name,txt):
    with open(txt_name,'a') as f:
        f.write(txt)
        f.write('\n')

class DF():
    def __init__(self,args):
        self.args = args

    def _3_sigma(self, Ser1):
        '''
        :param Ser1:
        :return: index
        '''
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index

    def delete_3_sigma(self,df):
        '''
        :param df: DataFrame
        :return: DataFrame
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        out_index = []
        for col in df.columns:
            index = self._3_sigma(df[col])
            out_index.extend(index)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0)
        df = df.reset_index(drop=True)
        df['Cycle'] =  df.index + 1
        return df

    def read_one_csv(self,file_name):
        '''
        read a csv file and return a DataFrame
        :param file_name: str
        :return: DataFrame
        '''
        df = pd.read_csv(file_name)
        df.insert(df.shape[1]-1,'cycle index',np.arange(df.shape[0]))
        df = self.delete_3_sigma(df)

        return df


def BatteryDataRead(args):
    root = 'data/TJU data/Dataset_3_NCM_NCA_battery/'
    files = os.listdir(root)
    Battery_list = {}
    for i in range(1,4):
        for f in files:
            if 'CY25-05_1' in f and '#'+str(i) in f:
                path = os.path.join(root,f)
                df_i = DF(args)
                df = df_i.read_one_csv(path)
                df = df.rename(columns={'capacity': 'Capacity'})
                df['BatteryName'] = f'CY25_{i}'
                Battery_list[f'CY25_{i}'] = df
    return Battery_list


def MultiVariateBatteryDataProcess(BatteryData,test_name,start_point,args):
    dict_without_test = {key: value for key, value in BatteryData.items() if key != test_name}
    df_train = pd.concat(dict_without_test.values())
    df_train = df_train.filter(items=['BatteryName', 'Cycle','voltage mean','voltage std','voltage kurtosis','voltage skewness','CC Q','CC charge time','voltage slope','voltage entropy',
                                      'current mean','current std','current kurtosis','current skewness','CV Q','CV charge time','current slope','current entropy','Capacity'])
    df_train['Capacity'] /= args.Rated_Capacity
    df_train['target'] = df_train['Capacity']
    df_train['time_idx'] = df_train['Cycle'].map(lambda x: int(x-1))
    df_train['group_id'] = df_train['BatteryName'].map({'CY25_1':0, 'CY25_2':1, 'CY25_3':2})
    df_train = df_train.drop(['BatteryName'],axis=1)
    df_train['idx'] = [x for x in range(len(df_train))]
    df_train.set_index('idx',inplace=True)

    df_test = BatteryData[test_name]
    df_test = df_test.filter(items=['BatteryName', 'Cycle','voltage mean','voltage std','voltage kurtosis','voltage skewness','CC Q','CC charge time','voltage slope','voltage entropy',
                                      'current mean','current std','current kurtosis','current skewness','CV Q','CV charge time','current slope','current entropy','Capacity'])
    df_test['Capacity'] /= args.Rated_Capacity
    df_test['target'] = df_test['Capacity']
    df_test['time_idx'] = df_test['Cycle'].map(lambda x: int(x-1))
    df_test['group_id'] = df_test['BatteryName'].map({'CY25_1':0, 'CY25_2':1, 'CY25_3':2})
    df_test = df_test.drop(['BatteryName'],axis=1)
    df_test['idx'] = [x for x in range(len(df_test))]
    df_test.set_index('idx',inplace=True)

    min_val = df_train['Capacity'].min()
    max_val = df_train['Capacity'].max()
    df_train['Capacity'] = (df_train['Capacity']-min_val) / (max_val - min_val)
    df_test['Capacity'] = (df_test['Capacity']-min_val) / (max_val - min_val)
    #测试集全部数据
    df_all = df_test 
    #测试集参与测试的数据
    df_test = df_all.loc[df_all['Cycle']>=start_point-args.seq_len,['time_idx','group_id','Cycle','voltage mean','voltage std','voltage kurtosis','voltage skewness','CC Q','CC charge time','voltage slope','voltage entropy',
                                      'current mean','current std','current kurtosis','current skewness','CV Q','CV charge time','current slope','current entropy','Capacity','target']] 
    df_test['idx'] = [x for x in range(len(df_test))]
    df_test.set_index('idx',inplace=True)
    return df_train,df_test,df_all

def BatteryDataProcess(BatteryData,test_name,start_point):
    dict_without_test = {key: value for key, value in BatteryData.items() if key != test_name}
    df_train = pd.concat(dict_without_test.values())
    df_train = df_train.filter(items=['BatteryName', 'Cycle','Capacity'])
    df_train['Capacity'] /= args.Rated_Capacity
    df_train['target'] = df_train['Capacity']
    df_train['time_idx'] = df_train['Cycle'].map(lambda x: int(x-1))
    df_train['group_id'] = df_train['BatteryName'].map({'CY25_1':0, 'CY25_2':1, 'CY25_3':2})
    df_train = df_train.drop(['BatteryName'],axis=1)
    df_train['idx'] = [x for x in range(len(df_train))]
    df_train.set_index('idx',inplace=True)

    df_test = BatteryData[test_name]
    df_test = df_test.filter(items=['BatteryName', 'Cycle','Capacity'])
    df_test['Capacity'] /= args.Rated_Capacity
    df_test['target'] = df_test['Capacity']
    df_test['time_idx'] = df_test['Cycle'].map(lambda x: int(x-1))
    df_test['group_id'] = df_test['BatteryName'].map({'CY25_1':0, 'CY25_2':1, 'CY25_3':2})
    df_test = df_test.drop(['BatteryName'],axis=1)
    df_test['idx'] = [x for x in range(len(df_test))]
    df_test.set_index('idx',inplace=True)

    min_val = df_train['Capacity'].min()
    max_val = df_train['Capacity'].max()
    df_train['Capacity'] = (df_train['Capacity']-min_val) / (max_val - min_val)
    df_test['Capacity'] = (df_test['Capacity']-min_val) / (max_val - min_val)
    #测试集全部数据
    df_all = df_test 
    #测试集参与测试的数据
    df_test = df_all.loc[df_all['Cycle']>=start_point-args.seq_len,['time_idx','group_id','Cycle','Capacity','target']] 
    df_test['idx'] = [x for x in range(len(df_test))]
    df_test.set_index('idx',inplace=True)
    return df_train,df_test,df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length') 
    parser.add_argument('--Battery_list', type=list, default=['CY25_1', 'CY25_2', 'CY25_3'], help='Battery data')
    parser.add_argument('--data_dir', type=str, default='data/TJU data/Dataset_3_NCM_NCA_battery/', help='path of the data file')
    parser.add_argument('--Rated_Capacity', type=float, default=2.5, help='Rate Capacity')
    parser.add_argument('--test_name', type=str, default='CY25_1', help='Battery data used for test')
    parser.add_argument('--start_point_list', type=int, default=[200,300,400], help='The cycle when prediction gets started.')
    args = parser.parse_args()

    BatteryData = BatteryDataRead(args)
    BatteryData_array = np.array([BatteryData], dtype=object)
    np.save('data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy', BatteryData_array, allow_pickle=True)
    
    BatteryData = np.load('data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy', allow_pickle=True)
    BatteryData = BatteryData.item()
    _,_,df_all = MultiVariateBatteryDataProcess(BatteryData,args.test_name,args.start_point_list[0],args)
    real_data = df_all['target'].values*args.Rated_Capacity
    if not os.path.exists('results'):
        os.makedirs('results')
    torch.save(real_data, 'results/Capacity_{}_real_data.pth'.format(args.test_name))