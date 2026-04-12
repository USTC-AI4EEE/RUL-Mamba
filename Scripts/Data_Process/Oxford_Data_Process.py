import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.io
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OXFORD_DATA_DIR = PROJECT_ROOT / 'Data' / 'Oxford_Data'
OXFORD_CACHE_PATH = OXFORD_DATA_DIR / 'Oxford.npy'
OXFORD_REAL_DATA_TEMPLATE = 'Results/Capacity_{test_name}_Real_Data.pth'
CELL_SIZE = [83, 78, 82, 52, 51, 51, 82, 82]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data/Oxford_Data/', help='path of the data file')
    parser.add_argument('--battery_cache_path', type=str, default='Data/Oxford_Data/Oxford.npy', help='path of the cached Oxford data file')
    parser.add_argument('--battery_list', nargs='+', default=['Cell1', 'Cell3', 'Cell8'], help='Battery data used in Oxford training')
    parser.add_argument('--train_cells', nargs='+', default=['Cell1', 'Cell3'], help='Battery data used for train')
    parser.add_argument('--Rated_Capacity', type=float, default=0.74, help='Rate Capacity')
    parser.add_argument('--test_name', type=str, default='Cell8', help='Battery data used for test')
    parser.add_argument('--start_point_list', nargs='+', type=int, default=[20, 30, 40], help='The time_idx/cycle windows when prediction gets started.')
    parser.add_argument('--seq_len', type=int, default=10, help='input sequence length in time_idx space')
    return parser


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_runtime_args(config=None):
    if config is None:
        return SimpleNamespace(
            data_dir='Data/Oxford_Data/',
            battery_cache_path='Data/Oxford_Data/Oxford.npy',
            battery_list=['Cell1', 'Cell3', 'Cell8'],
            train_cells=['Cell1', 'Cell3'],
            Rated_Capacity=0.74,
            test_name='Cell8',
            start_point_list=[20, 30, 40],
            seq_len=10,
        )
    dataset = config['dataset']
    window = config.get('window', {})
    return SimpleNamespace(
        data_dir=dataset.get('data_dir', 'Data/Oxford_Data/'),
        battery_cache_path=dataset.get('battery_cache_path', 'Data/Oxford_Data/Oxford.npy'),
        battery_list=dataset.get('battery_list', ['Cell1', 'Cell3', 'Cell8']),
        train_cells=dataset.get('train_cells', ['Cell1', 'Cell3']),
        Rated_Capacity=dataset['rated_capacity'],
        test_name=dataset['test_name'],
        start_point_list=dataset['start_points'],
        seq_len=window.get('seq_len', 10),
    )


def preprocess_oxford_data(data_dir='Data/Oxford_Data/'):
    resolved_data_dir = _resolve_project_path(data_dir)
    mat_path = resolved_data_dir / 'Oxford_Battery_Degradation_Dataset_1.mat'
    csv_path = resolved_data_dir / 'Oxford.csv'
    data = scipy.io.loadmat(str(mat_path))
    full_data_list = []
    for i in range(8):
        cell_num = f'Cell{i + 1}'
        cell_data = data[cell_num]
        max_cycle_number = (CELL_SIZE[i] - 1) * 100
        expected_cycles = range(0, max_cycle_number + 100, 100)
        cell_df = pd.DataFrame(index=expected_cycles, columns=['cycle_number', 'cell_number', 'capacity'])
        cell_df['cycle_number'] = cell_df.index
        cell_df['cell_number'] = cell_num
        cell_df = cell_df.astype({
            'cycle_number': 'int',
            'cell_number': 'str',
            'capacity': 'float',
        })
        for cycle in cell_data.dtype.names:
            cycle_number = int(cycle[3:])
            if 'C1dc' in cell_data[cycle][0, 0].dtype.names:
                charge_data = cell_data[cycle][0, 0]['C1dc'][0, 0]['q'][0, 0].flatten()
                cell_df.loc[cycle_number, 'capacity'] = abs(charge_data[-1]) / 1000
        full_data_list.append(cell_df)

    full_df = pd.concat(full_data_list).reset_index(drop=True)
    full_df['capacity'] = full_df['capacity'].interpolate(method='linear')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(csv_path, index=False)
    return full_df


def save_real_data(df_all=None, test_name='Cell8'):
    results_dir = PROJECT_ROOT / 'Results'
    results_dir.mkdir(parents=True, exist_ok=True)
    if df_all is None:
        df_all = pd.read_csv(PROJECT_ROOT / 'Data' / 'Oxford_Data' / 'Oxford.csv')
    df_i = df_all.loc[df_all['cell_number'] == test_name, ['cycle_number', 'cell_number', 'capacity']]
    real_data = df_i['capacity'].values
    torch.save(real_data, _resolve_project_path(OXFORD_REAL_DATA_TEMPLATE.format(test_name=test_name)))


def BatteryDataRead(args):
    full_df = preprocess_oxford_data(args.data_dir)
    cache_path = _resolve_project_path(args.battery_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, full_df.to_records(index=False), allow_pickle=True)
    return full_df


def LoadBatteryCache(cache_path):
    resolved_cache_path = _resolve_project_path(cache_path)
    battery_data = np.load(resolved_cache_path, allow_pickle=True)
    if isinstance(battery_data, np.ndarray) and battery_data.dtype.names:
        return pd.DataFrame.from_records(battery_data)
    return pd.DataFrame(battery_data)


def DataProcess(BatteryData, test_name, start_point, rated_capacity, seq_len=10, train_cells=None):
    args = SimpleNamespace(
        Rated_Capacity=rated_capacity,
        seq_len=seq_len,
        test_name=test_name,
        train_cells=train_cells or ['Cell1', 'Cell3'],
    )
    return BatteryDataProcess(BatteryData, test_name, start_point, args)


def BatteryDataProcess(BatteryData, test_name, start_point, args):
    df = BatteryData.copy()
    df_train = df.loc[df['cell_number'].isin(args.train_cells), ['cycle_number', 'cell_number', 'capacity']].copy()
    df_train['capacity'] /= args.Rated_Capacity
    df_train['target'] = df_train['capacity']
    df_train['time_idx'] = df_train['cycle_number'].map(lambda x: int(x / 100))
    df_train['group_id'] = df_train['cell_number'].map({'Cell1': 0, 'Cell3': 1, 'Cell8': 2})
    df_train = df_train.drop(['cell_number'], axis=1)
    df_train['idx'] = [x for x in range(len(df_train))]
    df_train.set_index('idx', inplace=True)

    df_all = df.loc[df['cell_number'] == test_name, ['cycle_number', 'cell_number', 'capacity']].copy()
    df_all['capacity'] /= args.Rated_Capacity
    df_all['target'] = df_all['capacity']
    df_all['time_idx'] = df_all['cycle_number'].map(lambda x: int(x / 100))
    df_all['group_id'] = df_all['cell_number'].map({'Cell1': 0, 'Cell3': 1, 'Cell8': 2})
    df_all = df_all.drop(['cell_number'], axis=1)
    df_all['idx'] = [x for x in range(len(df_all))]
    df_all.set_index('idx', inplace=True)

    min_val = df_train['capacity'].min()
    max_val = df_train['capacity'].max()
    df_train['capacity'] = (df_train['capacity'] - min_val) / (max_val - min_val)
    df_all['capacity'] = (df_all['capacity'] - min_val) / (max_val - min_val)

    df_test = df_all.loc[
        df_all['cycle_number'] >= start_point * 100 - args.seq_len * 100,
        ['time_idx', 'group_id', 'cycle_number', 'capacity', 'target'],
    ].copy()
    df_test['idx'] = [x for x in range(len(df_test))]
    df_test.set_index('idx', inplace=True)
    return df_train, df_test, df_all


def main():
    args = build_parser().parse_args()
    battery_data = BatteryDataRead(args)
    save_real_data(battery_data, args.test_name)


if __name__ == '__main__':
    main()
