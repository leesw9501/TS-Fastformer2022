import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

import warnings
warnings.filterwarnings('ignore')


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.isocalendar().week.to_numpy()#dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)

def load_forecast_csv(root, name, target, seq_len, univar=False):
    data = pd.read_csv(f'{root + name}', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
                        
    if univar:
        print('target :', target)
        data = data[[target]]
    name = name[:-4]
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2' or name == 'KPVPG':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24 - seq_len, 16*30*24)
        test_slice = slice(16*30*24 - seq_len, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4 - seq_len, 16*30*24*4)
        test_slice = slice(16*30*24*4 - seq_len, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.7 * len(data)))
        valid_slice = slice(int(0.7 * len(data)) - seq_len, int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)) - seq_len, None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)

    data = np.expand_dims(data, 0)
    print(data.shape)
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
        
    return data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols



class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, repr, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True):

        # info
        if size == None:
            self.LT_len = 24 * 14
            self.ST_len = 24 * 4
            self.Trg_len = 24 * 7
        else:
            self.LT_len = size[0]
            self.ST_len = size[1]
            self.Trg_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.repr = repr.reshape(repr.shape[1], repr.shape[2])

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.LT_len, 12 * 30 * 24 + 4 * 30 * 24 - self.LT_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.LT_len

        r_begin = x_end - self.ST_len
        r_end = r_begin + self.ST_len

        y_begin = x_end
        y_end = x_end + self.Trg_len


        seq_x = self.data_x[x_begin:x_end]
        seq_r = self.repr[r_begin:r_end]
        seq_y = self.data_y[y_begin:y_end]
        

        return seq_x, seq_r, seq_y

    def __len__(self):
        return len(self.data_x) - self.LT_len - self.Trg_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, repr, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True):

        if size == None:
            self.LT_len = 24 * 14 * 4
            self.ST_len = 24 * 4 * 4
            self.Trg_len = 24 * 7 * 4
        else:
            self.LT_len = size[0]
            self.ST_len = size[1]
            self.Trg_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.repr = repr.reshape(repr.shape[1], repr.shape[2])
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.LT_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.LT_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.LT_len

        r_begin = x_end - self.ST_len
        r_end = r_begin + self.ST_len

        y_begin = x_end
        y_end = x_end + self.Trg_len


        seq_x = self.data_x[x_begin:x_end]
        seq_r = self.repr[r_begin:r_end]
        seq_y = self.data_y[y_begin:y_end]
        

        return seq_x, seq_r, seq_y

    def __len__(self):
        return len(self.data_x) - self.LT_len - self.Trg_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, repr, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True):
        
        # info
        if size == None:
            self.LT_len = 24 * 14
            self.ST_len = 24 * 4
            self.Trg_len = 24 * 7
        else:
            self.LT_len = size[0]
            self.ST_len = size[1]
            self.Trg_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.repr = repr.reshape(repr.shape[1], repr.shape[2])
        self.target = target
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))


        num_train = int(len(df_raw) * 0.7)
        num_vali = int(len(df_raw) * 0.8)
        num_test = len(df_raw)
        border1s = [0, num_train - self.LT_len, num_vali - self.LT_len]
        border2s = [num_train, num_vali, num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.LT_len

        r_begin = x_end - self.ST_len
        r_end = r_begin + self.ST_len

        y_begin = x_end
        y_end = x_end + self.Trg_len


        seq_x = self.data_x[x_begin:x_end]
        seq_r = self.repr[r_begin:r_end]
        seq_y = self.data_y[y_begin:y_end]
        

        return seq_x, seq_r, seq_y

    def __len__(self):
        return len(self.data_x) - self.LT_len - self.Trg_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'KPVPG': Dataset_ETT_hour,
    'WTH': Dataset_Custom,
    'ECL': Dataset_Custom,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.dataset]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        data_path=args.dataset + '.csv',
        repr=args.repr[flag],
        flag=flag,
        size=[args.LT_len, args.ST_len, args.Trg_len],
        features='S',
        target=args.target
    )
    print(flag, len(data_set), data_set.data_x.shape)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader