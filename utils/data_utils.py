import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


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
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)

def load_forecast_csv(root, name, target):
    data = pd.read_csv(f'{root + name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
        data = data[['OT']]
    elif name == 'KPVPG':
        data = data[['device_1']]
    elif name == 'ECL':
        data = data[['MT_320']]
    elif name == 'electricity':
        data = data[['MT_001']]
    else:
        data = data.iloc[[target]]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2' or name == 'device1':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.7 * len(data)))
        valid_slice = slice(int(0.7 * len(data)), int(0.9 * len(data)))
        test_slice = slice(int(0.9 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)

    data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
        
    return data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols


class forecast_Dataset(Dataset):
    def __init__(self, data, name, flag='train', size=None):
        # size [LT_len, ST_len, Trg_len]
        self.LT_len = size[0]
        self.ST_len = size[1]
        self.Trg_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.data = data.reshape(data.shape[1],data.shape[2]) 
        self.name = name
        self.__read_data__()

    def __read_data__(self):
        if self.name == 'ETTh1' or self.name == 'ETTh2' or self.name == 'KPVPG':
            slice_arr = [slice(None, 12*30*24), 
                        slice(12*30*24 - self.LT_len, 12*30*24+4*30*24), 
                        slice(12*30*24+4*30*24 - self.LT_len, 12*30*24+8*30*24)]
            sel_slice = slice_arr[self.set_type]
            
        elif self.name == 'ETTm1' or self.name == 'ETTm2':
            slice_arr = [slice(None, 12*30*24*4), 
                        slice(12*30*24*4 - self.LT_len, 12*30*24*4+4*30*24*4), 
                        slice(12*30*24*4+4*30*24*4 - self.LT_len, 12*30*24*4+8*30*24*4)]
            sel_slice = slice_arr[self.set_type]

        else:
            num_train = int(len(self.data)*0.7)
            num_test = int(len(self.data)*0.2)
            num_vali = len(self.data) - num_train - num_test

            slice_arr = [slice(None, num_train), 
                        slice(num_train-self.LT_len, num_train+num_vali), 
                        slice(len(self.data)-num_test-self.LT_len, len(self.data))]
            sel_slice = slice_arr[self.set_type]

        self.data_x = self.data[sel_slice]
        self.data_y = self.data[sel_slice, 7:]
    
    def __getitem__(self, index):
        LT_begin = index
        LT_end = LT_begin + self.LT_len
        Trg_begin = LT_end 
        Trg_end = Trg_begin + self.Trg_len

        seq_x = self.data_x[LT_begin:LT_end]
        seq_y = self.data_y[Trg_begin:Trg_end]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.LT_len- self.Trg_len