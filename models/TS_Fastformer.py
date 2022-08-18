import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import time
import warnings

from utils.utils import EarlyStopping, adjust_learning_rate, metric
from utils.data_utils import forecast_Dataset 
from models.models import PastAttentionDecoder, DataRepresentation

warnings.filterwarnings('ignore')

    
class Run_TS_Fastformer():
    def __init__(self, args):
        super(Run_TS_Fastformer, self).__init__()
        self.repr_model = args.repr_model
        self.device = args.device
        self.scaler = args.scaler
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.data = args.data_
        self.data_name = args.dataset
        self.patience = args.patience
        self.epochs = args.epochs
        self.LT_len = args.LT_len
        self.ST_len = args.ST_len
        self.Trg_len = args.Trg_len

        self.model = TS_Fastformer(
        LT_in = args.LT_in,
        ST_in = args.ST_in, 
        Trg_out = args.Trg_out,
        Trg_len = args.Trg_len,
        model_dim = args.model_dim, 
        n_heads = args.n_heads,
        dec_layers = args.dec_layers, 
        fcn_dim = args.fcn_dim,
        dropout = args.dropout,
        device = args.device
        ).to(self.device)


    def _get_data(self, flag):

        Data = forecast_Dataset

        if flag == 'test':
            shuffle_flag = False
        else:
            shuffle_flag = True
        drop_last = True
        batch_size = self.batch_size
        data_set = Data(
            data = self.data,
            name = self.data_name,
            flag=flag,
            size=[self.LT_len, self.ST_len, self.Trg_len]
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last)

        return data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            pred, true = self._process_one_batch(batch_x, batch_y)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, path, num):
        train_loader = self._get_data(flag = 'train')
        vali_loader = self._get_data(flag = 'val')
        test_loader = self._get_data(flag = 'test')

        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(batch_x, batch_y)
                
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
                
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_time = time.time()
            vali_loss = self.vali(vali_loader, criterion)
            print('valitime:', time.time()-vali_time)
            test_time = time.time()
            test_loss = self.vali(test_loader, criterion)
            print('testtime:', time.time()-test_time)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path, num)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.learning_rate)
            
        best_model_path = path+'/'+'checkpoint_' + str(num) + '.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        return self.model

    def test(self, path, num, inverse):
        test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x, batch_y) in enumerate(test_loader):
            pred, true = self._process_one_batch(batch_x, batch_y)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if inverse:
            pred_inv = self.scaler.inverse_transform(preds.reshape(-1,preds.shape[-1])).reshape(preds.shape)
            true_inv = self.scaler.inverse_transform(trues.reshape(-1,preds.shape[-1])).reshape(trues.shape)
            mae, mse, rmse, mape, mspe = metric(pred_inv, true_inv)
            print('inverse __ mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}'.format(mae, mse, rmse, mape, mspe))
            np.save(path+'/metrics_inv_' + str(num) + '.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(path+'/pred_inv_' + str(num) + '.npy', pred_inv)
            np.save(path+'/true_inv_' + str(num) + '.npy', true_inv)
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}'.format(mae, mse, rmse, mape, mspe))
            np.save(path+'/metrics_' + str(num) + '.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(path+'/pred_' + str(num) + '.npy', preds)
            np.save(path+'/true_' + str(num) + '.npy', trues)

        return

    def _process_one_batch(self, batch_x, batch_y):
        # Long-term input
        batch_x = batch_x.float().to(self.device)
        batch_repr_x = self.repr_model.encode(batch_x).float().to(self.device)

        # Short-term input
        ST_inp = batch_x[:,-self.ST_len:, :]
        ST_repr_inp = batch_repr_x[:,-self.ST_len:, :]
        
        outputs = self.model(batch_x, batch_repr_x, ST_inp, ST_repr_inp)

        f_dim = -1
        batch_y = batch_y[:,:,f_dim:].float().to(self.device)

        return outputs, batch_y

class TS_Fastformer(nn.Module):
    def __init__(self, LT_in, ST_in, Trg_out, Trg_len, model_dim=512, n_heads=8, 
                dec_layers=4, fcn_dim=2048, dropout=0.0, device=torch.device('cuda:0')):
        super(TS_Fastformer, self).__init__()
        self.Trg_len = Trg_len

        self.LT_TPE = DataRepresentation(LT_in, model_dim, dropout)
        self.norm1 = torch.nn.LayerNorm(model_dim)
        
        self.ST_TPE = DataRepresentation(ST_in, model_dim, dropout)
        self.norm2 = torch.nn.LayerNorm(model_dim)

        self.Past_Att_Dec = PastAttentionDecoder(hidden_dim = model_dim, 
                            n_layers = dec_layers, n_heads = n_heads, pf_dim = fcn_dim,
                            dropout = dropout, device = device)
        
        self.linear = nn.Linear(model_dim, Trg_out, bias=True)

    def forward(self, x_LT_in, repr_LT_in, x_ST_in, repr_ST_in):
        LT_TPE_out = repr_LT_in + self.LT_TPE(x_LT_in)
        LT_TPE_out = self.norm1(LT_TPE_out)

        ST_TPE_out = repr_ST_in + self.ST_TPE(x_ST_in)
        ST_TPE_out = self.norm2(ST_TPE_out)
        dec_out, attention = self.Past_Att_Dec(ST_TPE_out, LT_TPE_out)
        dec_out = self.linear(dec_out)

        return dec_out[:,-self.Trg_len:,:] # [B, L, D]
