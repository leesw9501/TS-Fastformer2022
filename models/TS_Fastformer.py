import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import time
import warnings

from utils.utils import EarlyStopping, adjust_learning_rate, metric
from utils.data_utils import data_provider
from models.models import PastAttentionDecoder

warnings.filterwarnings('ignore')

    
class Run_TS_Fastformer():
    def __init__(self, args):
        super(Run_TS_Fastformer, self).__init__()
        self.args = args

        self.patience = args.patience
        self.Trg_len = args.Trg_len
        self.scaler = args.scaler
        self.lr = args.lr
        self.epochs = args.epochs
        self.model = TS_Fastformer(
            In_dim = args.In_dim,
            LT_len = args.LT_len,
            ST_len = args.ST_len,
            Trg_len = args.Trg_len,
            LT_win = args.LT_win,
            ST_win = args.ST_win,
            model_dim = args.model_dim, 
            n_heads = args.n_heads,
            dec_layers = args.dec_layers, 
            fcn_dim = args.fcn_dim,
            dropout = args.dropout
        ).cuda()


    def _get_data(self, flag):

        data_set, data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_r, batch_y) in enumerate(vali_loader):
            
            batch_x = batch_x.float().cuda()
            batch_r = batch_r.float().cuda()
            true = batch_y.float().cuda()

            with torch.no_grad():
                pred, attn = self.model(batch_x, batch_r)
            
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
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_r, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().cuda()
                batch_r = batch_r.float().cuda()
                true = batch_y.float().cuda()

                pred, attn = self.model(batch_x, batch_r)
                
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

            adjust_learning_rate(model_optim, epoch+1, self.lr)
            
        best_model_path = path+'/'+'checkpoint_' + str(num) + '.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cuda:0')))
        
        return self.model

    def test(self, path, num, inverse):
        test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x, batch_r, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().cuda()
            batch_r = batch_r.float().cuda()
            true = batch_y.float().cuda()

            with torch.no_grad():
                pred, attn = self.model(batch_x, batch_r)
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


class TS_Fastformer(nn.Module):
    def __init__(self, In_dim, LT_len, ST_len, Trg_len, LT_win, ST_win, model_dim=128, n_heads=4, 
                dec_layers=3, fcn_dim=512, dropout=0.3, device=torch.device('cuda:0')):
        super(TS_Fastformer, self).__init__()
        self.In_dim = In_dim
        self.LT_win = LT_win
        self.ST_win = ST_win

        self.ST_proj = nn.Linear(model_dim, In_dim)

        self.linear_x = nn.Linear(LT_win, model_dim)
        self.linear_y = nn.Linear(ST_win, model_dim)

        pos_x = torch.empty((LT_len//LT_win, model_dim))
        nn.init.uniform_(pos_x, -0.01, 0.01)
        self.pos_x = nn.Parameter(pos_x, requires_grad=True)
        
        pos_r = torch.empty((ST_len//ST_win, model_dim))
        nn.init.uniform_(pos_r, -0.01, 0.01)
        self.pos_r = nn.Parameter(pos_r, requires_grad=True)

        self.Past_Att_Dec = PastAttentionDecoder(hidden_dim = model_dim, 
                            n_layers = dec_layers, n_heads = n_heads, pf_dim = fcn_dim,
                            dropout = dropout, device = device)
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(model_dim * (LT_len//LT_win), Trg_len)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, repr):
        r = self.ST_proj(repr)

        #dim = tuple(range(1, x.ndim-1))
        #print(dim)
        
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        x = x - self.mean

        x = x.permute(0,2,1)
        r = r.permute(0,2,1)

        x = x.unfold(dimension=-1, size=self.LT_win, step=self.LT_win) # x: [b x In_dim x token_num x window_len]
        r = r.unfold(dimension=-1, size=self.ST_win, step=self.ST_win)

        x = self.linear_x(x)
        r = self.linear_y(r)

        x = x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3])
        r = r.reshape(r.shape[0]*r.shape[1],r.shape[2],r.shape[3])

        x = self.dropout(x + self.pos_x)
        r = self.dropout(r + self.pos_r)
        
        x, attn = self.Past_Att_Dec(x, r)
        x = x.reshape(-1, self.In_dim, x.shape[-2], x.shape[-1])
        #x = x.permute(0,1,3,2)

        x = self.flatten(x)
        x = self.linear(x)
        x = x.permute(0,2,1)
        x = x + self.mean
        

        return x, attn


