import torch
import argparse
import os
import time

from utils.utils import init_dl_program
from utils.data_utils import load_forecast_csv
from models.TS_Fastformer import Run_TS_Fastformer
from models.cost import CoST


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('run_name', type=str, default='RUN', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number')
    parser.add_argument('--seed', type=int, default=1995, help='Random seed')
    parser.add_argument('--max_threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--root_path', type=str, default='data/', help='Root path of the data file')
    parser.add_argument('--target', type=str, default=None, help='Target column name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
    parser.add_argument('--inverse', type=bool, default=False, help='Inverse output data')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')

    #pre-training model parameters
    parser.add_argument('--pre_dim', type=int, default=128, help='Representation Vector Dimension of pretrain ')
    parser.add_argument('--pre_batch_size', type=int, default=128, help='Batch size of pretrain')
    parser.add_argument('--pre_lr', type=float, default=0.001, help='Learning rate of pretrain')
    parser.add_argument('--pre_iters', type=int, default=100, help='Number of iterations')
    parser.add_argument('--pre_epochs', type=int, default=100, help='Number of epochs')

    #TS-Fastformer model parameters
    parser.add_argument('--LT_len', type=int, default=336, help='Long-term input length')
    parser.add_argument('--ST_len', type=int, default=96, help='Short-term input length')
    parser.add_argument('--Trg_len', type=int, default=24, help='Target length')
    parser.add_argument('--In_dim', type=int, default=1, help='Input dimension')
    parser.add_argument('--Out_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--LT_win', type=int, default=16, help='Long-term window size')
    parser.add_argument('--ST_win', type=int, default=12, help='Short-term window size')
    parser.add_argument('--n_heads', type=int, default=16, help='Num of heads')
    parser.add_argument('--dec_layers', type=int, default=3, help='Num of decoder layers')
    parser.add_argument('--model_dim', type=int, default=128, help='Dimension of model')
    parser.add_argument('--epochs', type=int, default=100, help='Train epochs of TS-Fastformer')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of TS-Fastformer')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate of TS-Fastformer')
    parser.add_argument('--itr', type=int, default=1, help='Experiments times')
    
    
    args = parser.parse_args()
    args.fcn_dim = args.model_dim * 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    if args.target == None:
        name = args.dataset
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            args.target = 'OT'
        elif name == 'KPVPG':
            args.target = 'device_1'
        elif name == 'ECL':
            args.target = 'MT_320'
        elif name == 'ELD':
            args.target = 'MT_001'
        elif name == 'WTH':
            args.target = 'WetBulbCelsius'

    print("Arguments:", str(args))
    args.gpu = 0

    args.device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols = load_forecast_csv(args.root_path, args.dataset+'.csv', args.target, args.LT_len, True)
    args.data_=data
    args.scaler = scaler

    train_data = data[:, train_slice]

    config = dict(
        batch_size=args.pre_batch_size,
        lr=args.pre_lr,
        output_dims=args.model_dim
    )
    
    save_path = '{}_LT{}_ST{}_Trg{}_dm{}_nh{}_dl{}_df{}_{}'.format(args.dataset, 
                    args.LT_len, args.ST_len, args.Trg_len, args.model_dim,
                    args.n_heads, args.dec_layers, args.fcn_dim, args.run_name)

    path = os.path.join(args.checkpoints, save_path)
    if not os.path.exists(path):
        os.makedirs(path)

    

    torch.cuda.empty_cache()
    for ii in range(args.itr):
        repr_model = CoST(
            input_dims=train_data.shape[-1],
            kernels=[1, 2, 4, 8, 16, 32, 64, 128],
            alpha=0.0005,
            max_train_length=args.ST_len,
            device=torch.device('cuda:{}'.format(args.gpu)),
            **config
        )

        em_time=time.time()
        loss_log = repr_model.fit(
            train_data,
            n_epochs=args.pre_epochs,
            n_iters=args.pre_iters,
            verbose=False
        )
        print('em_time:', time.time() - em_time)


        print('data.shape', data.shape)
        em_time=time.time()
        all_repr = repr_model.encode(
            data,
            mode='forecasting',
            casual=True,
            sliding_length=1,
            sliding_padding=args.ST_len-1,
            batch_size=args.pre_batch_size
        )
        print('em_encode:', time.time() - em_time)

        args.repr = dict()
        args.repr['train'] = all_repr[:, train_slice]
        args.repr['val'] = all_repr[:, valid_slice]
        args.repr['test'] = all_repr[:, test_slice]

        print(args.repr['train'].shape)
        print(args.repr['val'].shape)
        print(args.repr['test'].shape)

        # setting record of experiments
        TSF = Run_TS_Fastformer(args) # set experiments
        print('training : {}'.format(save_path))
        TSF.train(path, ii)

        print('testing : {}'.format(save_path))
        TSF.test(path, ii, args.inverse)

        torch.cuda.empty_cache()
