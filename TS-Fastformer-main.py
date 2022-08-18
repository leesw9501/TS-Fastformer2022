import torch
import argparse
import os
import time

from utils.utils import init_dl_program, name_with_datetime
from utils.data_utils import load_forecast_csv
from models.TS_Fastformer import Run_TS_Fastformer
from models.TS2Vec import TS2Vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('run_name', type=str, default='RUN', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--root_path', type=str, default='data/', help='Root path of the data file')
    parser.add_argument('--data_path', type=str, default='KPVPG.csv', help='Data file')
    parser.add_argument('--target', type=str, default='device_1', help='Target column name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
    parser.add_argument('--inverse', type=bool, default=False, help='Inverse output data')

    #pre-training model parameters
    parser.add_argument('--pre_batch_size', type=int, default=32, help='Batch size of pretrain')
    parser.add_argument('--pre_lr', type=float, default=0.001, help='Learning rate of pretrain ')
    parser.add_argument('--pre_max_train_length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>')
    parser.add_argument('--pre_iters', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--pre_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate')

    #TS-Fastformer model parameters
    parser.add_argument('--LT_len', type=int, default=7, help='Long-term input length')
    parser.add_argument('--ST_len', type=int, default=7, help='Short-term input length')
    parser.add_argument('--Trg_len', type=int, default=7, help='Target length')
    parser.add_argument('--LT_in', type=int, default=8, help='Long-term input dimension')
    parser.add_argument('--ST_in', type=int, default=8, help='Short-term input dimension')
    parser.add_argument('--Trg_out', type=int, default=1, help='Target dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Num of heads')
    parser.add_argument('--dec_layers', type=int, default=4, help='Num of decoder layers')
    parser.add_argument('--model_dim', type=int, default=512, help='Dimension of model')
    parser.add_argument('--fcn_dim', type=int, default=2048, help='Dimension of fcn')
    parser.add_argument('--epochs', type=int, default=10, help='Train epochs of TS-Fastformer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size of TS-Fastformer')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate of TS-Fastformer')
    parser.add_argument('--itr', type=int, default=10, help='Experiments times')
    
    args = parser.parse_args()

    print("Arguments:", str(args))


    
    args.device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols = load_forecast_csv(args.root_path, args.dataset, args.target)
    args.data_=data
    args.scaler = scaler

    train_data = data[:, train_slice]

    config = dict(
        batch_size=args.pre_batch_size,
        lr=args.pre_lr,
        output_dims=args.model_dim,
        max_train_length=args.pre_max_train_length
    )

    repr_model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=args.device,
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

    args.repr_model=repr_model
    
    save_path = '{}_LT{}_ST{}_Trg{}_dm{}_nh{}_dl{}_df{}_{}'.format(args.dataset, 
                    args.LT_len, args.ST_len, args.Trg_len, args.model_dim,
                    args.n_heads, args.dec_layers, args.fcn_dim, args.run_name)

    path = os.path.join(args.checkpoints, save_path)
    if not os.path.exists(path):
        os.makedirs(path)

    repr_model.save(f'{path}/repr_model.pth')

    torch.cuda.empty_cache()
    for ii in range(args.itr):

        # setting record of experiments

        TSF = Run_TS_Fastformer(args) # set experiments
        print('training : {}'.format(save_path))
        TSF.train(path, ii)

        print('testing : {}'.format(save_path))
        TSF.test(path, ii, args.inverse)

        torch.cuda.empty_cache()