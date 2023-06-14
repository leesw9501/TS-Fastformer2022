python -u TS-Fastformer-main.py KPVPG RUN --gpu 2 --Trg_len 24 --itr 1 --batch_size 128 --dec_layers 2 --dropout 0.1
python -u TS-Fastformer-main.py KPVPG RUN --gpu 2 --Trg_len 48 --itr 1 --batch_size 128 --dec_layers 2 --dropout 0.1
python -u TS-Fastformer-main.py KPVPG RUN --gpu 2 --Trg_len 168 --itr 1 --batch_size 128 --dec_layers 2 --model_dim 16
python -u TS-Fastformer-main.py KPVPG RUN --gpu 2 --Trg_len 336 --itr 1 --batch_size 128 --dec_layers 2 --model_dim 16
python -u TS-Fastformer-main.py KPVPG RUN --gpu 2 --Trg_len 720 --itr 1 --batch_size 128 --dec_layers 2 --model_dim 16