python -u TS-Fastformer-main.py ECL RUN --gpu 0 --max_threads 8 --LT_len 48 --ST_len 24 --Trg_len 24 --itr 10 --batch_size 32

python -u TS-Fastformer-main.py ECL RUN --gpu 0 --max_threads 8 --LT_len 96 --ST_len 48 --Trg_len 48 --itr 10 --batch_size 32

python -u TS-Fastformer-main.py ECL RUN --gpu 0 --max_threads 8 --LT_len 336 --ST_len 168 --Trg_len 168 --itr 10 --batch_size 32

python -u TS-Fastformer-main.py ECL RUN --gpu 0 --max_threads 8 --LT_len 672 --ST_len 336 --Trg_len 336 --itr 10 --batch_size 32

python -u TS-Fastformer-main.py ECL RUN --gpu 0 --max_threads 8 --LT_len 1440 --ST_len 720 --Trg_len 720 --itr 10 --batch_size 32
