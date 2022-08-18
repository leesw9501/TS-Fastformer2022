# TS-Fastformer: Fast Transformer for Time-Series Forecasting

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.9.0](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

This repository contains the official implementation for the paper [Fast Transformer for Time-Series Forecasting].

## Requirements

- Python 3.6
- numpy == 1.19.5
- pandas == 1.1.5
- scikit-learn == 0.24.1
- torch == 1.9.0

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [ETT datasets](https://github.com/zhouhaoyi/ETDataset) should be placed at `data/ETTh1.csv`, `data/ETTh2.csv` and `data/ETTm1.csv`.
* KPVPG datasets should be placed at `data/KPVPG.csv`.
* [ELD dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) should be preprocessed using `data/preprocess_ELD.py` and placed at `data/ELD.csv`.
* [ECL dataset](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR) should be preprocessed using `data/preprocess_ECL.py`


## Usage
Commands for training and testing TS-Fastformer on KPVPG respectively:

```bash
python -u TS-Fastformer-main.py KPVPG RUN --gpu 0 --max_threads 8 --LT_len 48 --ST_len 24 --Trg_len 24 --itr 10 --batch_size 32

python -u TS-Fastformer-main.py KPVPG RUN --gpu 0 --max_threads 8 --LT_len 96 --ST_len 48 --Trg_len 48 --itr 10 --batch_size 32

python -u TS-Fastformer-main.py KPVPG RUN --gpu 0 --max_threads 8 --LT_len 336 --ST_len 168 --Trg_len 168 --itr 10 --batch_size 32

python -u TS-Fastformer-main.py KPVPG RUN --gpu 0 --max_threads 8 --LT_len 672 --ST_len 336 --Trg_len 336 --itr 10 --batch_size 32

python -u TS-Fastformer-main.py KPVPG RUN --gpu 0 --max_threads 8 --LT_len 1440 --ST_len 720 --Trg_len 720 --itr 10 --batch_size 32
```

More parameter information please refer to ```python -u TS-Fastformer-main.py -h```.

We provide a more detailed and complete command description for training and testing the model:

```python
python -u TS-Fastformer-main.py <dataset> <run_name>
--gpu <gpu> --seed <seed> --max_threads <max_threads> --root_path <root_path> --data_path <data_path> 
--target <target> --checkpoints <checkpoints> --inverse <inverse>
--pre_batch_size <pre_batch_size> --pre_lr <pre_lr> --pre_max_train_length <pre_max_train_length> 
--pre_iters <pre_iters> --pre_epochs <pre_epochs>
--dropout <dropout> --LT_len <LT_len> --ST_len <ST_len> --Trg_len <Trg_len> 
--LT_in <LT_in> --ST_in <ST_in> --Trg_out <Trg_out> 
--n_heads <n_heads> --dec_layers <dec_layers> --model_dim <model_dim> --fcn_dim <fcn_dim> 
--epochs <epochs> --batch_size <batch_size> 
--patience <patience> --lr <lr> --itr <itr>
```

The detailed descriptions about the arguments are as following:

| Parameter name | Description of parameter |
| --- | --- |
|dataset               |Dataset name|
|run_name              |The folder name used to save model, output and evaluation metrics. This can be set to any word|
|gpu GPU             |Gpu number|
|seed SEED           |Random seed|
|max_threads |The maximum allowed number of threads used by this process|
|root_path |Root path of the data file|
|data_path |Data file|
|target |Target column name|
|checkpoints |Location of model checkpoints|
|inverse |Inverse output data|
|pre_batch_size |Batch size of pretrain|
|pre_lr |Learning rate of pretrain|
|pre_max_train_length |For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>|
|pre_iters |Number of iterations|
|pre_epochs |Number of epochs|
|dropout |Dropout rate|
|LT_len | Long-term input length|
|ST_len |Short-term input length|
|Trg_len |Target length|
|LT_in |Long-term input dimension|
|ST_in |Short-term input dimension|
|Trg_out |Target dimension|
|n_heads |Num of heads|
|dec_layers |Num of decoder layers|
|model_dim |Dimension of model|
|fcn_dim |Dimension of fcn|
|epochs |Train epochs of TS-Fastformer|
|batch_size |Batch size of TS-Fastformer|
|patience |Early stopping patience|
|lr |Learning rate of TS-Fastformer|
|itr |Experiments times|

