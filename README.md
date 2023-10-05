# TS-Fastformer: Fast Transformer for Time-Series Forecasting

![Python 3.8.10](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 2.0.1](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

This repository contains the official implementation for the paper [TS-Fastformer: Fast Transformer for Time-Series Forecasting].

## Requirements

- Python 3.8.10
- numpy==1.23.5
- pandas==2.0.1
- scikit-learn==1.2.2
- torch==2.0.0+cu118
- einops==0.6.1

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

The datasets can be obtained and put into `data/` folder in the following way:

* [ETT datasets](https://github.com/zhouhaoyi/ETDataset) should be placed at `data/ETTh1.csv`, `data/ETTh2.csv`, `data/ETTm1.csv` and `data/ETTm2.csv`.
* [KPVPG datasets](https://github.com/leesw9501/TS-Fastformer2022/tree/main/data) should be placed at `data/KPVPG.csv`.
* [WTH dataset](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR) should be placed at `data/WTH.csv`.
* [ECL dataset](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR) should be placed at `data/ECL.csv`.



## Usage
Commands for training and testing TS-Fastformer on WTH:

```bash
python -u TS-Fastformer-main.py WTH RUN --gpu 2 --Trg_len 720 --itr 1 --batch_size 128 --model_dim 128
```

More parameter information please refer to ```python -u TS-Fastformer-main.py -h```.

We provide a more detailed and complete command description for training and testing the model:

```python
python -u TS-Fastformer-main.py <dataset> <run_name>
--gpu <gpu> --seed <seed> --max_threads <max_threads> --root_path <root_path>
--target <target> --checkpoints <checkpoints> --inverse <inverse> --dropout <dropout>
--pre_dim <pre_dim> --pre_batch_size <pre_batch_size> --pre_lr <pre_lr> 
--pre_max_train_length <pre_max_train_length> --pre_iters <pre_iters> --pre_epochs <pre_epochs>
--LT_len <LT_len> --ST_len <ST_len> --Trg_len <Trg_len> 
--In_dim <In_dim> --Out_dim <Out_dim> --LT_win <LT_win> --ST_win <ST_win> 
--n_heads <n_heads> --dec_layers <dec_layers> --model_dim <model_dim>
--epochs <epochs> --batch_size <batch_size> 
--patience <patience> --lr <lr> --itr <itr>
```

The detailed descriptions about the arguments are as following:

| Parameter name | Description of parameter |
| --- | --- |
|dataset               |Dataset name|
|run_name              |The folder name used to save model, output and evaluation metrics. This can be set to any word|
|gpu             |Gpu number|
|seed           |Random seed|
|max_threads |The maximum allowed number of threads used by this process|
|root_path |Root path of the data file|
|target |Target column name|
|checkpoints |Location of model checkpoints|
|inverse |Inverse output data|
|dropout |Dropout rate|
|pre_dim |Representation Vector Dimension of pretrain|
|pre_batch_size |Batch size of pretrain|
|pre_lr |Learning rate of pretrain|
|pre_iters |Number of iterations|
|pre_epochs |Number of epochs|
|LT_len | Long-term input length|
|ST_len |Short-term input length|
|Trg_len |Target length|
|In_dim |Input dimension|
|Out_dim |Output dimension|
|LT_win |Long-term window size|
|ST_win |Short-term window size|
|n_heads |Num of heads|
|dec_layers |Num of decoder layers|
|model_dim |Dimension of model|
|epochs |Train epochs of TS-Fastformer|
|batch_size |Batch size of TS-Fastformer|
|patience |Early stopping patience|
|lr |Learning rate of TS-Fastformer|
|itr |Experiments times|
