# Jambo Algo

Jambo advertising Aalogrithm of NPL automatic forecasting procedure.

## Environment

tf1.12:

```bash
conda create -y -n tf1 python=3.6 tensorflow-gpu=1.12 keras=2.2.4 &&
conda activate tf1 &&
pip install pandas tqdm sklearn
# pip install glove_python gensim 
```

tf2.2

```bash
conda create -y -n tf2 python=3.8 tensorflow-gpu=2.2 &&
conda activate tf2 &&
pip install pandas tqdm sklearn
```

torch1.4

```bash
conda create -y -n torch python=3.8 pytorch=1.4.0 &&
conda activate torch &&
pip install gensim pandas tqdm sklearn
```
## Run

```bash
run.sh
```

Run sequence:

```bash
#!/bin/bash
preprocess/run.sh &&  # data preprocessing
get_emb/run.sh &&     # embedding
model/run.sh &&       # model training, 20 classifications for forecasting
oof/run.sh &&         # training result reduction, merge into model.npy, [4m, 20] matrix
ensemble/run.sh       # stacking of model oof with Ridge
echo "all done!"
```
