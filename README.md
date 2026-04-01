# Accurate and robust cardiotoxicity prediction via integrating diverse multimodal fusion strategies

This is a PyTorch implementation of FusionCTox.

## Data splitting

Before running the codes, you need to split the datasets according to your requirements.
```
cd dataset/
mkdir datas
python split.py --ds herg/cav/nav --rs 0/1/2/3/4
```

## Training

Get fingerprints first
```
cd src/
mkdir fps
python get_fp.py
```

Train the four submodels first
```
mkdir ckpts results
python train_fp.py --data herg/cav/nav --rs 0/1/2/3/4
python train_concat.py --data herg/cav/nav --rs 0/1/2/3/4
python train_film.py --data herg/cav/nav --rs 0/1/2/3/4
python train_sum.py --data herg/cav/nav --rs 0/1/2/3/4
```

Obtain the final results of FusionCTox
```
python prediction.py --data herg/cav/nav --rs 0/1/2/3/4
```
