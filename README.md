# Transformer Next Location Prediction

A Transformer-based model that predicts the next GPS location from a sequence of past trajectory points. 

Nothing fancy or SoTA, I just use it for some simple systems evaluation.

## How it works

Given a sequence of N past `(lat, lon)` points, the model predicts the next location. It uses sliding windows to extract training samples from variable-length trajectories.

## Data format

CSV file with columns: `id, lon, lat, time`

The folder `data/` has a few examples of synthetic trajectories generated in an urban environment using [SUMO](https://sumo.dlr.de/docs/index.html). Their lat/lons have been anonimized as I previously used them on a paper where *I could not disclose the location*.

## Usage

**Train:**
```bash
python train.py path/to/trajectories.csv [path/to/save/model.pth]
```

**Predict:**
```bash
python predict.py
```

## Install dependencies

```bash
pip install -r requirements.txt
```
