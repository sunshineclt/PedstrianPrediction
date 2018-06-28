# PedstrianPrediction

This repo re-implement social LSTM and implement a new Spatial Pyramid Social LSTM proposed by us.  
This implementation is based on [https://github.com/vvanirudh/social-lstm-tf](https://github.com/vvanirudh/social-lstm-tf). We fix several bugs and vital errors in their repo. We also upgrade the tensorflow API to 1.8.0.

## File Structure
### data
- `getPixelCoordinates.m`: the matlab code to transform original ETH dataset to `pixel_pos.csv`, which is used in our code. This file is based on the referred implementation.
- `pixel_pos.csv`: the data file used by our code
- `transformed_data.pkl`: `pixel_pos.csv` will be transformed in our code and save as transformed_data.pkl

### social_lstm
- `DataLoader.py`: deal with data loading and preprocess
- `grid.py`: calculate grid or pyramid mask, called by `train.py`
- ***`model.py`***: IMPORTANT! all model (including social lstm and spatial pyramid social lstm) are defined here
- ***`social_sample.py`***: predict/test code, could be called using proper console parameters (use `social_sample.py --help` to see)
- `social_visualize.py`: to draw predicted graphs
- ***`train.py`***: train code, could be called using proper console parameters (use `train.py --help` to see)

#### plot
This directory contains several prediction plots for "Spatial Pyramid Social LSTM" method. 
Other method's plot can't be obtained since lab server is under maintenance (explained in our report). 

#### save
This directory contains model file for "Spatial Pyramid Social LSTM" method. 
Other method's model can't be obtained since lab server is under maintenance (explained in our report). 

## Usage
1. delete everything under `social_lstm/save/`
2. run `train.py` to train a model
3. run `social_sample.py` to predict
4. run `social_visualize.py` to visualize

## Authored by Letian Chen & Tianyang Zhao