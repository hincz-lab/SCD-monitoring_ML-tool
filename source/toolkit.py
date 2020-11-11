from __future__ import division
import cv2 as cv
import pandas as pd
from skimage import measure 
from numba import jit
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import time
from tqdm import tqdm
import tensorflow.keras
from tensorflow.keras.models import load_model

# load model
def load_zoo(path):
    """
    load the ensemble model found in a given folder with the
    following directory: 'path'
    """
    ensemble = []
    for index, filename in enumerate(os.listdir(path)):
        model = load_model((path+filename), compile = False)
        ensemble.append(model)
    return ensemble
    
# load list of channel names
def list_channels(path):
    return os.listdir(path)

def list_channels_df(path):
    df = pd.read_csv(path)
    channel_filenames = df.filename
    return channel_filenames


