import numpy as np
import torch
import os
import sys
import re
import math
from torch.utils.data import Dataset, DataLoader

dataset_name = 'neutral'#'interpolation'
if len(sys.argv) < 2:
    print("Missing data path!")
    exit()

datapath = os.path.join(sys.argv[1], dataset_name)
datapath_preprocessed = os.path.join(sys.argv[1], dataset_name+'_preprocessed')

os.mkdir(datapath_preprocessed)

all_data = os.listdir(datapath)

for filename in all_data:
    with np.load(os.path.join(datapath, filename)) as data:
        image = data['image'].astype(np.uint8).reshape(16, 160, 160)[:,::2,::2]
        target = data['target']
        np.savez_compressed(os.path.join(datapath_preprocessed, filename), image=image, target=target)

print('Done')
