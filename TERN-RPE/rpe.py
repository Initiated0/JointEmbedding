import os
import math
import time
import copy
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
# from torchviz import make_dot
import matplotlib.pyplot as plt

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

from pycocotools.coco import COCO
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, pipeline

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from pytorch_metric_learning import losses

import faiss


# Train DataLoader Configurations
trainloader_args = {
  'root_dir': "../../Data/flickr30k_images/flickr30k_images/flickr30k_images",
  'ann_path': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/flickr30k_annotations/train.json",
  'feat_dir': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/bottom_up",
  'text_dir': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/bert_features",
  'batch_size': 128,
  'num_workers': 0
}

# Validation DataLoader Configurations
valloader_args = {
  'root_dir': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/flickr30k_images/flickr30k_images/flickr30k_images",
  'ann_path': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/flickr30k_annotations/val.json",
  'feat_dir': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/bottom_up",
  'text_dir': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/bert_features",
  'batch_size': 128, 
  'num_workers': 0
}

# Dataset for Metric Evaluation -Here I use the validation set itself for metric evaluation
valset1_args = { # BottomUp Features
  'feat_dir': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/bottom_up",
  'ann_path': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/flickr30k_annotations/val.json"
}

# Dataset for Metric Evaluation -Here I use the validation set itself for metric evaluation
valset2_args = { # BERT Features
  'feat_dir': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/bert_features",
  'ann_path': "/Users/rishideychowdhury/Desktop/Joint-Embedding/Data/flickr30k_annotations/val.json"
}

device = torch.device('cuda') # Set it to 'cuda' for gpu or 'cpu' for cpu or 'mps' for M1
print(device)
print(torch.cuda.is_available())
