import os
import cv2
import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils



def add_noise(tensor, sigma):
    sigma = sigma / 255 if sigma > 1 else sigma
    return tensor + torch.randn_like(tensor) * sigma






