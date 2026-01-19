from torchvision import transforms
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math
from torch import nn
import os
from torchvision.datasets import ImageFolder
import kagglehub
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")