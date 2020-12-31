import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import os 
from PIL import Image
import numpy as np 
import math

class GenericDataLoader():
    def __init__(self, source, device='cpu', batch=32):
        self.source = source 
        self.images = os.listdir(source)
        self.i = 0
        self.batch = batch
        self.device = torch.device(device)
        self.transform = transforms.ToTensor()
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __len__(self):
        return math.ceil(len(self.images) / self.batch)
    
    def __next__(self):
        if self.i >= len(self.images):
            raise StopIteration()
        toload = self.images[self.i:self.i + self.batch]
        toloadcount = len(toload)

        self.i += toloadcount

        accumilator = []
        for name in toload:
            full_path = os.path.join(self.source, name)
            # tensor = self.transform(Image.open(full_path))
            # tensor = tensor.to(self.device)
            ndarray = np.array(Image.open(full_path))
            accumilator.append(ndarray)
        
        return accumilator, toload


