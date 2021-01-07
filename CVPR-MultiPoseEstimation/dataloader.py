import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import os 
from PIL import Image
import numpy as np 
import math
import json
import cv2

# change this to your dataset folder wher it includes crowdpose, lsp and coco14 folders
DATA_PATH = 'D:/DATASETS/'

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


class GenericPoseDataloader():
    def __init__(self, src_images, src_annotation, src_detected, cuda=False, batch=32):
        self.src_images = src_images 
        self.annotation = json.loads(open(src_annotation, 'r').read())
        self.detected = json.loads(open(src_detected, 'r').read()) 

        self.image_names = os.listdir(src_images)
        self.i = 0
        self.batch = batch
        self.cuda = cuda
        self.transform = transforms.ToTensor()
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __len__(self):
        return math.ceil(len(self.image_names) / self.batch)

    @property
    def length(self):
        return len(self.image_names)
    
    def __next__(self):
        if self.i >= len(self.image_names):
            raise StopIteration()
        image_names = self.image_names[self.i:self.i + self.batch]
        toloadcount = len(image_names)

        self.i += toloadcount

        images = []
        detected_annotation = []
        detected_detected = []

        for name in image_names:
            full_path = os.path.join(self.src_images, name)
            # tensor = self.transform(Image.open(full_path))
            # tensor = tensor.cuda() if self.cuda else tensor.cpu()

            images.append(cv2.imread(full_path))
            detected_annotation.append(self.annotation[name])
            detected_detected.append(self.detected[name])


        
        return images, detected_annotation, detected_detected, image_names

class GenericPoseDataloaderWithoutAnnotation():
    def __init__(self, src_images, src_detected, cuda=False, batch=32):
        self.src_images = src_images 
        self.detected = json.loads(open(src_detected, 'r').read()) 

        self.image_names = os.listdir(src_images)
        self.i = 0
        self.batch = batch
        self.cuda = cuda
        self.transform = transforms.ToTensor()
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __len__(self):
        return math.ceil(len(self.image_names) / self.batch)

    @property
    def length(self):
        return len(self.image_names)
    
    def __next__(self):
        if self.i >= len(self.image_names):
            raise StopIteration()
        image_names = self.image_names[self.i:self.i + self.batch]
        toloadcount = len(image_names)

        self.i += toloadcount

        images = []
        detected_detected = []

        for name in image_names:
            full_path = os.path.join(self.src_images, name)
            tensor = self.transform(Image.open(full_path))
            tensor = tensor.cuda() if self.cuda else tensor.cpu()

            images.append(tensor)
            detected_detected.append(self.detected[name])


        
        return images, detected_detected, image_names

class COCO14TrainLoader(GenericPoseDataloader):
    def __init__(self, dataset_folder='coco14/train2014', **kwargs):
        super().__init__(
            DATA_PATH + dataset_folder, 
            './datasets/annotations/coco_processed_train2014.json', 
            './datasets/preprocessed/yolo_detected_coco14_train2014.json', 
            **kwargs)

class COCO14ValLoader(GenericPoseDataloader):
    def __init__(self, dataset_folder='coco14/val2014', **kwargs):
        super().__init__(
            DATA_PATH + dataset_folder, 
            './datasets/annotations/coco_processed_val2014.json', 
            './datasets/preprocessed/yolo_detected_coco14_val2014.json', 
            **kwargs)

class COCO14TestLoader(GenericPoseDataloaderWithoutAnnotation):
    def __init__(self, dataset_folder='coco14/val2014', **kwargs):
        super().__init__(
            DATA_PATH + dataset_folder,
            './datasets/preprocessed/yolo_detected_coco14_val2014.json', 
            **kwargs)

class CrowdposeLoader(GenericPoseDataloader):
    def __init__(self, dataset_folder='crowdpose/set', **kwargs):
        super().__init__(
            DATA_PATH + dataset_folder, 
            './datasets/annotations/coco_processed_set.json', 
            './datasets/preprocessed/yolo_detected_crowdpose.json', 
            **kwargs)

class LSPLoader(GenericPoseDataloader):
    def __init__(self, dataset_folder='lsp/set', **kwargs):
        super().__init__(
            DATA_PATH + dataset_folder, 
            './datasets/annotations/lsp_processed_set.json', 
            './datasets/preprocessed/alphapose_detected_lsp.json', 
            **kwargs)

class Stepper:
    def __init__(self, loader, cuda=True):
        self.loader = loader
        self.inmemory = next(loader)
        self.i = 0
        self.cuda = cuda
    
    def __len__(self):
        return self.loader.length
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= len(self.inmemory[0]):
            for e in self.inmemory:
                del e
            self.inmemory = next(self.loader)
            self.i = 0
        
        self.i += 1
        gg = tuple([e[self.i-1] for e in self.inmemory])
        for e in gg:
            if torch.is_tensor(e) and self.cuda:
                e.cuda()
        
        return gg

'''
loader = COCO14TestLoader(batch=10, cuda=False)
images, detected_detected, image_names = next(loader)

i = 4
print((image_names[i]))
print((detected_detected[i]))
print('all values must be equal')
'''