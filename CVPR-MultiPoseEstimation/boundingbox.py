import torch
from PIL import Image
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.patches as patches


__utils = None
SSD_INPUT_SIZE = 300

def load_yolo(device='cuda', model_name='yolov5s'):
    yolo_model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    return yolo_model.to(torch.device(device))

def evaluate_yolo(image, model):
    '''
        image.shape = (W, H, 3)
        image.max = 255
        image.min = 0
        image.dtype = uint8
        numpy array
    '''
    model.eval()
    results = model([image], size=640)
    pred = results.xyxy[0]

    boxes = []
    for p in pred:
        if int(p[5]) != 0: continue
        (x1, y1, x2, y2, conf) = int(p[0]), int(p[1]), int(p[2]), int(p[3]), float(p[4])
        boxes.append([x1, y1, x2, y2, conf])
    return boxes

def evaluate_yolo_multiple(images, model):
    '''
        image.shape = (W, H, 3)
        image.max = 255
        image.min = 0
        image.dtype = uint8
        numpy array
    '''
    model.eval()
    results = model(images, size=640)

    image_boxes = []
    for i in range(len(images)):
        pred = results.xyxy[i]

        boxes = []
        for p in pred:
            if int(p[5]) != 0: continue
            (x1, y1, x2, y2, conf) = int(p[0]), int(p[1]), int(p[2]), int(p[3]), float(p[4])
            boxes.append([x1, y1, x2, y2, conf])
        image_boxes.append(boxes)
    return image_boxes

@DeprecationWarning
def load_ssd():
    global __utils

    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp16')
    __utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    return ssd_model

@DeprecationWarning
def evaluate_ssd(image, ssd_model, conf_thresh=0.20):
    '''
        image.shape = (300, 300, 3)
        image.max = 255
        image.min = 0
        image.dtype = uint8
        numpy array
    '''
    global __utils

    image = (image * 255 / 2 + 255 / 2).astype(np.uint8)
    tensor = __utils.prepare_tensor([image], True)
    ssd_model.to('cuda')
    ssd_model.eval()

    with torch.no_grad():
        detections_batch = ssd_model(tensor)
    
    results_per_input = __utils.decode_results(detections_batch)
    best_results_per_input = [__utils.pick_best(results, conf_thresh) for results in results_per_input]

    bboxes, classes, confidences = best_results_per_input[0]
    boxes = []
    for i in range(len(bboxes)):
        # if classes[i] != 1: continue
        x1, y2, x2, y1 = bboxes[i]
        conf = confidences[i]
        boxes.append((int(x1*300), int(y1*300), int(x2*300), int(y2*300), float(conf)))
    
    return boxes

def show(image, pred):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for (x1, y1, x2, y2, conf) in pred:
        rect = patches.Rectangle((min(x1, x2), min(y1, y2)), abs(x1-x2), abs(y1-y2), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, "{} {:.0f}%".format("Person", conf*100), bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
