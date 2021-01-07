import torch
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image 
import json
import cv2

def load_pair(image_path, anno_path):
    with open(anno_path, 'r')as f:
        annotation = json.loads(f.read())
    
    image = np.array(Image.open(image_path))
    name = image_path.split('/')[-1]
    return image, annotation[name]

def plot_image(image_original, image_boxed, image_pose, i, dataset, add_title=True, height=3):
    plt.subplot(height, 3, 3 * i + 1)
    plt.imshow(image_original)
    plt.axis('off')
    if add_title: plt.title(dataset + ' original')

    plt.subplot(height, 3, 3 * i + 2)
    plt.imshow(image_boxed)
    plt.axis('off')
    if add_title: plt.title(dataset + ' bounding boxes')

    plt.subplot(height, 3, 3 * i + 3)
    plt.imshow(image_pose)
    plt.axis('off')
    if add_title: plt.title(dataset + ' pose')

def draw_bbox(image, annotation, dxy=(0, 0)):
    size = (image.shape[0] + image.shape[1]) // 250
    image_boxed = np.copy(image)
    for box in annotation['boxes']:
        x1, y1, x2, y2 = box
    
        image_boxed = cv2.rectangle(image_boxed, (int(x1 + dxy[0]), int(y1 + dxy[1])), (int(x2 + dxy[0]), int(y2 + dxy[1])), (0,255,0), size)

    return image_boxed

def draw_pose(image, annotation, dxy=(0, 0)):
    size = (image.shape[0] + image.shape[1]) // 200
    result = np.copy(image)
    for kps in annotation['keypoints']:
        for i in range(0, len(kps), 2):
            x, y = int(kps[i] + dxy[0]), int(kps[i + 1] + dxy[1])
            result = cv2.circle(result, (x,y), size, (255,0,0), -1)
    
    return result


coco_image, coco_anno = load_pair('./demods/COCO_val2014_000000001700.jpg', './datasets/annotations/coco_processed_val2014.json')
crowd_image, crowd_anno = load_pair('./demods/100481.jpg', './datasets/annotations/crowdpose_processed_set.json')
lsp_image, lsp_anno = load_pair('./demods/im0116.jpg', './datasets/annotations/lsp_processed_set.json')

coco_boxed = draw_bbox(coco_image, coco_anno)
crowd_boxed = draw_bbox(crowd_image, crowd_anno)
lsp_boxed = draw_bbox(lsp_image, lsp_anno)

coco_pose = draw_pose(coco_image, coco_anno)
crowd_pose = draw_pose(crowd_image, crowd_anno)
lsp_pose = draw_pose(lsp_image, lsp_anno)


# crop
coco_image = coco_image[:450, 150:450, :]
crowd_image = crowd_image
lsp_image = lsp_image[180:, 160:650, :]

coco_boxed = coco_boxed[:450, 150:450, :]
crowd_boxed = crowd_boxed
lsp_boxed = lsp_boxed[180:, 160:650, :]

coco_pose = coco_pose[:450, 150:450, :]
crowd_pose = crowd_pose
lsp_pose = lsp_pose[180:, 160:650, :]



plot_image(coco_image, coco_boxed, coco_pose, 0, 'Coco14')
plot_image(crowd_image, crowd_boxed, crowd_pose, 1, 'Crowdpose')
plot_image(lsp_image, lsp_boxed, lsp_pose, 2, 'LSP')

plt.show()


'''
print(coco_image.shape)

with plt.style.context("seaborn-white"):
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 3, 1)
    plt.imshow(coco_image[:450, 150:450, :]); plt.title('COCO 2014'); plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.imshow(lsp_image[180:, 160:650, :]); plt.title('LSP'); plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.imshow(crowd_image); plt.title('Crowdpose'); plt.axis('off')

plt.show()
'''