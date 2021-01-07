import torch
from util.dataloader import *
from util.rootnet import *
from tqdm import tqdm
from util.model import get_pose_net
from torch.nn.parallel.data_parallel import DataParallel
from util.rootnet import *
import os
from datetime import datetime
from datasetdemo import draw_pose

bbox_real = (2000, 2000)
focal = [1500, 1500] # x-axis, y-axis
input_shape = (256, 256)
output_shape = (input_shape[0]//4, input_shape[1]//4)

folder = os.path.join('runs', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) # 'runs\\' +  # str(datetime.strptime(datetime.now().strftime('%Y-%m-%d_%H-%M'), '%Y-%m-%d_%H-%M'))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

print(f'current folder {folder}')

os.mkdir(folder)
os.mkdir(os.path.join(folder, 'outputs'))
os.mkdir(os.path.join(folder, 'boxes'))
os.mkdir(os.path.join(folder, 'keypoints_annotation'))


def load_model(path='rootnet/rootnet_snapshot_18.pth.tar'):
    model = DataParallel(get_pose_net()).cuda()
    model.load_state_dict(torch.load(path)['network']) 
    model.eval()
    return model

def load_image(path):
    image = np.array(Image.open(path))
    height, width = image.shape[:2]

    image = transform(image)

    return image, width, height


model = load_model()

logging_object = []

print('model loaded!')
THRESH = 0.75

for image, annotation, detected, name in tqdm(Stepper(LSPLoader(batch=50))):
    image_array = image
    bbox_list = detected['boxes']
    bbox_length = len(bbox_list)
    width, height = tuple(annotation['size'])

    for n in range(bbox_length):
        if detected['confidences'][n] < THRESH: continue

        bbox = process_bbox(np.array(bbox_list[n]), width, height)
        img, img2bb_trans = generate_patch_image(image_array, bbox, False, 0.0)
        img = transform(img).cuda()[None,:,:,:]
        k_value = np.array([math.sqrt(bbox_real[0]*bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
        k_value = torch.FloatTensor([k_value]).cuda()[None,:]

        with torch.no_grad():
            root_3d = model(img, k_value) # x,y: pixel, z: root-relative depth (mm)
    
        img = img[0].cpu().numpy()
        root_3d = root_3d[0].cpu().numpy()

        vis_img = img.copy()
        vis_img = vis_img * np.array((0.229, 0.224, 0.225)).reshape(3,1,1) + np.array((0.485, 0.456, 0.406)).reshape(3,1,1)
        vis_img = vis_img.astype(np.uint8)
        vis_img = vis_img[::-1, :, :]
        vis_img = np.transpose(vis_img,(1,2,0)).copy()
        vis_root = np.zeros((2))
        vis_root[0] = root_3d[0] / output_shape[1] * input_shape[1]
        vis_root[1] = root_3d[1] / output_shape[0] * input_shape[0]
        cv2.circle(vis_img, (int(vis_root[0]), int(vis_root[1])), radius=5, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)

        out_name = f'{name}_{str(root_3d[2])}_{str(n)}.jpg'

        logging_object.append({
            'name': name,
            'n': n,
            'out': out_name ,
            'output': [int(e) for e in root_3d],
            'box': bbox_list[n],
            'size': [width, height],
            'confidence': detected['confidences'][n]
        })

        x1, y1, x2, y2 = bbox_list[n]

        image_boxed = cv2.rectangle(np.array(image), (int(x1), int(y1)), (int(x2)-int(x1), int(y2)-int(x2)), (0,255,0), 5)
        image_keypoints_annotation = draw_pose(image, annotation)
        # image_keypoints_detected = draw_pose(image, detected)

        cv2.imwrite(os.path.join(folder, 'outputs', out_name), vis_img)
        cv2.imwrite(os.path.join(folder, 'boxes', out_name), image_boxed)
        cv2.imwrite(os.path.join(folder, 'keypoints_annotation', out_name), image_keypoints_annotation)
        # cv2.imwrite(os.path.join(folder, 'keypoints_annotation', out_name), image_keypoints_annotation)


        # print('Root joint depth: ' + str(root_3d[2]) + ' mm ' + name)
    

with open(os.path.join(folder, 'log.json'), 'w+') as f:
    json.dump({'log': logging_object, 'tresh': THRESH}, f, indent=4)

'''
# load the whole batch into the memory then load it into the gpu one at a time
for image, annotation, detected, name in tqdm(Stepper(COCO14TrainLoader(batch=512))):
    pass
    # boxes = normalize_bbox(detected['boxes'], *annotation['size'])
    # img, img2bb_trans = generate_patch_image(image, boxes, False, 0.0)
    

    
            
def calculatekeyvalue():
    k_value = 1500
    return k_value
    

k_value = calculatekeyvalue()
print(k_value)
'''