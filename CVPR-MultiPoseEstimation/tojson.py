# converts the folder of images to json bounding boxes
import torch
import util.boundingbox as bb
import argparse
import json
from util.dataloader import GenericDataLoader
import util.boundingbox as bb 
from tqdm import tqdm

if __name__ == "__main__":
    # main func
    parser = argparse.ArgumentParser(description='Detect The Bounding Boxes!')
    parser.add_argument('--source', type=str, default='in', help='source')
    parser.add_argument('--dest', type=str, default='test.json', help='destination')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--batch', type=int, default=32, help='source')

    args = parser.parse_args()
    print('Parsed Arguments:', args)

    model = bb.load_yolo()
    model.eval()

    all_boxes = {}
    for batch, names in tqdm(GenericDataLoader(args.source, args.device, args.batch)):
        for item, name in zip(batch, names):
            with torch.no_grad():
                width = item.shape[1]
                height = item.shape[0]
                boxes = bb.evaluate_yolo(item, model)
                all_boxes[name] = {
                    'boxes': [e[:-1] for e in boxes],
                    'size': [width, height],
                    'confidences': [e[-1] for e in boxes],
                }
    
    with open(args.dest, 'w+') as outfile:
        json.dump(all_boxes, outfile, indent=4)

    print('Done!')
# python .\tojson.py --source 'D:\DATASETS\lsp\set' --dest 'results/yolo_detected_lsp.json' --batch 512
# python .\tojson.py --source 'D:\DATASETS\crowdpose\set' --dest 'results/yolo_detected_crowdpose.json' --batch 512
# python .\tojson.py --source 'D:\DATASETS\coco14\val2014' --dest 'results/yolo_detected_coco14_val2014.json' --batch 512
# python .\tojson.py --source 'D:\DATASETS\coco14\train2014' --dest 'results/yolo_detected_coco14_train2014.json' --batch 512
# python .\tojson.py --source 'D:\DATASETS\coco14\test2014' --dest 'results/yolo_detected_coco14_test2014.json' --batch 512
# I ahvent tested this on gpu because it were giving errors.

