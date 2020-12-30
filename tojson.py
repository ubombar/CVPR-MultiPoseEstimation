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
                boxes = bb.evaluate_yolo(item, model)
                all_boxes[name] = boxes
    
    with open(args.dest, 'w+') as outfile:
        json.dump(all_boxes, outfile, indent=4)

    print('Done!')
# python .\tojson.py --source 'D:\DATASETS\lsp\images' --dest 'runs/lsp.json' --batch 512
# python .\tojson.py --source 'D:\DATASETS\crowdpose\images' --dest 'runs/crowdpose.json' --batch 512