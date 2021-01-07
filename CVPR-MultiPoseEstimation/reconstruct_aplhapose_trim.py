import json
from collections import defaultdict
import argparse
from PIL import Image
from tqdm import tqdm
# only use this for keypoints file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Crowdpose')
    parser.add_argument('--source', type=str, help='source')
    parser.add_argument('--dest', type=str, help='destination')

    args = parser.parse_args()
    
    array = None
    with open(args.source, 'r') as f:
        array = json.loads(f.read())
    
    output = []
    
    for element in tqdm(array):
        element['keypoints'] = element['keypoints'][:51]
        output.append(element)
        
    
    with open(args.dest, 'w+') as f:
        json.dump(output, f, indent=4)
    
'''

    array = None
    with open(args.source, 'r') as f:
        array = json.loads(f.read())

    # print(type(keypoints))
    id2obj = defaultdict(list)
    for e in array:
        id2obj[e['image_id']].append(dict(e))
    # id2size = {e['id']:[e['width'], e['height']] for e in array}


    id2kpbb = defaultdict(lambda: {
        "keypoints": [],
        "boxes": [],
        "scores": [],
        "confidences": []
    })

    for idname in tqdm(id2obj.keys()):

        temp_list = id2obj[idname][:51]
        tt = id2kpbb[idname]
        
        for temp in temp_list:
            conf = list(temp['keypoints'][2::3])
            confidence = sum(conf) / len(conf)
            
            # del temp['keypoints'][2::3] # remove c

            x1 = min(temp['keypoints'][0::2])
            y1 = min(temp['keypoints'][1::2])
            x2 = max(temp['keypoints'][0::2])
            y2 = max(temp['keypoints'][1::2])

            tt['keypoints'].append(temp['keypoints'][:51])
            tt['boxes'].append([x1, y1, x2, y2])
            tt['scores'].append(temp['score'])
            tt['confidences'].append(conf[:17])

        width, height = Image.open(f'./val2014/{idname}').size
        tt['scores'] = [width, height]
        
        break

    with open(args.dest, 'w+') as f:
        json.dump(id2kpbb, f)
    
    print('Done!')
'''
# # python process_coco.py --source 'person_keypoints_train2014.json' --dest 'coco_processed_train2014.json'
# # python process_crowdpose.py --source 'crowdpose_val.json' --dest 'crowdpose_processed_val.json'
# # python process_crowdpose.py --source 'crowdpose_train.json' --dest 'crowdpose_processed_train.json'