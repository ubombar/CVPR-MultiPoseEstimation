import json 

input_file = 'alphapose-results-coco2014-val.json'
annotation = 'coco_processed_val2014.json'
selected = 'COCO_val2014_000000004554.jpg'

with open(input_file, 'r') as f:
    json_object = json.loads(f.read())
    
with open(annotation, 'r') as f:
    json_anno = json.loads(f.read())


bboxes = []
for elem in []: # json_object:
    if not selected in elem['image_id']: continue
    gg = json_anno[selected]
    for box in gg['boxes']
        x1, y1, x2, y2 = box
        bboxes.append([x1, y1, x2-x1, y2-y1])
    