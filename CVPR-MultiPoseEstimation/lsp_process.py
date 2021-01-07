import scipy.io as io
import json
import numpy as np
import argparse
from glob import glob
from PIL import Image
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Crowdpose')
    parser.add_argument('--source', type=str, help='source')
    parser.add_argument('--dest', type=str, help='destination')

    args = parser.parse_args()

    mat = io.loadmat('./others/joints.mat')
    mat = mat["joints"]
    
    result = {}

    for k in range(2000):
        bbx1 = np.min(mat[0, :, k])
        bbx2 = np.max(mat[0, :, k])
        bby1 = np.min(mat[1, :, k])
        bby2 = np.max([mat[1, :, k]])
        #print(bbx1, bbx2, bby1, bby2)
        keypoints = list()
        for x, y in zip(mat[0, :, k], mat[1, :, k]):
            keypoints.append(x)
            keypoints.append(y)
            
           

        dirs = os.listdir("./set")
        
        name = ''
        for d in dirs:
            if str(k + 1) in d:
                name = d 
                break
        
        im = Image.open(f'./set/{name}')
        w,h = im.size
        result[str(d)] = {
            "boxes": [[bbx1, bby1, bbx2, bby2]],
            "keypoints": [keypoints],
            "size": [w , h]
        }

    out_file = open("myfile.json", "w")
    json.dump(result, out_file)
