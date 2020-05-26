import os
import cv2

from detector import performDetect, convertBack
from config import *


def pred_parser(labels, img_dir, output_path, model_req):    
    for label in model_req:
        with open(labels[label]['path_test']) as fr:
            img_paths = fr.readlines()
        for img_path in img_paths:
            img_path = img_path.strip()
            image_prefix = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path)
            h, w = img[:2]
            detections = performDetect(img_path, model_req[label]['configPath'], model_req[label]['weightPath'], model_req[label]['metaPath'])
            for pred, confidence, bbox in detections:
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)
                with open(os.path.join(output_path[label], f'{image_prefix}.txt'), 'a+') as fa:
                    fa.write(f'{pred} {confidence} {xmin} {ymin} {xmax} {ymax}\n')
                
                
if __name__ == "__main__":
    for label, path in pred_save_dir.items():    
        if os.path.exists(path):
            os.system(f"rm {path}/*")
        os.makedirs(path, exist_ok=True)
    os.system(f"rm {img_dir}/*.txt")
    pred_parser(labels, img_dir, pred_save_dir, model_req)