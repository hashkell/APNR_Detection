import cv2
import numpy as np
import os
import re
import json
from urllib import request

data_file_path = './Nplate.json'
data_store_path = './data/'
formats = [
    '.jpg',
    '.jpeg',
    '.png',
    '.tif'
]

image_key = ''
annotation_key = ''
records_read = 0


def parse_name(img_path):
    '''
    parses  a given image file img_path, a URL or Localpath
    returns a tuple containing image's path, file_name
    args :
        img_path : a url to image or local path
    '''

    path, file = os.path.split(img_path)
    return path, file


def gen_data_from_json(feature_type='nd', target_vector='yolo'):
    records_read = 0
    with open(data_file_path, 'r') as f:
        f = list(f)
        for line in f[records_read:]:
            data = json.loads(line.rstrip())
            img_path = data[image_key]
            try:
                path, file = parse_name(img_path)
                file_names_split = file.split('.')
                root_name = file_names_split[0]
                extension = file_names_split[-1]

                img = request.urlopen(img_path).read()

                with open(data_store_path+root_name+extension, 'wb') as f:
                    f.write(img)
                with open(data_store_path+root_name+'.txt', 'w') as f:
                    annotations = data[annotation_key]
                    for annotation in annotations:
                        lbl = classes[annotation[class_key][0]]
                        p1, p2 = annotation[points_key]
                        x1, y1 = p1['x'], p1['y']
                        x2, y2 = p2['x'], p2['y']
                        cx = (x1+x2)/2
                        cy = (y1+y2)/2
                        w = abs(x1-x2)
                        h = abs(y1-y2)
                        yolo_vector = [lbl, cx, cy, w, h]
                        annotation_string = ''
                        for val in yolo_vector:
                            annotation_string += str(val)+' '
                        f.write(annotation_string)
                records_read += 1
            except e:
                summary = str(
                    {
                        'records_read': records_read,
                    }
                )
                with open('summary.txt', 'w') as f:
                    f.write(summary)


if __name__ == '__main__':
    with open('summary.txt', 'r') as f:
        summary = eval(f.read())
        records_read = summary['records_read']
    with open('./config/ioconfig.txt', 'r') as f:
        config = eval(f.read())
        image_key = config['image_key']
        annotation_key = config['annotation_key']
        points_key = config['points_key']
        class_key = config['class_key']
