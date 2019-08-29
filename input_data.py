import cv2
import numpy as np
import os
import re
import json
from urllib import request

data_file_path = ''

formats = [
    '.jpg',
    '.jpeg',
    '.png',
    '.tif'
]

image_key = ''
annotation_key = ''


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
    with list(open(dat_file_path)) as f:
        for line in f[records_read:]:
            data = json.loads(line.rstrip())
            img_path = data[image_key]
            try:
                path, file = parse_name(img_path)
                file_names_split = file.split('.')
                root_name = file_names_split[0]
                extension = file_names_split[-1]

                img = request.urlopen(img_path).read()

                with open(file, 'wb') as f:
                    f.write(img)
                with open(root_name+'.txt', 'w') as f:
                    annotations = data[annotation_key]

                    for annotation in annotations:
                        annotation_string = ''
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
