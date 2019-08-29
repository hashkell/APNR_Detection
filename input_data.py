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


def parse_name(img_path):
    '''
    parses  a given image file img_path, a URL or Localpath
    returns a tuple containing image's path, file_name
    and a boolean (True if image has allowed file format)

    '''
    path, file = os.path.split(img_path)
    for f in formats:
        if f in file:
            return path, file, True
    return path, file, False


def gen_data_from_json(feature_type='nd', target_vector='yolo'):
    with open(dat_file_path) as f:
        for line in f:
            data = json.loads(line)
            img_path = data[image_key]
            try:
                img_attribs = parse_name(img_path)

                img = request.urlopen(img_path)
                pass
            except e:
                pass
