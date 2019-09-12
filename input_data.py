import cv2
import numpy as np
import os
import glob
import re
import json
from urllib import request
from urllib.error import URLError
from pprint import pprint
data_file_path = './Nplate.json'
data_store_path = './data/'
formats = ['.jpg', '.jpeg', '.png', '.tif']

image_key = ''
annotation_key = ''
class_key = ''
points_key = ''
records_read = 0
n = 7
nx = 7
ny = 7
ppg = 70
n_bboxes = 2
with open('./classes.txt', 'r') as f:
    classes = eval(f.read())
    classes = {x: i for i, x in enumerate(classes)}
with open('./iosummary.txt', 'r') as f:
    summary = eval(f.read())
    records_read = summary['records_read']
with open('./config/ioconfig.txt', 'r') as f:
    config = eval(f.read())
    image_key = config['image_key']
    annotation_key = config['annotation_key']
    points_key = config['points_key']
    class_key = config['class_key']
with open('./config/netconfig.txt', 'r') as f:
    netconfig = eval(f.read())
    inpt_res = netconfig['inpt_res']
    grid_size = netconfig['grid_size']

image_attributes = {
    'grid_size': n,
    'ppg': ppg,
    'img_res': (nx*ppg, ny*ppg),
    'n_channels': 1,
    'classes': classes
}


def parse_name(img_path):
    '''
    parses  a given image file img_path, a URL or Localpath
    returns a tuple containing image's path, file_name
    args :
        img_path : a url to image or local path
    '''

    path, file = os.path.split(img_path)
    return path, file


def gen_data_from_json():
    global records_read
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
                if extension not in formats :
                    continue
                img = request.urlopen(img_path).read()

                with open(data_store_path+root_name+'.'+extension, 'wb') as f:
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

                        f.write(str(yolo_vector)+'\n')
                records_read += 1
            except URLError:

                continue
            except RuntimeError:

                summary = str(
                    {
                        'records_read': records_read,
                        'data_read': False
                    }
                )
                with open('iosummary.txt', 'w') as f:
                    f.write(summary)
            else:
                summary = str(
                    {
                        'records_read': records_read,
                        'data_read': False,
                    }
                )
                with open('iosummary.txt', 'w') as f:
                    f.write(summary)
    with open('./iosummary.txt', 'w') as f:
        f.write(summary)


def transform_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (nx*ppg, ny*ppg))
    return img.reshape(img.shape+(1,))


def gen_yolo_data():
    images = []
    target_vectors = []
    all_files = set(glob.glob('./data/*'))
    annotation_files = set(glob.glob('./data/*.txt'))
    img_files = all_files-annotation_files
    def sortf(x): return os.path.split(x)[-1].split('.')[0]
    img_files, annotation_files = (sorted(list(img_files), key=sortf),
                                   (sorted(list(annotation_files), key=sortf)))
    for image, annotation in zip(img_files, annotation_files):
        try :
            
            images.append(transform_image(cv2.imread(image)))
        except  :
            print(image)
            continue
        target_vector = np.zeros((nx, ny, n_bboxes*5+len(classes)))
        with open(annotation, 'r') as f:
            targets = []
            for line in f:
                try:
                    targets.append(eval(line))
                    
                except TypeError :
                    print(line)
                    print(annotation)
            targets = sorted(targets, key=lambda x: x[-1]*x[-2])
            dx = 1/nx
            dy = 1/ny
            for target in targets:
                cl, cx, cy, w, h = target
                xllt = int(max(0, (cx-w/2)//dx))
                xult = int(min(nx-1, (cx+w/2)//dx))
                yllt = int(max(0, (cy-h/2)//dy))
                yult = int(min((cy+h/2)//dy, ny-1))
                for i in range(xllt, xult+1):
                    for j in range(yllt, yult+1):
                        for k in range(n_bboxes):

                            target_vector[i][j][k*5] = 1
                            target_vector[i][j][k*5+1] = cx
                            target_vector[i][j][k*5+2] = cy
                            target_vector[i][j][k*5+3] = w
                            target_vector[i][j][k*5+4] = h
                        for l in range(len(classes)):
                            target_vector[i][j][n_bboxes*5+l] = 0
                        target_vector[i][j][n_bboxes*5+cl] = 1
            target_vectors.append(target_vector.reshape(-1,))
    return [images, target_vectors]


if __name__ == '__main__':

    if not summary['data_read']:
        gen_data_from_json()
