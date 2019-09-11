import cv2
from input_data import *
img_path = r'/home/sai/Documents/ANPR/APNR_Detection/data/0c9ebe94-827d-4c74-9950-6816e70d1bab___IMG_8883.jpeg'

img = cv2.imread(img_path)
img = transform_image(img)
print(img.shape)
