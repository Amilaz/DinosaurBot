import cv2
from io import StringIO
from PIL import ImageGrab, Image
import numpy as np
import os
import random
import string
import re
from utils import const

def grab_screen(driver_image=None, x=None, y=None, height=None, width=None, image_sample=False):
    left = x
    top = max(y - 30, 0)
    right = x + width
    bottom = y + height
    screen = driver_image[top:bottom, left:right]
    image = process_img(screen) #processing image as required
    if image_sample:
        save_file_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))
        save_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image', '{}.jpg'.format(save_file_name))
        cv2.imwrite(save_file_path, image)
    return image

def process_img(image):
    # resale image dimensions
    image = cv2.resize(image, const.IMAGE_SIZE) 
    #crop out the dino agent from the frame
    image = cv2.Canny(image, threshold1 = 100, threshold2 = 200) #apply the canny edge detection
    return image