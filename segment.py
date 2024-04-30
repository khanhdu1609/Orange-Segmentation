import os
import time
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import os

class Segment():
    def __init__(self, model_path='models/Conveyor_segment.pb'):
        self.model = tf.saved_model.load(model_path)
    
    def segment(self, image):
        '''
        Get an binary image including orange mask(pixel 1) and back ground(pixel 0)
        '''
        image = image / 255.0
        image = np.expand_dims(image, axis = 0).astype(np.float32)
        p = self.model(image)
        mask = p[0] > 0.5
        mask = tf.cast(mask, dtype=tf.uint8)
        return mask
    def result(self, img, mask):
        '''
        Multiply the image with the mask, if pixel is orange, it will remain its value, otherwise, its pixel will change to zero
        '''
        return np.multiply(img, np.expand_dims(mask, axis=-1)) 

if __name__ == 'main':
    segment = Segment()
    image = cv2.imread('test img/A1_1.jpeg')
    mask = segment.segment(image)
    cv2.imwrite('test img/mask.jpg', mask)
    result = segment.result(image, mask)
    cv2.imwrite('test img/result.jpg', result)
    cv2.imshow('a', result)
    cv2.waitKey(0)

    
