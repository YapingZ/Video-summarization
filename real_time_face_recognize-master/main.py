from face_recognize import face_detection
import cv2
import time
import os


if __name__=='__main__':
    image_path = './output2/'
    threshold = 0.95
    face_detection(image_path,threshold)
