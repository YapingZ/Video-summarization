#coding:utf-8
import tensorflow as tf
import numpy as np
import cv2
import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import random
import facenet
from scipy import misc
import pyttsx3
import argparse
import subprocess
import json
from PIL import Image, ImageDraw, ImageFont
#import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
dist = []
name_tmp = []
Emb_data = []
image_tmp = []

pic_store= 'picture'  #要检测人物的图片路径
image_size=160        #裁剪图片的人物图像的大小，为了对应facenet网络
pool_type='MAX'
use_lrn=False
seed=42,
batch_size= None

frame_interval=3 # 图像间隔，这是每三帧取一帧，可以随意设置数值

engine = pyttsx3.init()

def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret
#定义图像路径，gpu利用率
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    tmp_image_paths = []
    img_list = []
    path = pjoin(pic_store,image_paths)

    #print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, 'model_check_point/') # facnet 训练模型路径，我用的官方给的预训练模型

    if (os.path.isdir(path)):
        for item in os.listdir(path):
            #print(item)
            tmp_image_paths.insert(0,pjoin(path,item))
    else:
        tmp_image_paths=copy.copy(image_paths)

    #print(tmp_image_paths)
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)# 检测人脸，利用MTCNN
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          #print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        image_tmp.append(prewhitened)
    images = np.stack(img_list)
    return images,len(tmp_image_paths), pnet, rnet, onet



def face_detection(image_path,thresh_hold):
    dist = []
    name_tmp = []
    Emb_data = []
    image_tmp = []
    minsize = 20
    with tf.Graph().as_default():
        with tf.Session() as sess:
            #print('开始加载模型')
            # Load the model
            model_checkpoint_path='model_check_point/20180408-102900'
            facenet.load_model(model_checkpoint_path)

            #print('建立facenet embedding模型')
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            #print('模型建立完毕！')

            #print('载入人脸库>>>>>>>>')
            #start_time = time.time()
            for items in os.listdir(pic_store):
                emb_data1 = []
                name_tmp.append(items)
                images_tmp, count, pnet, rnet, onet = load_and_align_data(items,160,44,1.0)
                for i in range(9):
                    emb_data = sess.run(embeddings, feed_dict={images_placeholder: images_tmp, phase_train_placeholder: False })
                    emb_data = emb_data.sum(axis=0)
                    emb_data1.append(emb_data)
                emb_data1 = np.array(emb_data1)
                emb_data = emb_data1.sum(axis=0)
                Emb_data.append(np.true_divide(emb_data,9*count))
                #print(len(Emb_data))
            #print('-'*50)
            #nrof_images = len(name_tmp)

            for filename in os.listdir(image_path):
                frame = cv2.imread(image_path + filename)
                print('image {}'.format(image_path + filename))

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if gray.ndim == 2:
                    img = to_rgb(gray)

                bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

                if len(bounding_boxes) < 1:
                    #print('nobody appears :0')
                    return
                else:
                    img_size = np.asarray(frame.shape)[0:2]
                    #nrof_faces = bounding_boxes.shape[0]#一张图中的人脸数目
                    #print('找到人脸数目为：{}'.format(nrof_faces))

                    dist_matrix=np.zeros((1,2))
                    #print(dist_matrix.shape)

                    for item,face_position in enumerate(bounding_boxes):
                        #print('*'*50)
                        face_position=face_position.astype(int)  # 人脸的位置
                        #print((int(face_position[0]), int( face_position[1])))
                        det = np.squeeze(bounding_boxes[item,0:4])
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0]-44/2, 0)
                        bb[1] = np.maximum(det[1]-44/2, 0)
                        bb[2] = np.minimum(det[2]+44/2, img_size[1])
                        bb[3] = np.minimum(det[3]+44/2, img_size[0])
                        cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]

                        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')#裁剪人脸在图像中
                        prewhitened = facenet.prewhiten(aligned)#利用facenet进行矩阵对比
                        image_tmp.append(prewhitened)

                        image = np.stack(image_tmp)

                        emb_data = sess.run(embeddings, feed_dict={images_placeholder: image, phase_train_placeholder: False })

                        image_tmp.pop()


                        for i in range(len(Emb_data)):
                            dist.append(np.sqrt(np.sum(np.square(np.subtract(emb_data[len(emb_data)-1,:], Emb_data[i])))))

                        dist_np = np.array(dist)
                        dist_np=dist_np.reshape((1,len(Emb_data)))
                        #print(dist_np)
                        dist_matrix = np.concatenate((dist_matrix,dist_np),axis=0)
                        dist=[]

                    dist_matrix=np.delete(dist_matrix,0,axis=0)

                    min_index = np.argmin(dist_matrix,axis=0).tolist() # [2,2]
                    min_arg = np.min(dist_matrix,axis=0).tolist() # [0.66275054 0.91318572]
                    #start_time = time.time()
                    if min(min_arg) > thresh_hold :
                        print('i dont know this man')
                    elif max(min_arg)< thresh_hold :
                        if min_index[0]==min_index[1]:
                            a = min_arg.index(min(min_arg))
                            print('he is {}'.format(a+1))
                        else:
                            print('they are 1 and 2')
                    else:
                        b = min_arg.index(min(min_arg))
                        print('he is {}'.format(b+1))
                #print('time is {}'.format(time.time()-start_time))
                print ('*'*50)



if __name__ == '__main__':
    pass


