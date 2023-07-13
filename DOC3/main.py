from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from time import sleep
from threading import Thread

import tensorflow as tf
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import collections
from sklearn.svm import SVC
import time 
import RPi.GPIO as GPIO
from lock_control import unlock, lock
from testkeyboard import keypad
import khoangcach
from lock_control1 import unlock1, lock1

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

with open(CLASSIFIER_PATH, 'rb') as file:
   model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

def nhapphim():
    kp = keypad()
    list = ['']
    # Loop while waiting for a keypress
    digit = None
    while True:
        digit = kp.getKey()
        if digit != None and digit != '*':
            list.append(str(digit))
            time.sleep(.5) #otherwise you end up repeating the same number until a new one is pressed
            digit = None
        elif digit == '*':
            print (''.join(list))
            break
    return ''.join(list)

def unlockdoor():
    global choosecamera, dist
    unlock()
    time.sleep(10)
    lock()
    #GPIO.cleanup(7)
    print('login successfully')
    choosecamera = True

def unlockled():
    global dist
    unlock1()

def lockled():
    global dist
    lock1()

def facereco(count, dist,faces_found, bounding_boxes, frame,photo, choosecamera ):
    if(dist < 100 ):
        unlockled()
        print('Dist: ', dist)
        #show
        canvas.create_image(0,0, image = photo, anchor=tkinter.NW)
        button['text'] = ""
        if faces_found > 1 and choosecamera:
            cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (255, 255, 255), thickness=1, lineType=2)
        elif faces_found > 0 and choosecamera:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                print('i', faces_found)
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]
                if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    if (cropped.size != 0):
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[best_class_indices[0]]
                        #print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                        if best_class_probabilities > 0.80:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            name = class_names[best_class_indices[0]]
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1
                          *********************  print("Tên: {}, Khả năng dự đoán: {}".format(best_name, best_class_probabilities))
                            print('Mở cửa thành công -----------------------------------------')
                            choosecamera = False
                            unlockdoor()
                            count = 1
                            sleep(10)
                            choosecamera = True
    else:
        lockled()
        button['text'] = "Mời bạn vào khoảng cách 100cm"
def update_frame():
    global canvas, photo, count, choosecamera, dist
    dist = dist
    choosecamera = True
    #cap  = VideoStream(src=1).start()
    #fps = FPS (). start ()
    #print("fps",fps.fps())
    frame = cap.read()
    frame = imutils.resize(frame, width=300)
    frame = cv2.flip(frame, 1)
    bounding_boxes, _ = align.detect_face.detect_face(frame,  MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    faces_found = bounding_boxes.shape[0]
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
    dist  =  khoangcach.Distance()
    #dist = 50
    #facereco(count, dist,faces_found, bounding_boxes, frame,photo, choosecamera )
    try:
        facereco(count, dist,faces_found, bounding_boxes, frame,photo, choosecamera )
    except:
        pass
    if faces_found != 0:
        count = count + 1
    else:
        count = count
    print('Số khuôn mặt:', count)
    if count%10==0 and choosecamera:
        choosecamera = False
        print("Nhập mật khẩu rồi nhấn * để mở cửa\n Hoặc \nNhấn #* để lựa chọn nhận dạng khuôn mặt")
        kp = keypad()
        list = ['']
        digit = None
        while True:
            digit = kp.getKey()
            if digit != None and digit != '*':
                list.append(str(digit))
                time.sleep(.5) #otherwise you end up repeating the same number until a new one is pressed
                digit = None
            elif digit == '*':
                print (''.join(list))
                break
        choice  = ''.join(list)
        if choice == "123":
            print('Bạn mở khóa thành công với mật khẩu:')
            unlockdoor()

        elif choice == "#":
            print('Bạn lựa chọn tiếp tục mở cửa bằng khuôn mặt')
            facereco(count, dist,faces_found, bounding_boxes, frame,photo, choosecamera )

        else:
            print('Bạn nhập mật khẩu sai')
            facereco(count, dist,faces_found, bounding_boxes, frame,photo, choosecamera )

        count = 1
        choosecamera = True
    window.after(5, update_frame)

if __name__ == "__main__":
    
    with tf.Graph().as_default():
        gpu_options =  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder =  tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            people_detected = set()
            person_detected = collections.Counter()

            cap  = VideoStream(src=0).start()

        #create Tkinter
        window = Tk()
        window.title("Hệ thống mở cửa tự động")

        canvas = Canvas(window, width = 300, height= 230 , bg= "grey")
        canvas.pack()
        button = Button(window,text = "Mời bạn vào khoảng cách 100cm")
        button.pack()
        photo = None
        count = 1
        choosecamera = True
        #Distance = int(input('distance: '))
        dist  = 300
        while True:
            if choosecamera:
                update_frame()
            else:
                print("Error")
            window.mainloop()