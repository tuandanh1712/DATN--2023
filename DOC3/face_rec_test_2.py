# remake with no rashberry
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from time import time

import tensorflow as tf
from imutils.video import VideoStream


import argparse
import facenet
import imutils
import math
import pickle
import detect_face
import numpy as np
import cv2
import collections

import time

from gtts import gTTS
from pygame import mixer  # Load the popular external library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'DOC2/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'DOC2/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False))

        with sess.as_default():
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = detect_face.create_mtcnn(sess, "../DOC/DOC3/align")
            
            people_detected = set()
            person_detected = collections.Counter()

            cap  = VideoStream(src=0).start()
            
            fps_start_time = 0
            fps = 0
            
            count_time_regconize = 0
            
            name = ''
            
            while (True):
                frame = cap.read()
                frame = imutils.resize(frame, width=250, height=250)
                
                fps_end_time = time.time()
                time_diff = fps_end_time - fps_start_time
                fps = 1/time_diff
                fps_start_time = fps_end_time
                fps_text = "FPS: {:.2f}".format(math.ceil(fps))
                
                frame = cv2.flip(frame, 1)
                
                cv2.putText(frame, fps_text, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

                bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            print(bb[i][3]-bb[i][1])
                            print('----Resolution------------')
                            print(frame.shape[0])
                            print(frame.shape[1])
                            print('-----End Resolution-----------')                            
                            print((bb[i][3]-bb[i][1])/frame.shape[0])
                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))



                                if best_class_probabilities >0.9:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    best_name_class = class_names[best_class_indices[0]]
                                    if name != best_name_class:
                                        name = class_names[best_class_indices[0]];
                                        count_time_regconize = 0
                                    else: count_time_regconize = count_time_regconize +1    

                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    person_detected[best_name] += 1
                                    
                                    
                                    
                                    # cap.release()
                                    # cv2.destroyAllWindows()
                                    # exit()
                                else:
                                    name = "Unknown"
                                    print(name)
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                    
                    print('count time : ')
                    print(count_time_regconize)
                    if count_time_regconize == 50:
                        mytext = "successful identification, welcome, " + name
                        audio = gTTS(text=mytext, lang="en", slow=False)

                        audio.save("face_regconize_success.mp3")

                        mixer.init()
                        mixer.music.load('face_regconize_success.mp3')
                        mixer.music.play()
                
                except:
                    pass

                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


main()