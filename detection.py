import json
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
import numpy as np
import pandas as pd
import argparse
#from wide_resnet import WideResNet
#from keras.utils.data_utils import get_file
import face_recognition
import cv2
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

model = tf.keras.models.load_model('/root/data1.h5')
model.summary()

age_map=[['0-2'],['4-6'],['8-13'],['15-20'],['25-32'],['38-43'],['48-63'],['60+']]

def detect_face(self):
    cap=cv2.VideoCapture(0)
    while True:
        grb,frame=cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not grb:
            break
        face_locations = face_recognition.face_locations(frame)
        print(face_locations)
        if(face_locations==[]):
            cv2.imshow('Gender and age', frame)
        if cv2.waitKey(1) == 27:
            break
        else:
            cv2.rectangle(frame, (face_locations[0][3], face_locations[0][0]), (face_locations[0][1], face_locations[0][2]), (255, 200, 0), 2)
            img=frame[face_locations[0][0]-25: face_locations[0][2]+25, face_locations[0][3]-25: face_locations[0][1]+25]
            img2 = cv2.resize(img, (64, 64))
            img2 = np.array([img2]).reshape((1,64,64,3))
            results = self.model.predict(img2)    
            predicted_genders = results[0]
            gen="F" if predicted_genders[0][0] > 0.5 else "M"
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()   
            pred=""
            pred=str(int(predicted_ages[0]))+" "+str(gen)
            print(pred)
            cv2.putText(frame, pred,(face_locations[0][3],face_locations[0][0]) , cv2.FONT_HERSHEY_SIMPLEX,0.7, (2, 255, 255), 2)
            cv2.imshow('Gender and age', frame)
            if cv2.waitKey(1) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
