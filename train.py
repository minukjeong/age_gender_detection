import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import torch
import inspect
import urllib.request
from tensorflow.keras import optimizers
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Dropout, LayerNormalization, Input, BatchNormalization, SeparableConv2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pydot
from tensorflow.keras.utils import plot_model

fold_0 = pd.read_csv("/root/fold_0_data.txt", sep = "\t")
fold_1 = pd.read_csv("/root/fold_1_data.txt", sep = "\t")
fold_2 = pd.read_csv("/root/fold_2_data.txt", sep = "\t")
fold_3 = pd.read_csv("/root/fold_3_data.txt", sep = "\t")
fold_4 = pd.read_csv("/root/fold_4_data.txt", sep = "\t")

total_data = pd.concat([fold_0, fold_1, fold_2, fold_3, fold_4], ignore_index=True)

imp_data = total_data[['age', 'gender']].copy()
imp_data.info()

img_path = []
for row in total_data.iterrows():
    path = "/root/faces/"+row[1].user_id+"/coarse_tilt_aligned_face."+str(row[1].face_id)+"."+row[1].original_image
    img_path.append(path)

imp_data['img_path'] = img_path

age_mapping = [('(0, 2)', '0-2'), ('2', '0-2'), ('3', '0-2'), ('(4, 6)', '4-6'), ('(8, 12)', '8-13'), ('13', '8-13'), ('22', '15-20'), ('(8, 23)','15-20'), ('23', '25-32'), ('(15, 20)', '15-20'), ('(25, 32)', '25-32'), ('(27, 32)', '25-32'), ('32', '25-32'), ('34', '25-32'), ('29', '25-32'), ('(38, 42)', '38-43'), ('35', '38-43'), ('36', '38-43'), ('42', '48-53'), ('45', '38-43'), ('(38, 43)', '38-43'), ('(38, 42)', '38-43'), ('(38, 48)', '48-53'), ('46', '48-53'), ('(48, 53)', '48-53'), ('55', '48-53'), ('56', '48-53'), ('(60, 100)', '60+'), ('57', '60+'), ('58', '60+')]
age_mapping_dict = {each[0]: each[1] for each in age_mapping}
drop_labels = []
for idx, each in enumerate(imp_data.age):
    if each == 'None':
        drop_labels.append(idx)
    else:
        imp_data.age.loc[idx] = age_mapping_dict[each]
imp_data = imp_data.drop(labels=drop_labels, axis=0) #droped None values
imp_data.age.value_counts(dropna=False)
imp_data = imp_data.dropna()
clean_data = imp_data[imp_data.gender != 'u'].copy()

gender_to_label_map = {'f' : 0, 'm' : 1}
clean_data['gender'] = clean_data['gender'].apply(lambda g: gender_to_label_map[g])

age_to_label_map = {'0-2'  :0, '4-6'  :1, '8-13' :2, '15-20':3, '25-32':4, '38-43':5, '48-53':6, '60+':7}
clean_data['age'] = clean_data['age'].apply(lambda age: age_to_label_map[age])

clean_data.to_csv("/root/gender_age_detection1.csv")

df = pd.read_csv("/root/gender_age_detection1.csv", encoding = "utf-8")
df

x = df[['img_path']]
y1 = df[['gender']]
y2 = df[['age']]

x1_train, x1_test, y1_train, y1_test, y2_trian, y2_test = train_test_split(x, y1, y2, test_size=0.3, random_state=42)

train1_images = []
test1_images = []

for row in x1_train.iterrows():
    image = Image.open(row[1].img_path)
    image = image.resize((227, 227))   # Resize the image
    data = np.asarray(image)
    train1_images.append(data)
for row in x1_test.iterrows():
    image = Image.open(row[1].img_path)
    image = image.resize((227, 227))  # Resize the image
    data = np.asarray(image)
    test1_images.append(data)
    
train1_images = np.asarray(train1_images)
test1_images = np.asarray(test1_images)

train1_images = tf.convert_to_tensor(train1_images)
test1_images = tf.convert_to_tensor(test1_images)

print('Train images shape {}'.format(train1_images.shape))
print('Test images shape {}'.format(test1_images.shape))

resnet = ResNet50(weights="imagenet", input_shape=(227, 227, 3), include_top=False)
input = Input(shape=(227, 227, 3))
x = resnet(input)
x = GlobalAveragePooling2D()(x)
output_age = Dense(8, activation="relu")(x)
output_gender = Dense(2, activation="relu")(x)


def scheduler(epoch):
    if epoch<10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 *(10-epoch))

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model = Model(inputs = input, outputs = [output_age, output_gender])
model.compile(optimizer =  "adam", loss =["sparse_categorical_crossentropy","binary_crossentropy"],  metrics=['accuracy'])
model.summary()
h = model.fit(train1_images, y1_train, validation_data= (test1_images, y1_test), epochs = 50, batch_size=32, callbacks=[callback], shuffle = True)
model.save("data1.h5")

    

