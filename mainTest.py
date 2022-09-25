from pickletools import pylist
from unittest import result
import cv2
import os
from mainTrain import INPUT_SIZE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model=load_model('BrainTumor10Epochs.h5')

image=cv2.imread('D:\\Minor Project medical analysis\\archive\\pred\\pred0.jpg')
    
img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result = model.predict(input_img)

result_final=np.argmax(result,axis=1)

print(result)