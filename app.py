from flask import Flask, render_template
app = Flask(__name__)

#Importing Libraries 

import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pathlib
import cv2
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

filename = r"static\iim.jpg"

model = tf.keras.models.load_model("model_2.hdf5")
img1 = image.load_img(filename,target_size=(300,300))

Y = image.img_to_array(img1)
    
X = np.expand_dims(Y,axis=0)
val = model.predict(X) #val returns the predicted label 
if val == 1:
    result = "Jaguar"
else:
    result = "Cheetah"


@app.route('/')
def index():
   return render_template('./main.html',result=result,filename=filename)
