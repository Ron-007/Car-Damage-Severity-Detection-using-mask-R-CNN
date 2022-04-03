from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle 
import cv2
import tensorflow
import boto3
from flask import Flask, redirect, url_for, request, render_template
# import flask
from werkzeug.utils import secure_filename
# from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
#from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
import json
import imutils
from gevent.pywsgi import WSGIServer
import os

import scipy.io

# Define a flask app
# from utils import load_model

app = Flask(__name__)
mlb = None
img_path = None
global model


def load_model1():
  global model1
  model1 = load_model('DamageDetection.h5')  
  return model1

def load_model2():
  global model2
  model2 = load_model('SideDetection.h5')
  return model2


@app.route('/hello', methods=['GET'])
def hello():
  return "hello car"
  
@app.route('/predict', methods=['POST'])
def upload():
  # pdb.set_trace() 
  data = {"success": False,"predictions": []}
  print(request.get_json())
  req_data = format(request.get_json()).replace('\'','"')
  
  print("POST Call Recieved"+req_data)
  x = json.loads(req_data)
  #print(x['bucket']+'/Incoming/'+x('object'))
  global bucket
  bucket=x['bucket']
  global img_path

  totalobject = x['object']
  print(totalobject)
  global df_add
  
  global threshold1
  threshold1 = 70
  # threshold1 = int(x['threshold1'])
  global threshold2
  threshold2 = 6
  # threshold2 = int(x['threshold2'])

  global storage
  storage = []

  global carName

  length = len(totalobject)

  global counter
  counter = 1

  for images in totalobject :
    s3= boto3.resource('s3')
    key = 'Storage/'+ images
    s3.Bucket(bucket).download_file(key,images)
    print(" downloaded successfully in S3://"+bucket+"/"+ images)
    input_img = cv2.imread(images)
    img = cv2.resize(input_img, (224,224))
    img = img.astype("float") / 255.0
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    #x = preprocess_input(x, mode='caffe')
    #######################
    img0 = cv2.resize(input_img, (224, 224))
    y = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    y = np.expand_dims(y, axis=0)
  


    model1=load_model1()
    proba = model1.predict(x)[0]
    idxs = np.argsort(proba)[::-1][:]
    #output = imutils.resize(x, width=400)
    output = imutils.resize(input_img, width=400)
    label1={}
    for (i, j) in enumerate(idxs):
      label1.update({str(j):str(proba[j] * 100)})
    print(label1)
    storage.append(label1)

    os.remove(images) 
  
  
  # print(float("{:.2f}".format(storage)))
  
  print(storage)
  y= json.dumps({"result" : storage[0]}, indent=4 )  
  print(y)
  return json.dumps({"result" : storage})


if __name__ == '__main__':
  print('Loading Keras model')
  # model0=typeofcarmodel()
  model1=load_model1()

  #app.run(debug=True)
  app.run(host='0.0.0.0',port=80)
