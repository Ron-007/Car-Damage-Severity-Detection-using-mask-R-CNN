from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle 
import cv2
import tensorflow
import boto3
from flask import Flask, redirect, url_for, request, render_template
import flask
from werkzeug.utils import secure_filename
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
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
# global model
# model = load_model()




# def typeofcarmodel():  
#   model.load_weights('models/model.96-0.89.hdf5')
#   cars_meta = scipy.io.loadmat('devkit/cars_meta')
#   class_names = cars_meta['class_names']  # shape=(1, 196)
#   class_names = np.transpose(class_names)
#   global mlb0
#   mlb0 = class_names 
#   global model0
#   model0 = model
#   return model0

def load_model1():
  global model1
  model1 = load_model('DamageDetection.h5')  
  return model1

def load_model2():
  global model2
  model2 = load_model('SideDetection.h5')
  return model2

def load_model3():
  global model3
  model3 = load_model('Front.h5')
  return model3

def load_model4():
  global model4
  model4 = load_model('SideModel.h5')
  return model4

def load_model5():
  global model5
  model5 = load_model('SideSeverity.h5')
  return model5

def load_model6():
  global model6
  model6 = load_model('SeverityFront.h5')
  return model6

# def load_model3():
#   json_file = open('modelfront.json', 'r')
#   loaded_model_json = json_file.read()
#   json_file.close()
#   loaded_model = model_from_json(loaded_model_json)
#   loaded_model.load_weights('front.model')
#   print("Loaded frontal  model from disk")
#   global mlb3
#   mlb3 = pickle.loads(open('mlbfront.pickle', "rb").read())
#   print("Loaded frontal  model pickle")
#   # Load your trained model
#   global model3
#   model3 = loaded_model
#   #model._make_predict_function()
#   return model3

# def load_model4():
#   json_file = open('modelside.json', 'r')
#   loaded_model_json = json_file.read()
#   json_file.close()
#   loaded_model = model_from_json(loaded_model_json)
#   loaded_model.load_weights('side.model')
#   print("Loaded frontal  model from disk")
#   global mlb4
#   mlb4 = pickle.loads(open('mlbside.pickle', "rb").read())
#   print("Loaded frontal  model pickle")
#   # Load your trained model
#   global model4
#   model4 = loaded_model
#   #model._make_predict_function()
#   return model4


@app.route('/hello', methods=['GET'])
def hello():
  return "hello car"
  
@app.route('/predict', methods=['POST'])
def upload():
  # pdb.set_trace() 
  data = {"success": False,"predictions": []}
 
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
    

    model1=load_model1() # Damage or No damage
    proba = model1.predict(x)[0]
    idxs = np.argsort(proba)[::-1][:]
    #output = imutils.resize(x, width=400)
    output = imutils.resize(input_img, width=400)
    label1={}
    for (i, j) in enumerate(idxs):
      label1.update({j:proba[j] * 100})
    print(label1)
    if  label1[0] > 60 :
      label2={}
      model2 = load_model2() # Side detection 
      proba = model2.predict(x)[0]
      idxs = np.argsort(proba)[::-1][:]
      # counter = counter + 1
      output = imutils.resize(input_img, width=400)
      for (i,j) in enumerate(idxs):
        label2.update({j:proba[j] * 100})
      print(label2)
      if label2[0] > 40:  # If Front  
        label3={}
        model3 = load_model3()
        proba = model3.predict(x)[0]
        idxs = np.argsort(proba)[::-1][:]
        # counter = counter + 1
        output = imutils.resize(input_img, width=400)
        for (i,j) in enumerate(idxs):
          label3.update({j:proba[j] * 100})
        print(label3)
        storage.append(label3)
      if label2[1] > 40:
        label4={}
        model4 = load_model4()
        proba = model4.predict(x)[0]
        idxs = np.argsort(proba)[::-1][:]
        # counter = counter + 1
        output = imutils.resize(input_img, width=400)
        for (i,j) in enumerate(idxs):
          label4.update({j:proba[j] * 100})
        print(label4)
        storage.append(label4)
    os.remove(images) 
    
  if storage:
    print(storage)
    print("total Counter",counter)
    result = {} 
    for d in storage: 
      for k in d.keys(): 
          result[k] = result.get(k, 0) + d[k] 
    
    for key, value in list(result.items()):
      result[key]= result[key] / counter
    
    print("resultant dictionary : ",result) 

    ########################

    sorted(result)

      # Delete Values less then 2 
    for key, value in list(result.items()):
        if (value < 4):
            del result[key]
             

    data["predictions"] = []

    #Raji Requirement
    r = {"label": "Damaged", "probability": "True"}
    # data["predictions"].append(carName)
    data["predictions"].append(r)
    
    
    for label in (result):
        r = {"label": label, "probability": str(float("{:.2f}".format(result[label])))}
        l = {"label": "Damaged", "probability": "True"}
        print("typeof r=", type(r))
        print("contents of r=\n\t",r)
        data["predictions"].append(r)
      


    print("Type of object data", type(data))  

    print(json.dumps(data["predictions"], indent=4))
  
    ########################
    y= json.dumps({"result" : data["predictions"]}, indent=4 )
    print(y)
    return json.dumps({"result" : data["predictions"]})

  else:
    data["predictions"] = []
    l = {"label": "NoDamage", "probability": "True"}
    # data["predictions"].append(carName)
    data["predictions"].append(l)
    return json.dumps({"result" : data["predictions"]})

if __name__ == '__main__':
  print('Loading Keras model')
  # model0=typeofcarmodel()
  model1=load_model1()
  model2=load_model2()
  model3=load_model3()
  model4=load_model4()
  model5=load_model5()
  model6=load_model6()
  #app.run(debug=True)
  app.run(host='0.0.0.0',port=80)
