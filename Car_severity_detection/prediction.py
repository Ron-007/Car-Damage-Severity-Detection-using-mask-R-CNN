from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize


import colorsys
import random
import math
import numpy as np
import cv2
import boto3
import os

from flask import Flask, redirect, url_for, request, render_template
import flask
from werkzeug.utils import secure_filename
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import json
import imutils
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
model = None
mlb = None
img_path = None

#####################################
# Added for Object Detection

args = vars()
args["labels"] = "coco_labels.txt"
args["weights"] = "mask_rcnn_scratch_0050.h5"


# Load the class label names one at a time, one label per line.
CLASS_NAMES = open(args["labels"]).read().strip().split("\n")

# For each class label, produce a set of random (but visually different) colors.
# Matterport Mask R-CNN for the method
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

class SimpleConfig(Config):
	# give the configuration a recognizable name
	NAME = "coco_inference"

	# select the number of GPUs to be used, as well as the number of pictures
	# per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# number of classes (we would ordinarily add +1 for the background class, but it is *already* included in the class names)
	NUM_CLASSES = len(CLASS_NAMES)

# initialize the inference configuration
config = SimpleConfig()

# Load the weights into the Mask R-CNN model once it has been initialized for inference.
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

# functinality to save the model and make available to all Flask 
model.keras_model._make_predict_function()

#####################################


@app.route('/hello', methods=['GET'])
def hello():
  return "hello car"
  
@app.route('/predict', methods=['POST'])
def upload():
  data = {"success": False,"predictions": []}
 # if request.method == 'POST':
    # Get the file from post request
    #f = request.files['file']
  req_data = format(request.get_json()).replace('\'','"')
  print("POST Call Recieved"+req_data)
  x = json.loads(req_data)
  print(x['bucket']+'/Storage/'+x['object'])
  global bucket
  bucket=x['bucket']
  global img_path
  img_path = x['object']
  print(img_path)


  s3= boto3.resource('s3')
  key = 'Storage/'+ img_path
  s3.Bucket(bucket).download_file(key,img_path)
  print(" downloaded successfully in S3://"+bucket+"/"+img_path)
  image = cv2.imread(img_path)

  # image = cv2.imread(args["image"])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = imutils.resize(image, width=512)

  # perform a forward pass of the network to obtain the results
  print("[INFO] making predictions with Mask R-CNN...")
  r = model.detect([image], verbose=1)[0]

  ##########################

  for i in range(0, r["rois"].shape[0]):
	# extract the class ID and mask for the current detection, then grab the color to visualize the mask (in BGR format)
    classID = r["class_ids"][i]
    mask = r["masks"][:, :, i]
    color = COLORS[classID][::-1]

    # visualize the pixel-wise mask of the object
    image = visualize.apply_mask(image, mask, color, alpha=0.5)
    
    ###############################
    #SECOND PHASE

  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  print(r["scores"])
  # loop over the predicted scores and class labels
  for i in range(0, len(r["scores"])):
    # extract the bounding box information, class ID, label, predicted
    # probability, and visualization color
    if r["scores"][i] > 0.9:
      (startY, startX, endY, endX) = r["rois"][i]
      classID = r["class_ids"][i]
      label = CLASS_NAMES[classID]
      score = r["scores"][i]
      color = [int(c) for c in np.array(COLORS[classID]) * 255]

      # draw the bounding box, class label, and score of the object
      cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
      text = "{}: {:.3f}".format(label, score)
      y = startY - 10 if startY - 10 > 10 else startY + 10
      cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, color, 2)

      ####################################	
      print(startY, startX, endY, endX)
      x = (startX-endX)*(startX-endX)
      y = (startY-endY)*(startY-endY)
      global diagonal_len
      diagonal_len = math.sqrt(x+y)
      print("Diagonall of box",i)
      print(diagonal_len)
	####################################
  cv2.imwrite('OUT.jpg', image)
  # s3.upload_file(file_path,bucket_name, '%s/%s' % (bucket_folder,dest_file_name))
  # boto3.resource('s3').ObjectAcl(bucket,image).put(ACL='public-read')
  # s3 = boto3.client("s3")
  
  # s3.client.Bucket('storevia').put_object(Key='/{}'.format("Output.jpg"), Body=image)


  s2 = boto3.client('s3')
  filename = "OUT.jpg"
  s2.upload_file(filename, bucket,filename,ExtraArgs={"ACL":'public-read'})
            # ExtraArgs={
            #     "ACL": 'public-read',
            # }
        
  # s3 = boto3.resource('s3')
  # s3.Bucket('storevia').put_object(Key='random_generated_name.png', Body=image,ContentType='image/png',ACL='public-read')
  return flask.jsonify({"Diagonal Length" : diagonal_len})


if __name__ == '__main__':
  print('Loading Keras model')
  #app.run(debug=True)
  app.run(host='0.0.0.0',port=80)

# **License**
# Copyright 2021 Ronak Bhushan Patil, Bharat Tankala
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWAR