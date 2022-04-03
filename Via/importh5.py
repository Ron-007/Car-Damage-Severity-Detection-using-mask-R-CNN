from keras.preprocessing.image import img_to_array
# from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import json
from keras.models import model_from_json
import glob
from tensorflow.keras.models import load_model

global model1
model1 = load_model('SideDetection.h5')
global modelfront
modelfront = load_model('Front.h5')
global modelseverity
modelfrontseverity = load_model('SeverityFront.h5')
global modelside
modelside = load_model('SideModel.h5')
global modelsideseverity
modelsideseverity = load_model('SideSeverity.h5')

img_dir = "examples/all"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
# labels =open("labels.txt", "r")
for f1 in files:
    image = cv2.imread(f1)
    output = imutils.resize(image)

    # pre-process the image for classification
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    second=["Front","Rear","Side"]
    proba = model1.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:]
    label1={}
    for (i, j) in enumerate(idxs):
        label1.update({second[j]:proba[j] * 100})
    print(f1)
    print(label1)

    if label1["Front"]>60:
        label2={}
        label3={}
        ############################################
        # for determing severity
        severityfront = ["0","0-20 %","20-40 %","40-70 %","80-100 %"]
        proba = modelfrontseverity.predict(image)[0]
        idxs = np.argsort(proba)[::-1][:]
        for (i, j) in enumerate(idxs):
            label3.update({severityfront[j]:proba[j] * 100})

        ############################################
        # Fo determing the part
        frontpart=["Bonnet","Bumper","GlassShatter"]
        proba = modelfront.predict(image)[0]
        idxs = np.argsort(proba)[::-1][:]
        # print(idxs)
        # print(type(idxs))
        for (j) in range(2):
            label2.update({"Damage":list(label3.keys())[0]})
            label2.update({frontpart[j]:proba[j] * 100})
        print(label2)

    if label1["Side"]>60:
        label4={}
        label5={}
         ############################################
        # for determing severity
        severityside = ["0","0-20 %","20-50 %","50-80 %","80-100 %"]
        proba = modelsideseverity.predict(image)[0]
        idxs = np.argsort(proba)[::-1][:]
        for (i, j) in enumerate(idxs):
            label5.update({severityside[j]:proba[j] * 100})

        side=["SideWindow","Door","SideMirror"]
        proba = modelside.predict(image)[0]
        idxs = np.argsort(proba)[::-1][:]
        label1={}
        for (i, j) in enumerate(idxs):
            label4.update({"Damage":list(label5.keys())[0]})
            label4.update({side[j]:proba[j] * 100})
        print(label4)