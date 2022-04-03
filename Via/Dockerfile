# Specify your base image
FROM python:3.7.3-stretch
# create a work directory
RUN mkdir /app
# navigate to this work directory
WORKDIR /app
#Copy all files
COPY . .
# Install dependencies
RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y python-opencv
RUN pip install opencv-python
# RUN ENV/bin/activate
# RUN pip install tensorflow==2.4

# RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl 
# RUN pip install --upgrade pip setuptools wheel
# RUN pip install h5py==2.10.0
RUN pip install -r requirements.txt
# install tensorflow


# Run
CMD ["python","test.py"]