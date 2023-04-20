import tensorflow as tf
import numpy as np
from sklearn.datasets import load_files
from keras.utils import np_utils
import cv2
import json
import sys
from PIL import Image
import requests
from io import BytesIO

def face_detector(img):
    face_cascade = cv2.CascadeClassifier('Data/haarcascades/haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def define_model(input_shape, output_neurons):
    '''Returns a model defined on the basis of the input shape and output neurons'''
    Resnet50_model = tf.keras.models.Sequential()
    Resnet50_model.add(tf.keras.layers.GlobalAveragePooling2D(input_shape=input_shape))
    Resnet50_model.add(tf.keras.layers.Dense(1024, activation='relu'))
    Resnet50_model.add(tf.keras.layers.Dense(output_neurons, activation='softmax'))

    return Resnet50_model

def load_dataset(dataset_path):
    '''Returns image paths and labels from the given path'''
    data = load_files(dataset_path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 133)
    return files, targets

def label_to_category_dict(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return data

def extract_Resnet50(tensor):
    '''Returns the VGG16 features of the tensor'''
    return tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False).predict(tf.keras.applications.resnet50.preprocess_input(tensor))

def path_to_tensor(img):
    '''Converts the image in the given path to a tensor'''
    img = cv2.resize(img, (224, 224)) 
    x = np.array(img)

    return np.expand_dims(x, axis=0)

def predict_breed(img_path):
    '''Predicts the breed of the given image'''
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    bottleneck_feature = tf.keras.models.Sequential([
                            tf.keras.layers.GlobalAveragePooling2D(input_shape=bottleneck_feature.shape[1:])
                        ]).predict(bottleneck_feature).reshape(1, 1, 1, 2048)
    # obtain predicted vector
    Resnet50_model = define_model((1, 1, 2048), 133)
    Resnet50_model.load_weights('saved_models/weights_best_Resnet50.hdf5')
    
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    Resnet50_model = None
    bottleneck_feature = None

    label_to_cat = label_to_category_dict('./label_to_cat.json')
    
    return label_to_cat[str(np.argmax(predicted_vector))]


def dog_breed_classifier(url_img):
    '''
       Returns the breed of the dog in the image if present.
       If a human is present, predicts the most resembling dog breed
    '''
    response = requests.get(url_img)
    try:
        img = Image.open(BytesIO(response.content)).convert('RGB') 
        img = np.array(img) 
        img = img[:, :, ::-1].copy() 
    except:
        return {"message": "can't detected"}
    is_face_detected = face_detector(img)
    dog_breed = predict_breed(img)
    return {"dog_breed": dog_breed, "is_face_detected": is_face_detected}
