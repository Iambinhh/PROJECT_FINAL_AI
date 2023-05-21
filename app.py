import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn import preprocessing
import os
import h5py


st.title("CLASSIFICATION OF ANIMAL IMAGES")
st.text("This is a simple project which display the model classifications of the animals.  ")
st.text("Cause the limits of the data so I just use 10 classes for animals")
def main():
    file_uploaded=st.file_uploader("Choose your file",type=['jpg','png','jpeg'])
    if file_uploaded is not None:
        image=Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result=predict_class(image)
        st.write(result)
        st.pyplot(figure)
        
        
def predict_class(image):
    #global score_result
    classifer_model = tf.keras.models.load_model(r'animal_classify.hdf5')
    shape = ((128,128,3))
    model = tf.keras.Sequential([hub.KerasLayer(classifer_model,input_shape=shape)])
    test_image = image.resize((128,128))
    test_image = img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    class_names = ['BEAR', 'CAT', 'DOG', 'ELEPHANT', 'PARROT', 'RAT', 'RHINO', 'SNAKE', 'TIGER', 'WHALE']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result  =  "The image uploaded is: {}".format(image_class)
    #score_result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(scores)], 100 * np.max(scores))
    
    return result
if __name__ == '__main__':
    main()
    