import os
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

my_model = keras.models.load_model('cats_vs_dogs_vgg16.keras')

img, x = get_image('TestImages/lobo.jpeg')
probability = my_model.predict([x])
print('probability: ', probability)
print('shape: ', probability.shape)