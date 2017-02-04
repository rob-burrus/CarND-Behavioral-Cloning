import numpy as np
import tensorflow as tf
import pandas
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import cv2
from PIL import Image

tf.python.control_flow_ops = tf

def getImages(image_paths):
    features = np.empty([len(image_paths), 40, 80, 3])

    for i,x in enumerate(image_paths):
        #print(x)
        image = cv2.imread(x)
        features[i] = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
        #histogram equalization. Increases image contrast
        #image = cv2.equalizeHist(image)
        #feature = np.array(image, dtype=np.float32).flatten()
        #features.append(image)
    return features

def grayscale(imgs):
    return np.mean(imgs, axis=3, keepdims=True)

def normalize(imgs):
    return imgs / (255.0 / 2) - 1

def normalize_grayscale(image_data):
    a = -1.0
    b = 1.0
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


#
#
# Load the Data
#
#
driving_log_data = pandas.read_csv('./driving_log.csv')
driving_log_data.head()
X_train_image_paths = driving_log_data['center'].as_matrix()
y_train = driving_log_data['steering'].as_matrix()


#shuffle the data
X_train, y_train = shuffle(X_train_image_paths, y_train)

#split into training and validation
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20)

X_train = np.array(X_train)
y_train = np.array(y_train)
#X_val = np.array(X_val)
#y_val = np.array(y_val)


X_train = getImages(X_train)
X_train = grascale(X_train)
X_train = normalize(X_train)
#
#
# Define Keras Sequential Model
#
#
model = Sequential([
    Convolution2D(64, 3, 3, input_shape=(40, 80, 1)),
    Activation('relu'),
    Convolution2D(128, 3, 3),
    Activation('relu'),
    Convolution2D(256, 3, 3),
    Activation('relu'),
    Flatten(),
    Dense(1164, activation='relu'),
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)
])

#
#
# Train the Model
#
#
model.compile('adam', 'mse', ['accuracy'])

history = model.fit(X_train, y_train,
                              10,
                              validation_split=0.2,
                              verbose=1, max_q_size=1,
                              pickle_safe=False)


#
#
# Save the Model
#
#
model.save_weights('model.h5')
with open('model.json', 'w') as f:
    f.write(model.to_json())

print('Done')
