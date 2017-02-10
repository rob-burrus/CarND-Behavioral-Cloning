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
from utilities import get_images



# Load the Data
driving_log_data = pandas.read_csv('./driving_log.csv')
driving_log_data.head()
X_train = driving_log_data['center'].as_matrix()
y_train = driving_log_data['steering'].as_matrix()
X_train = np.array(X_train)
y_train = np.array(y_train)
#print(X_train[0:5])
get_images(X_train[0:5], y_train[0:5])

print('Done')
