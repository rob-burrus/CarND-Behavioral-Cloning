import numpy as np
import tensorflow as tf
import pandas
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import cv2
from utilities import get_left_right, get_images, generate_validation, generate, combine_data
#tf.python.control_flow_ops = tf

def generate(x, y, batch_size):
    size = len(x)
    while True:
        rng = np.random.choice(size, batch_size)
        # randomly flips some images
        x_batch, y_batch = get_images(x[rng], y[rng])
        yield x_batch, y_batch

def load_data():
    # Load the Data
    driving_log_data = pandas.read_csv('./1_lap/driving_log.csv')
    driving_log_data.head()
    center_images = driving_log_data['center'].as_matrix()
    left_images = driving_log_data['left'].as_matrix()
    right_images = driving_log_data['right'].as_matrix()
    center_angles = driving_log_data['steering'].as_matrix()
    c_images = np.array(center_images)
    l_images = np.array(left_images)
    r_images = np.array(right_images)
    c_angles = np.array(center_angles)

    #get left / right images / angles. Does not include center images / center angles
    lr_images, lr_angles = get_left_right(c_images, l_images, r_images, c_angles)

    #split into training and validation - only use center images in validation data
    c_images, X_val, c_angles, y_val = train_test_split(c_images, c_angles, test_size=0.30)

    #combine left/right data with center training data
    X_train, y_train = combine_data(lr_images, lr_angles, c_images, c_angles)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    return X_train, X_val, y_train, y_val


def get_model():
    # Define Keras Sequential Model
    model = Sequential()
    #normalize the color values
    model.add(Lambda(lambda x: x / 255 - .5, input_shape=(160, 320, 3)))
    #crop the images to ignore unecessary details like sky and hood of the car
    model.add(Cropping2D(cropping=((75,25), (0,0))))
    #nvidia model
    model.add(Convolution2D(24, 8, 8, subsample=(3,3), activation='relu'))
    model.add(Convolution2D(36, 8, 8, subsample=(3,3), activation='relu'))
    model.add(Convolution2D(48, 3, 3, subsample=(1,1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(75))
    model.add(Dropout(0.5))
    model.add(Dense(30))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
def get_commaai():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((75,25), (0,0))))
    model.add(Convolution2D(16, 8, 8, subsample=(4,4), activation='relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model



def main():
    model = get_commaai()
    print(model.summary())
    return
    # Train the Model
    model.compile('adam', 'mse')


    X_train, X_val, y_train, y_val = load_data()

    history = model.fit_generator(generate(X_train, y_train, 128),
                                  nb_epoch=4,
                                  samples_per_epoch=(len(X_train)//128)*128,
                                  validation_data=(generate_validation(X_val, y_val, 128)),
                                  nb_val_samples=(len(X_val)//128)*128,
                                  verbose=1)

    # Save the Model
    model.save('model.h5')

if __name__ == "__main__":
    main()
