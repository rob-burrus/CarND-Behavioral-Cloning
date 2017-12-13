import cv2
import numpy as np
import tensorflow as tf
import pandas
#from scipy.misc import imread, imresize, imshow

def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    #cv2.imshow('img', image1)
    #cv2.waitKey(5000)
    return image1

def random_flip(image, steering):
    if np.random.choice(2):
        image = cv2.flip(image, 1)
        steering = steering*-1
    else:
        steering = steering
    return image, steering

def get_images(image_paths, steering):
    images = np.empty([128, 160, 320, 3])
    for i,path in enumerate(image_paths):
        image = cv2.imread(path)
        image = random_brightness(image)
        images[i], steering[i] = random_flip(image, steering[i])
        #cv2.imshow('img', image)
        #cv2.waitKey(5000)
    return images, steering

def get_images_only(image_paths, steering):
    images = np.empty([128, 160, 320, 3])
    for i,path in enumerate(image_paths):
        images[i] = cv2.imread(path)
    return images, steering

def generate(x, y, batch_size):
    size = len(x)
    while True:
        rng = np.random.choice(size, batch_size)
        # randomly flips some images
        x_batch, y_batch = get_images(x[rng], y[rng])
        yield x_batch, y_batch

def generate_validation(x, y, batch_size):
    size = len(x)
    while True:
        rng = np.random.choice(size, batch_size)
        x_batch, y_batch = get_images_only(x[rng], y[rng])
        yield x_batch, y_batch

def get_left_right(center_images, left_images, right_images, center_angles):
    lr_paths = []
    lr_angles = []
    correction = 0.225
    for i, angle in enumerate(center_angles):
        #img_paths.append(center_images[i])
        lr_paths.append(left_images[i])
        lr_paths.append(right_images[i])
        a = float(angle)
        #all_angles.append(a)
        lr_angles.append(a+correction)
        lr_angles.append(a-correction)

    return lr_paths, lr_angles

def combine_data(lr_images, lr_angles, center_images, center_angles):
    all_paths = []
    all_angles = []
    for img, angle in zip(center_images, center_angles):
        all_paths.append(img)
        all_angles.append(angle)
    for img, angle in zip(lr_images, lr_angles):
        all_paths.append(img)
        all_angles.append(angle)

    return all_paths, all_angles

def augment_data(center_images, left_images, right_images, center_angles):
    imgs, steering = add_left_right(center_images, left_images, right_images, center_angles)
    return imgs, steering
