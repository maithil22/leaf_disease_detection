import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import cv2


def instantiate_model():
    # Define the model
    model = models.Sequential()

    # First Convolution Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolution Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolution Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fourth Convolution Block
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fifth Convolution Block
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Dropout
    model.add(layers.Dropout(0.5))

    # Flatten layer
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(1500, activation='relu'))

    # Dropout
    model.add(layers.Dropout(0.5))

    # Output layer (for 38 classes)
    model.add(layers.Dense(38, activation='softmax'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def instantiate_alexnet():
    model = Sequential()

    # Convolution Step 1 (Updated input shape to (128, 128, 3))
    model.add(Convolution2D(96, 11, strides=(4, 4), padding='same', input_shape=(128, 128, 3), activation='relu'))

    # Max Pooling Step 1
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # Convolution Step 2
    model.add(Convolution2D(256, 11, strides=(1, 1), padding='same', activation='relu'))

    # Max Pooling Step 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # Convolution Step 3
    model.add(Convolution2D(384, 3, strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # Convolution Step 4
    model.add(Convolution2D(384, 3, strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # Convolution Step 5
    model.add(Convolution2D(256, 3, strides=(1,1), padding='same', activation='relu'))

    # Max Pooling Step 3
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # Flattening Step
    model.add(Flatten())

    # Full Connection Step
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(units=1000, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(units=38, activation='softmax'))

    # Define an exponential decay learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,  # You can adjust this
        decay_rate=0.96)

    # Using SGD optimizer with learning rate schedule and momentum
    optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)

    # Compiling the model
    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    for i, layer in enumerate(model.layers[:20]):
        print(i, layer.name)
        layer.trainable = False
    
    return model

def load_pretained_model(mechanism):
    match mechanism:
        case "raw":
            path = os.getcwd() + "/leaf_disease_detection/saved_models/raw_model.h5"
        case "segmented":
            path = os.getcwd() + "/leaf_disease_detection/saved_models/segmented_model.keras"
        case "clahe":
            path = os.getcwd() + "/leaf_disease_detection/saved_models/clahe_model.keras"
        case "alexnet_clahe":
            path = os.getcwd() + "/leaf_disease_detection/saved_models/alexnet.h5"
    model = load_model(path)
    return model

def segment_image(image):
    # segmented_image = seg_with_sam(mask_generator, resized_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # find the green color
    mask_green = cv2.inRange(hsv, (36,0,0), (86,255,255))
    # find the brown color
    mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    # find the yellow color in the leaf
    mask_yellow = cv2.inRange(hsv, (21, 39, 64), (40, 255, 255))

    # find any of the three colors(green or brown or yellow) in the image
    mask = cv2.bitwise_or(mask_green, mask_brown)
    mask = cv2.bitwise_or(mask, mask_yellow)
    # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((8,8),dtype=np.uint8))
    # apply mask to original image
    result = cv2.bitwise_and(image, image,mask=mask)
    return result

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Convert image to LAB, apply CLAHE on L channel, and segment using mask in HSV
    # Step 1: Read the image in RGB format
    if image is not None:

        # Step 2: Convert the image from RGB to LAB spectrum
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Step 3: Split LAB image into L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Step 4: Apply CLAHE on the L (luminance) channel
        clahe_l = clahe.apply(l_channel)

        # Step 5: Merge the enhanced L channel back with the original A and B channels
        lab_clahe_image = cv2.merge((clahe_l, a_channel, b_channel))

        # Step 6: Convert the LAB image back to RGB
        enhanced_rgb_image = cv2.cvtColor(lab_clahe_image, cv2.COLOR_LAB2RGB)

        return enhanced_rgb_image
    else:
        raise Exception("Image path incorrect")