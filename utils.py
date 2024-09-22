import tensorflow as tf
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

def load_model(mechanism):
    match mechanism:
        case "raw":
            path = ""
        case "segmented":
            path = ""
        case "clahe":
            path = ""
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

def apply_clahe():
    pass