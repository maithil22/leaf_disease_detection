import argparse

import numpy as np
import cv2

from utils import apply_clahe, load_pretained_model, segment_image
import tensorflow as tf


# Define function to predict disease
def predict_disease(model, preprocessed_image):
    preprocessed_image = tf.expand_dims(preprocessed_image, axis=0) 
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

def preprocess_image(image_path, model_name):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128,128,))
    match model_name:
        case "raw":
            return image
        case "segmented":
            segmented_image = segment_image(image)
            return segmented_image
        case "clahe":
            clahe_segmented_image = apply_clahe(segment_image(image))
            return clahe_segmented_image
        case "alexnet_clahe":
            clahe_segmented_image = apply_clahe(segment_image(image))
            return clahe_segmented_image

# Main script
def main(image_path, model_name):
    try:
        # Load the specified model
        model = load_pretained_model(model_name)
        
        # Preprocess the input image
        preprocessed_image = preprocess_image(image_path, model_name)
        
        # Predict the disease
        disease_name = predict_disease(model, preprocessed_image)
        
        print(f"Predicted Disease: {disease_name}")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plant Disease Prediction using Pretrained Models")
    parser.add_argument("image_path", type=str, help="Path to the plant leaf image")
    parser.add_argument("model_name", type=str, choices=["raw", "segmented", "clahe", "alexnet_clahe"],
                        help="Approach of the pretrained model file to use")
    
    args = parser.parse_args()
    
    main(args.image_path, args.model_name)