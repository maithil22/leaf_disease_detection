# Plant Leaf Disease detection using Machine Learning

This project uses various approaches for classifying plant diseases based on leaf images. A custom model is trained from scratch for 3 approaches:
-  Raw image data
- Segmented image data
- Segmented image data with CLAHE

## Approaches

### Raw data training
For this approcah, raw dataset used for training.

### Segmented data training
For this approach, each leaf image is segmented to separate it from the background. This is performed using a mask to separate the leaf pixes from the background

### Segmented + CLAHE data training
CLAHE stands for Contrast Limited Adaptive Histogram Equalization. It is a method used to improve image contrast, particularly in cases where lighting conditions are uneven or where the image has poor contrast.

- Step 1: Conversion of RGB image to LAB color space

![lab_breakdown](https://github.com/maithil22/leaf_disease_detection/blob/main/images/lab_breakdown.png)

- Step 2: Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L dimension.

![clahe_l_channel](https://github.com/maithil22/leaf_disease_detection/blob/main/images/clahe_l_channel.png)

- Step 3: Reconstruct a better image by merging the LAB spectrum again, now with the new L layer.

![reconstruct_leaf](https://github.com/maithil22/leaf_disease_detection/blob/main/images/reconstruct_leaf.png)

## Dataset
We have used the augmented version of [Plant Village](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) dataset comprising of 87.9k images. 
This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.

## Model Summary
 The model is custom trained CNN with the following topology:
 ![model_summary](https://github.com/maithil22/leaf_disease_detection/blob/main/images/model_summary.png)

## Results