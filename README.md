# Plant Leaf Disease detection using Machine Learning

This project uses various approaches for classifying plant diseases based on leaf images. A custom model is trained from scratch for 3 approaches:
- Raw image data
- Segmented image data
- Segmented image data with CLAHE

As benchmark, we have also compared our custom model with AlexNet.

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

## Run the model
```
## Run the pre-tranined raw data model
$ python plant_disease_evaluator.py /path/to/leaf_image.jpg raw

## Run the pre-tranined segmented data model
$ python plant_disease_evaluator.py /path/to/leaf_image.jpg segmented

## Run the pre-tranined CLAHE + segmented data model
$ python plant_disease_evaluator.py /path/to/leaf_image.jpg clahe

## Run the pre-tranined Alexnet CLAHE + segmented data model
$ python plant_disease_evaluator.py /path/to/leaf_image.jpg alexnet_clahe
```

## Results
After training the mode for 17,000 steps with each approach. We found minor improvements in accuracy in model by CLAHE, followed by segmented images over raw data.

### Loss charts for custom model
![loss_chart](https://github.com/maithil22/leaf_disease_detection/blob/main/images/loss_chart.png)

### Accuracy charts for custom model
![accuracy_chart](https://github.com/maithil22/leaf_disease_detection/blob/main/images/accuracy_chart.png)

### Loss charts for AlexNet with CLAHE
![loss_chart_alexnet](https://github.com/maithil22/leaf_disease_detection/blob/main/images/loss_chart_alexnet.png)

### Accuracy charts for AlexNet with CLAHE
![accuracy_chart_alexnet](https://github.com/maithil22/leaf_disease_detection/blob/main/images/accuracy_chart_alexnet.png)


Final accuracy for each model
- Raw: 96.10% (Validation accuracy: 94.91%)
- Segmented: 96.94% (Validation accuracy: 95.23%)
- Clahe: 97.28% (Validation accuracy: 96.83%)

AlexNet model with Clahe: 98.47% (Validation accuracy: 97.86%)
