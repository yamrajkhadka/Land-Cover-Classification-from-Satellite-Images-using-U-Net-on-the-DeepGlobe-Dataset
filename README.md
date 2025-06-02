# Land Cover Classification using U-Net (500 Epochs)

## Objective

The goal of this project is to perform pixel-wise land cover classification on satellite images using a U-Net deep learning model. Each pixel in the image is categorized into classes such as urban area, agriculture, water, and other land types. This enables automated mapping of land use from satellite data for applications in agriculture, urban planning, and environmental monitoring.

## Summary

- Dataset: DeepGlobe Land Cover Classification  
- Model: U-Net (implemented in TensorFlow/Keras)  
- Training: 500 epochs on Google Colab  
- Output: Segmented visual maps of land cover categories  

## Problem Description

Satellite images are provided as RGB images, with corresponding ground-truth masks in RGB format where specific colors represent different land classes. Since machine learning models require numerical labels, the RGB masks are converted to integer class labels for training. After prediction, the class labels are converted back to RGB for visualization.

## Key Steps

### 1. Data Preprocessing

- Download dataset from Kaggle.  
- Convert RGB mask images to integer class labels using a predefined color-to-class mapping (`class_dict.csv`).  
- Normalize input images as needed for neural network input.  
- Split data into training and validation sets.  

### 2. Model Architecture

- U-Net architecture consisting of:  
  - Encoder: stacked Conv2D and MaxPooling layers for feature extraction.  
  - Decoder: Conv2DTranspose layers with skip connections to recover spatial information.  
  - Output layer applies Softmax activation to predict pixel-wise class probabilities.  
- Model handles dynamic input size and number of classes.  

### 3. Training

- Loss function: Categorical Crossentropy.  
- Optimizer: Adam.  
- Training for 500 epochs with monitoring of accuracy and loss.  
- Visualization of training curves to evaluate convergence.  

### 4. Postprocessing and Visualization

- Convert predicted class label masks back to RGB using the color mapping.  
- Visualize random validation images side-by-side with ground truth and predicted masks for qualitative assessment.  


## Files Included

- `500_epoch_unet_model_visualize3.py`: Main training, prediction, and visualization script.  
- `class_dict.csv`: Color-to-class label mapping file.  
 


## Author

Yamraj Khadka  
BTech Computer Engineering, Nepal
