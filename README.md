
# Food Image Recognition and Calorie Estimation

## Overview

This project is focused on building a deep learning model that can recognize food items from images and estimate their calorie content. It leverages the [Food-101 dataset](https://www.kaggle.com/dansbecker/food-101) to train and test the model, providing an automated solution for food recognition and calorie tracking.

## Dataset

The [Food-101 dataset](https://www.kaggle.com/dansbecker/food-101) consists of 101 different food categories, with 1,000 images for each category. This comprehensive dataset allows the model to learn from diverse examples and generalize well to unseen food images.

## Model Architecture

The model is based on Convolutional Neural Networks (CNNs) using TensorFlow/Keras. It processes the input images through multiple layers of convolution, pooling, and fully connected layers, aiming to identify the food categories accurately and predict calorie content.

## Key Features

- **Input**: Images from the Food-101 dataset.
- **Output**: Predicted food category and estimated calorie content.
- **Tools**:
  - TensorFlow/Keras for the deep learning model.
  - Matplotlib for visualizations.
  - Kaggle’s Food-101 dataset for training and testing.

## Prerequisites

To run the project, you need to install the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy
- Pandas


## Installation

Follow these steps to set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adnaanzshah/PRODIGY_ML_05
   cd PRODIGY_ML_05
   ```

2. **Download the Food-101 dataset**:
   Download the dataset from [Kaggle](https://www.kaggle.com/dansbecker/food-101) and place it inside the project directory in a folder named `food-101`.

3. **Prepare the dataset**:
   Ensure that the dataset is organized into training and test sets. Preprocess the images if necessary (e.g., resizing, normalization).

The output will display the predicted food item and its approximate calorie content.

## Results

The model's performance is evaluated based on accuracy and loss on the test dataset. Although initial results show a low accuracy, with further tuning and optimization, the model’s performance is expected to improve.

Visualize the training progress and evaluate the accuracy through the generated plots.

## Next Steps

- **Data Augmentation**: Implement techniques such as rotation, flipping, and zooming to improve the model’s generalization.
- **Hyperparameter Tuning**: Adjust the learning rate, batch size, and other parameters to improve accuracy.
- **Advanced Architectures**: Experiment with deeper models like ResNet or EfficientNet to enhance performance.

