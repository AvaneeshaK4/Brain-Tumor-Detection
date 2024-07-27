# Brain Tumor Detection Using Convolutional Neural Networks (CNN)

## Abstract

Brain tumors are a critical medical condition that requires accurate and early detection for effective treatment. Traditional diagnostic methods may be time-consuming and require expert interpretation. This project leverages machine learning and computer vision techniques to detect brain tumors from MRI scans. Using Convolutional Neural Networks (CNN), this study aims to classify MRI images into two categories: "Tumor" and "No Tumor". The model's performance is evaluated based on accuracy, providing a preliminary diagnostic tool that supports healthcare professionals in early tumor detection. The results demonstrate that the CNN model effectively captures complex patterns in MRI images, offering a robust solution for brain tumor detection.

## Project Overview

This project aims to detect brain tumors from MRI images using Convolutional Neural Networks (CNN). By analyzing MRI scans, the model assists in early diagnosis and provides valuable support to healthcare professionals in managing brain tumor cases more effectively.

## Model Architecture

The CNN model includes:
- **Convolutional Layers**: Three convolutional layers with increasing filter sizes to capture features at different levels.
- **Activation Function**: ReLU (Rectified Linear Unit) for introducing non-linearity.
- **Pooling Layers**: Max pooling layers to reduce spatial dimensions and retain important features.
- **Flattening**: Converts 2D feature maps into a 1D vector.
- **Fully Connected Layers**: Dense layers for decision making and dropout for regularization.
- **Output Layer**: A sigmoid activation function to provide binary classification (Tumor/No Tumor).

## Dataset

The dataset used in this project contains MRI images of brain tumors. The images are categorized into two classes:
- **Tumor**: Images where a brain tumor is present.
- **No Tumor**: Images without any tumors.

The dataset can be accessed from: [Brain Tumor Dataset]([https://www.kaggle.com/datasets/johndasilva/brain-tumor-dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection))

## Performance Metrics

- **Accuracy**: The primary metric used to evaluate the model's performance.

## Results

- **Accuracy**: The CNN model achieved an accuracy of 76.9% in detecting brain tumors from MRI scans.

## Key Findings

- The CNN model effectively captures complex patterns in MRI images, demonstrating strong performance in brain tumor detection.
- The model's accuracy highlights its potential as a reliable diagnostic tool for early tumor detection.
- Future work should focus on enhancing the model's generalizability and exploring advanced techniques to further improve accuracy and clinical applicability.


