# Cat vs. Dog Image Classification using VGG16 & Transfer Learning
For this project, I initially explored the Dogs vs. Cats dataset from [Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats), which contains 20,000 training images and 4,000 testing images. However, due to computational constraints, I decided to use a smaller dataset and apply data augmentation to improve performance. The goal was to build an efficient image classification model that distinguishes between cats and dogs with high accuracy.

### 1. Initial CNN Model (Baseline Approach)
I first implemented a Convolutional Neural Network (CNN) trained on the full dataset. However, the model faced two key challenges:

Slow training time due to the large dataset.
Overfitting, leading to poor generalization on unseen images.
While the CNN performed reasonably well, it required improvements to enhance accuracy and efficiency.

### 2. Data Augmentation for Improved Performance
To address overfitting and reduce training time, I applied data augmentation, which artificially increases the training dataset by modifying existing images.

What is Data Augmentation?
Data augmentation is a technique used to improve model generalization by applying transformations such as:

Rotation
Flipping (Horizontal & Vertical)
Zooming
Brightness Adjustments
Shifting & Cropping
Why Data Augmentation?
Instead of using the full dataset, I reduced the training set size and applied augmentation techniques. This helped the model generalize better, prevented overfitting, and reduced computational cost while maintaining high accuracy.

### 3. Transfer Learning with VGG16 + Data Augmentation
To further enhance performance, I implemented transfer learning using VGG16, a deep learning model pretrained on ImageNet.

### What is Transfer Learning?
Transfer learning is a technique where a pre-trained model (trained on a large dataset like ImageNet) is reused and fine-tuned for a different but related task. Instead of training a deep network from scratch, we leverage the existing knowledge of a pre-trained model, significantly improving accuracy while reducing training time.

### Why Transfer Learning?
Reduces training time by reusing pre-trained weights.
Extracts useful features from lower layers of the network.
Works effectively even with a smaller dataset.
### What is VGG16?
VGG16 (developed by Oxford’s Visual Geometry Group) is a deep CNN architecture with 16 layers, trained on millions of images. It is widely used for feature extraction in image classification tasks.

### How I Used VGG16 in My Model:
Used VGG16 as a base model (with pretrained ImageNet weights).
Removed the original fully connected layers and replaced them with custom dense layers for binary classification (cat vs. dog).
Fine-tuned deeper layers to improve feature learning.
Combined with data augmentation for better generalization.
This approach significantly improved efficiency and accuracy, allowing the model to perform well even with a reduced dataset. Further fine-tuning and hyperparameter tuning can further optimize the results.

# Deployment
To make the model accessible, I deployed it on Streamlit Cloud, allowing users to upload an image and receive real-time predictions.
[Live Demo](https://cat-vs-dog-image-classification-jfjxataftysf9pf8r97ojr.streamlit.app/)

### Tech Stack Used
Deep Learning: TensorFlow, Keras
Computer Vision: OpenCV, Image Processing
Data Handling: NumPy, Pandas
Model Deployment: Streamlit Cloud
This project demonstrates the effectiveness of transfer learning and data augmentation in building deep learning models with limited data. Let me know if you have any suggestions for further improvements.
