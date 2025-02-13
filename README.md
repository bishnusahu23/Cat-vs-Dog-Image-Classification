# Cat vs Dog Image Classification
For this project, I used the Dogs vs. Cats dataset from Kaggle (https://www.kaggle.com/datasets/salader/dogs-vs-cats), which contains 20,000 training images and 4,000 testing images.

1. Initial CNN Model
I started with a Convolutional Neural Network (CNN) trained on the full dataset. However, training was slow, and the model struggled to achieve high accuracy due to computational constraints and overfitting.

2. Data Augmentation for Improved Performance
To address the issues, I reduced the dataset to 1,000 training images and 400 testing images, applying data augmentation techniques (rotation, flipping, zooming, etc.) to artificially expand the training set. This helped improve generalization and significantly reduced training time while maintaining decent accuracy.

3. Transfer Learning with VGG16
For further improvements, I implemented transfer learning using VGG16 as the base model. This approach leveraged pre-trained weights from ImageNet, drastically boosting accuracy compared to the previous models. While the results were much better, there is still room for optimization through fine-tuning and hyperparameter adjustments.
