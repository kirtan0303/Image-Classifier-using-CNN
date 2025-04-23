# CNN Image Classifier for CIFAR-10

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset. It includes data preprocessing, data augmentation, and dropout for regularization.

## Table of Contents

- [Description](#description)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

## Description

The goal of this project is to build and train a robust image classification model capable of distinguishing between the 10 different classes in the CIFAR-10 dataset. Key techniques employed include:

-   **Convolutional Neural Networks (CNNs):** Suitable for learning hierarchical features from image data.
-   **Data Normalization:** Scaling pixel values to the [0, 1] range for stable training.
-   **Data Augmentation:** Applying random transformations (flips, rotations, zooms) to the training data *on-the-fly* to increase dataset diversity and reduce overfitting. Implemented using Keras preprocessing layers.
-   **Dropout:** A regularization technique to prevent overfitting by randomly setting a fraction of input units to 0 during training.
-   **Batch Normalization:** Helps stabilize and speed up training.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 colour images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The 10 classes are:
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

The dataset is automatically downloaded via `tf.keras.datasets.cifar10`. More info: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Tech Stack

-   Python (3.8+)
-   TensorFlow (>=2.8.0)
-   NumPy
-   Matplotlib (for plotting training history)

## Model Architecture

The CNN architecture consists of several convolutional blocks followed by a dense classifier head. Each convolutional block typically includes:

1.  `Conv2D` layer(s) with ReLU activation.
2.  `BatchNormalization` layer.
3.  `MaxPooling2D` layer for downsampling.
4.  `Dropout` layer for regularization.

The final layers include:
1.  `Flatten` layer to convert feature maps to a vector.
2.  `Dense` layer(s) with ReLU activation.
3.  `BatchNormalization` and `Dropout`.
4.  Output `Dense` layer with `softmax` activation for multi-class classification (10 units).

Data augmentation (RandomFlip, RandomRotation, RandomZoom) is applied as the first layer of the model.

## Results

After training for `[Number]` epochs (potentially stopped early by EarlyStopping), the model achieved the following performance on the *test set*:

-   **Test Loss:** `[Test Loss Value]`
-   **Test Accuracy:** `[XX.XX]%`  **<-- IMPORTANT: Replace this with your actual result!**

The training history (accuracy and loss curves) is saved as `training_history.png`:

![Training History](training_history.png)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/cnn-cifar10-classifier.git
    cd cnn-cifar10-classifier![pngwing com](https://github.com/user-attachments/assets/828a334f-c8cb-4ea6-bffd-ea1f5726b43e)

    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: TensorFlow installation might require specific steps depending on your system, especially if you need GPU support. Refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install).)*

## Usage

To train the model and evaluate it on the test set, run the main script:

```bash
python train.py

## Future Improvements

Hyperparameter Tuning: Experiment with learning rate, batch size, dropout rates, number of filters, layer depth using tools like KerasTuner or Optuna.
Advanced Architectures: Implement more complex CNN architectures like ResNet, VGG, or EfficientNet (potentially using transfer learning).
More Augmentation: Explore more advanced data augmentation techniques.
Learning Rate Scheduling: Implement learning rate decay or cyclic learning rates.
PyTorch Implementation: Create an alternative implementation using PyTorch for comparison.
