# data_utils.py
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_and_preprocess_cifar10(num_classes=10):
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(f"Initial shapes: x_train={x_train.shape}, y_train={y_train.shape}, x_test={x_test.shape}, y_test={y_test.shape}")

    # Normalization
    print("Normalizing data...")
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-Hot Encode Labels
    print("One-hot encoding labels...")
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def create_augmentation_layer(input_shape):
     # Using Keras preprocessing layers within the model is often efficient
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", input_shape=input_shape),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )
    return data_augmentation
