
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape, num_classes, augmentation_layer):
    model = models.Sequential(name="CIFAR10_CNN")
    # Add augmentation layer first
    model.add(augmentation_layer)
    # ... (rest of the layers from build_model in train.py) ...
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model
