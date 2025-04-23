import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 50 # Adjust as needed, start with fewer, maybe 20-50
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
MODEL_SAVE_PATH = 'cifar10_cnn_best.h5' # Keras H5 format

# --- 1. Load and Preprocess Data ---
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Initial shapes: x_train={x_train.shape}, y_train={y_train.shape}, x_test={x_test.shape}, y_test={y_test.shape}")

# Normalization: Scale pixel values from 0-255 to 0-1
print("Normalizing data...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-Hot Encode Labels
print("One-hot encoding labels...")
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# --- 2. Data Augmentation ---
# Using Keras preprocessing layers within the model is often efficient
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=INPUT_SHAPE),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        # layers.RandomContrast(0.1), # Add more augmentations if needed
    ],
    name="data_augmentation",
)

# --- 3. Build the CNN Model ---
print("Building the CNN model...")
def build_model(input_shape, num_classes, augmentation_layer):
    model = models.Sequential(name="CIFAR10_CNN")

    # Add augmentation layer first
    model.add(augmentation_layer)

    # Convolutional Base
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization()) # Helps stabilize training
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25)) # Dropout for regularization

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Classifier Head
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax')) # Softmax for multi-class classification

    return model

model = build_model(INPUT_SHAPE, NUM_CLASSES, data_augmentation)

# --- 4. Compile the Model ---
print("Compiling the model...")
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), # Adam is a common choice
              loss=losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.build((None,) + INPUT_SHAPE) # Build model to print summary
model.summary()

# --- 5. Callbacks ---
# Save the best model based on validation accuracy
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

# Stop training early if validation loss doesn't improve
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10, # Number of epochs with no improvement
                               verbose=1,
                               mode='min',
                               restore_best_weights=True) # Restores best weights found

callbacks_list = [checkpoint, early_stopping]

# --- 6. Train the Model ---
print("\n--- Starting Training ---")
start_time = datetime.datetime.now()

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.1, # Use 10% of training data for validation
                    callbacks=callbacks_list,
                    verbose=1)

end_time = datetime.datetime.now()
training_time = end_time - start_time
print(f"--- Training Finished in {training_time} ---")

# --- 7. Evaluate the Model ---
print("\n--- Evaluating Model on Test Data ---")
# Load the best saved model weights if early stopping restored them,
# otherwise the last epoch's weights are used unless restore_best_weights=True
# To be certain, we can load the best model explicitly:
# print(f"Loading best model from {MODEL_SAVE_PATH}")
# best_model = models.load_model(MODEL_SAVE_PATH) # Requires h5py
# score = best_model.evaluate(x_test, y_test, verbose=0)
# Or evaluate the model state after training (which might be the best if restore_best_weights=True)
score = model.evaluate(x_test, y_test, verbose=1)

print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]*100:.2f}%') # <-- Replace [XX]% in README

# --- 8. Plot Training History (Optional) ---
print("\n--- Plotting Training History ---")
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history.png') # Save the plot
    print("Training history plot saved as training_history.png")
    # plt.show() # Uncomment to display plot interactively

plot_history(history)

print("\n--- Project Execution Finished ---")
