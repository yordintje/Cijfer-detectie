import os
import cv2 # Load and process images
import numpy as np # Numpy Array's
import matplotlib.pyplot as plt # Visualize of digits
import tensorflow as tf # Using for the Machine Learning part

mnist = tf.keras.datasets.mnist # Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data() # Split the data into training and testing sets

# Normalize the data (0-1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# # Create the model
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # Flatten the input
# model.add(tf.keras.layers.Dense(128, activation='relu')) # Hidden layer with 128 neurons
# model.add(tf.keras.layers.Dense(128, activation='relu')) # Hidden layer with 128 neurons
# model.add(tf.keras.layers.Dense(10, activation='softmax')) # Output layer with 10 neurons | probability distribution

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Compile the model

# model.fit(x_train, y_train, epochs=3) # Train the model

# model.save('geschreven.keras') # Save the model

model = tf.keras.models.load_model('geschreven.keras') # Load the model

loss, accuracy = model.evaluate(x_test, y_test) # Evaluate the model

print(loss, accuracy) # Print the loss and accuracy


