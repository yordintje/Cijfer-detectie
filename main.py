import os
import cv2 # Load and process images
import numpy as np # Numpy Array's
import matplotlib.pyplot as plt # Visualize of digits
import tensorflow as tf # Using for the Machine Learning part

model = tf.keras.models.load_model('cijfermodel.keras') # Load the model

image_number = 1
while os.path.isfile(f"getallen/getal{image_number}.png"):
    try:
        img = cv2.imread(f"getallen/getal{image_number}.png")[:,:,0] # Load the image
        img = np.invert(np.array([img])) # Invert the image
        prediction = model.predict(img) # Predict the image
        print(f"Het getal is waarschijnlijk een {np.argmax(prediction)}") # Print the prediction
        plt.imshow(img[0], cmap=plt.cm.binary) # Show the image
        plt.show()
    finally:
        image_number += 1