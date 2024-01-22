# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:19:09 2023

@author: akshi
"""

import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load your own image
input_image_path = "C:/Users/akshi/Downloads/(10_10) Sponsorship_banner.jpg"
original_image = load_img(input_image_path, target_size=(128, 128))
original_image_array = img_to_array(original_image)
original_image_array = original_image_array / 255.0  # Normalize pixel values

# Add noise to the image (for training purposes)
noise_factor = 0.5
noisy_image_array = original_image_array + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=original_image_array.shape)
noisy_image_array = np.clip(noisy_image_array, 0.0, 1.0)

# Define the compression model
input_img = Input(shape=(128, 128, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (use noisy images as both input and target for simplicity)
autoencoder.fit(noisy_image_array.reshape(-1, 128, 128, 3), original_image_array.reshape(-1, 128, 128, 3),
                epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Use the trained model for image compression
compressed_image_array = autoencoder.predict(noisy_image_array.reshape(-1, 128, 128, 3))

# Display original, noisy, and compressed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(original_image_array)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisy_image_array.squeeze())
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Compressed Image')
plt.imshow(compressed_image_array.squeeze())
plt.axis('off')

plt.show()
