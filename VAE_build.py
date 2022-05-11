from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

ComputeLB = False

import os, gc, zipfile
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

BATCH_SIZE = 256; EPOCHS = 10
train_datagen = ImageDataGenerator(rescale=1./255)
train_batches = train_datagen.flow_from_directory('./tmp/',
        target_size=(64,64), shuffle=True, class_mode='input', batch_size=BATCH_SIZE)

# ENCODER
input_img = Input(shape=(64, 64, 3))
x = Conv2D(48, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Conv2D(32, (1, 1), activation='relu', padding='same')(x)

# LATENT SPACE
latentSize = (8,8,32)

# DECODER
direct_input = Input(shape=latentSize)
x = Conv2D(192, (1, 1), activation='relu', padding='same')(direct_input)
x = UpSampling2D((2, 2))(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# COMPILE
encoder = Model(input_img, encoded)
decoder = Model(direct_input, decoded)
autoencoder = Model(input_img, decoder(encoded))


autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

# Train
history = autoencoder.fit_generator(train_batches,
        steps_per_epoch = train_batches.samples // BATCH_SIZE,
        epochs = EPOCHS, verbose = 2)




# View Reconstruction
images = next(iter(train_batches))[0]
for i in range(5):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)

        # ORIGINAL IMAGE
        orig = images[i, :, :, :].reshape((-1, 64, 64, 3))
        img = Image.fromarray((255 * orig).astype('uint8').reshape((64, 64, 3)))
        plt.title('Original')
        plt.imshow(img)

        # LATENT IMAGE
        latent_img = encoder.predict(orig)
        mx = np.max(latent_img[0])
        mn = np.min(latent_img[0])
        latent_flat = ((latent_img[0] - mn) * 255 / (mx - mn)).flatten(order='F')
        img = Image.fromarray(latent_flat[:2025].astype('uint8').reshape((45, 45)), mode='L')
        plt.subplot(1, 3, 2)
        plt.title('Latent')
        plt.xlim((-10, 55))
        plt.ylim((-10, 55))
        plt.axis('off')
        plt.imshow(img)

        # RECONSTRUCTED IMAGE
        decoded_imgs = decoder.predict(latent_img[0].reshape((-1, latentSize[0], latentSize[1], latentSize[2])))
        img = Image.fromarray((255 * decoded_imgs[0]).astype('uint8').reshape((64, 64, 3)))
        plt.subplot(1, 3, 3)
        plt.title('Reconstructed')
        plt.imshow(img)

        plt.show()