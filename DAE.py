#from PIL import Image, ImageFilter
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

# setup
def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    array = np.array(array, dtype=np.float32) / 255.0
    array = np.reshape(array, (len(array), 512, 512, 1))
    return array


def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """

    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)


def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(512, 512))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(512, 512))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

# 이미지 개수
img_count = 220

X = []
y = []
# convert color image to 2D array (grayscale) & rescale
for i in range(img_count):
    data = cv2.imread('./dataset/metal_nut/'+ str(i).zfill(3) + '.jpg', 0)
    label = 0 # label/class of the image
    X.append(data)
    y.append(label)

# split for training & testing
# Since we only need images from the dataset to encode and decode,
# we won't use the labels.
# (train_data, _), (test_data, _) = mnist.load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Normalize and reshape the data
train_data = preprocess(X_train)
test_data = preprocess(X_test)

# Create a copy of the data with added noise
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

# Display the train data and a version of it with added noise
# display(train_data, noisy_train_data)

input = layers.Input(shape=(512, 512, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)


# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
# loss : binary_crossentropy or mse or mae
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

# Denoise autoencoder
autoencoder.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=100,
    batch_size=10,
    shuffle=True,
    validation_data=(noisy_test_data, test_data),
)

predictions = autoencoder.predict(noisy_test_data)
display(test_data, predictions)
display(noisy_test_data, predictions)

autoencoder.save('metal_nut_binary_100_b10.h5')
