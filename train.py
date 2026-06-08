import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

dataset_path = "dataset"

data = []
labels = []

IMG_SIZE = 64

label_map = {
    "fist-2": 0,
    "palm-1": 1,
    "thumbs-up-3": 2
}

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        data.append(img)

        labels.append(label_map[folder])

data = np.array(data)

data = data / 255.0

labels = np.array(labels)

labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(64,64,3)
))

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_test, y_test)
)

model.save("gesture_model.h5")

print("Model Saved Successfully")
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)