import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("gesture_model.h5")

labels = ["fist", "palm", "thumbs_up"]

img = cv2.imread("hand.jpg")

img = cv2.resize(img, (64, 64))

img = img / 255.0

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

predicted_class = np.argmax(prediction)

print("Prediction:", labels[predicted_class])