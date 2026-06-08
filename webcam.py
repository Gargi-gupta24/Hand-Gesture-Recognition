import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("gesture_model.h5")

labels = ["fist", "palm", "thumbs_up"]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    img = cv2.resize(frame, (64, 64))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    gesture = labels[predicted_class]

    cv2.putText(
        frame,
        gesture,
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()

cv2.destroyAllWindows()