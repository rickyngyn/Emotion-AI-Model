import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('../models/emotion_cnn.h5')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face region
        face = gray[y:y+h, x:x+w]

        # Resize to 48x48 and normalize
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized.astype('float32') / 255.0
        face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(face_reshaped, verbose=0)
        emotion_label = emotions[np.argmax(prediction)]

        # Draw box + emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()