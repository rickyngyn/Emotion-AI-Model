import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Load CSV
df = pd.read_csv('data/fer2013.csv')

#Parse and normalize image data
pixels = np.vstack(df['pixels'].apply(lambda x: np.fromstring(x, sep=' ', dtype=np.float32)))
X = pixels.reshape(-1, 48, 48, 1) / 255.0
y = to_categorical(df['emotion'], num_classes=7)

# Split into training, validation, and test sets
train_mask = df['Usage'] == 'Training' # Makes 'training' columns equal true (avoid testing)
X_train_full, y_train_full = X[train_mask], y[train_mask] # Assigne variables to training data
X_test, y_test = X[~train_mask], y[~train_mask] #assigning variables to testing data

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

#Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64
)

# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# Save model
model.save('models/emotion_cnn.h5')
print("🎉 Model saved to 'models/emotion_cnn.h5'")

