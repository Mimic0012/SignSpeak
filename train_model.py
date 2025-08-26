import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


X, y = [], []
for file in os.listdir(DATA_DIR):
    label = file.replace(".npy", "")
    data = np.load(os.path.join(DATA_DIR, file))
    X.extend(data)
    y.extend([label] * len(data))

X = np.array(X)
y = np.array(y)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)


with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)


model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_categorical, epochs=30, batch_size=32, validation_split=0.2)

model.save(os.path.join(MODEL_DIR, "gesture_model.h5"))
print("Model trained and saved.")

