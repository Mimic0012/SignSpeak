import cv2
import mediapipe as mp
import numpy as np
import os

LABELS = ["Hello", "How", "Are", "You"]
SAVE_DIR = "data"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

os.makedirs(SAVE_DIR, exist_ok=True)

def capture(label):
    cap = cv2.VideoCapture(0)
    data = []

    print(f"Collecting data for: {label}. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                data.append(landmarks)

        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Collecting", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    np.save(os.path.join(SAVE_DIR, f"{label}.npy"), np.array(data))
    print(f"Saved {len(data)} samples for label: {label}")

if __name__ == "__main__":
    for label in LABELS:
        input(f"\nPress Enter to start collecting for '{label}'...")
        capture(label)

