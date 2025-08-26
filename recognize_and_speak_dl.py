import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa
from tensorflow.keras.models import load_model
import tempfile
import tkinter as tk
from tkinter import ttk
from threading import Thread
from transformers import MarianMTModel, MarianTokenizer


model = load_model("model/gesture_model.h5")
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


sentence = []
prediction_buffer = []
buffer_size = 10
last_prediction = ""


LANG_NAME_TO_GTTS = {
    "Hindi": "hi"
}

SUPPORT_LANGUAGES = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi"
}

translation_models = {}
translation_tokenizers = {}


def speak_text(text, lang_code='hi'):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_fp:
            tts = gTTS(text=text, lang=lang_code)
            tts.save(mp3_fp.name)
        sound = AudioSegment.from_file(mp3_fp.name, format="mp3")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_fp:
            sound.export(wav_fp.name, format="wav")
            wave_obj = sa.WaveObject.from_wave_file(wav_fp.name)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        time.sleep(0.2)
        os.remove(mp3_fp.name)
        os.remove(wav_fp.name)
    except Exception as e:
        print(f"[TTS Error] {e}")


def speak_sentence():
    if sentence:
        full = " ".join(sentence)
        Thread(target=speak_text, args=(full, 'hi'), daemon=True).start()


def clear_sentence():
    sentence.clear()
    update_gui()
    translation_var.set("Translation:")


def update_gui():
    sentence_str = " ".join(sentence)
    sentence_var.set(f"Sentence: {sentence_str}")


def load_model_tokenizer(language):
    if language not in translation_models:
        model_name = SUPPORT_LANGUAGES.get(language)
        if model_name is None:
            return None, None
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translation_models[language] = model
        translation_tokenizers[language] = tokenizer
    return translation_models.get(language), translation_tokenizers.get(language)


def offline_translate(text, language):
    if not text.strip():
        return text
    model, tokenizer = load_model_tokenizer(language)
    if model is None or tokenizer is None:
        return "(No Hindi translation model)"
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


def translate_sentence():
    if not sentence:
        translation_var.set("Translation:")
        return
    source_text = " ".join(sentence)
    target_lang = language_var.get()
    def do_translate():
        translated_text = offline_translate(source_text, target_lang)
        translation_var.set(f"Translation ({target_lang}): {translated_text}")
    Thread(target=do_translate, daemon=True).start()


def speak_translation():
    translation = translation_var.get()
    if ':' in translation:
        translated_text = translation.split(':', 1)[1].strip()
        lang_code = LANG_NAME_TO_GTTS.get(language_var.get(), 'hi')
        Thread(target=speak_text, args=(translated_text, lang_code), daemon=True).start()


root = tk.Tk()
root.title("Sign Language Recognition - Hindi Only")
window_width = 480
window_height = 440
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.configure(bg="#f5f5f5")


sentence_var = tk.StringVar()
sentence_var.set("Sentence: ")
sentence_label = tk.Label(root, textvariable=sentence_var, font=("Segoe UI", 16, "bold"), bg="#f5f5f5", fg="#333")
sentence_label.pack(pady=(20, 10), padx=20, fill="x")

buttons_frame = tk.Frame(root, bg="#f5f5f5")
buttons_frame.pack(pady=(0, 15))

speak_btn = tk.Button(buttons_frame, text="Speak Sentence", command=speak_sentence,
                      font=("Segoe UI", 13), bg="#4CAF50", fg="white", activebackground="#45a049", padx=20, pady=8)
speak_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(buttons_frame, text="Clear Sentence", command=clear_sentence,
                      font=("Segoe UI", 13), bg="#f44336", fg="white", activebackground="#da190b", padx=20, pady=8)
clear_btn.grid(row=0, column=1, padx=10)

separator = ttk.Separator(root, orient='horizontal')
separator.pack(fill='x', pady=10, padx=20)

language_var = tk.StringVar()
language_var.set("Hindi")
language_menu = ttk.OptionMenu(root, language_var, "Hindi")
language_menu.config(width=15)
language_menu.pack(pady=10)

translate_btn = tk.Button(root, text="Translate Sentence", command=translate_sentence,
                          font=("Segoe UI", 13), bg="#2196F3", fg="white", activebackground="#0b7dda", padx=20, pady=8)
translate_btn.pack(pady=5)

speak_translation_btn = tk.Button(root, text="Speak Translation", command=speak_translation,
                                  font=("Segoe UI", 13), bg="#FF9800", fg="white", activebackground="#e68a00", padx=20, pady=8)
speak_translation_btn.pack(pady=5)

translation_var = tk.StringVar()
translation_var.set("Translation:")
translation_label = tk.Label(root, textvariable=translation_var, font=("Segoe UI", 14), bg="#f5f5f5",
                             fg="#444", wraplength=440, justify="left")
translation_label.pack(pady=(10, 20), padx=20, fill="x")


def webcam_loop():
    global last_prediction
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        prediction = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                if len(landmarks) == 63:
                    input_data = np.array(landmarks).reshape(1, -1)
                    pred = model.predict(input_data, verbose=0)[0]
                    class_id = np.argmax(pred)
                    prediction = label_encoder.inverse_transform([class_id])[0]
                    prediction_buffer.append(prediction)
                    if len(prediction_buffer) > buffer_size:
                        prediction_buffer.pop(0)
                    if prediction_buffer.count(prediction) == buffer_size and prediction != last_prediction:
                        sentence.append(prediction)
                        update_gui()
                        last_prediction = prediction
        else:
            prediction_buffer.clear()
        cv2.putText(frame, f"Current: {prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Language Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


Thread(target=webcam_loop, daemon=True).start()
root.mainloop()

