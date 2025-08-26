from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import io
import cv2
import numpy as np
import pickle
from gtts import gTTS
from transformers import MarianMTModel, MarianTokenizer
from tensorflow.keras.models import load_model
import mediapipe as mp

app = Flask(__name__)
CORS(app)  # Allow all origins for local dev

# Load gesture model and label encoder
model = load_model("model/gesture_model.h5")
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Translation setup
LANG_NAME_TO_GTTS = {"Hindi": "hi", "hi": "hi", "English": "en", "en": "en"}
LANG_TO_MODEL = {"Hindi": "Helsinki-NLP/opus-mt-en-hi", "hi": "Helsinki-NLP/opus-mt-en-hi"}

translation_models = {}
translation_tokenizers = {}

def load_translation_model(lang_key):
    if lang_key not in translation_models:
        model_name = LANG_TO_MODEL.get(lang_key)
        if not model_name:
            return None, None
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        tmodel = MarianMTModel.from_pretrained(model_name)
        translation_models[lang_key] = tmodel
        translation_tokenizers[lang_key] = tokenizer
    return translation_models[lang_key], translation_tokenizers[lang_key]

def offline_translate(text, lang):
    if not text.strip():
        return ""
    key = lang if lang in LANG_TO_MODEL else None
    if not key:
        return "(unsupported language)"
    tmodel, tokenizer = load_translation_model(key)
    if not tmodel:
        return "(no model)"
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    out = tmodel.generate(**inputs)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def read_image_from_request(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files.get("image")
    if not file:
        return jsonify(error="no image"), 400

    img = read_image_from_request(file)
    if img is None:
        return jsonify(prediction="", confidence=0.0)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return jsonify(prediction="", confidence=0.0)

    lm = results.multi_hand_landmarks[0].landmark
    landmarks = np.array([[p.x, p.y, p.z] for p in lm]).flatten().reshape(1, -1)
    pred_probs = model.predict(landmarks, verbose=0)[0]
    class_id = int(np.argmax(pred_probs))
    label = label_encoder.inverse_transform([class_id])[0]
    confidence = float(pred_probs[class_id])
    return jsonify(prediction=label, confidence=confidence)

@app.route("/translate", methods=["POST"])
def translate_api():
    data = request.get_json() or {}
    sentence = data.get("sentence", "")
    lang = data.get("lang", "hi")
    translated = offline_translate(sentence, lang)
    return jsonify(translation=translated)

@app.route("/speak", methods=["POST"])
def speak_api():
    data = request.get_json() or {}
    text = data.get("text", "")
    lang_code = LANG_NAME_TO_GTTS.get(data.get("lang", "hi"), "hi")
    if not text.strip():
        return jsonify(error="empty text"), 400
    tts = gTTS(text=text, lang=lang_code)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return send_file(mp3_fp, mimetype="audio/mpeg")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
