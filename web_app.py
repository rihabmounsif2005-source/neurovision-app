import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "brain_tumor_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["notumor", "glioma", "meningioma", "pituitary"]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Méningiome",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}

def prepare_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image reçue"})

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Aucun fichier choisi"})

    img = prepare_image(file)
    prediction = model.predict(img)[0]

    index = int(np.argmax(prediction))
    class_key = CLASS_NAMES[index]
    confidence = float(prediction[index]) * 100

    result = DISPLAY_NAMES[class_key]

    return jsonify({
        "result": result,
        "confidence": round(confidence, 2),
        "probabilities": {
            DISPLAY_NAMES[CLASS_NAMES[i]]: round(float(prediction[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
