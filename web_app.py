import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "brain_tumor_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Méningiome",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}

def prepare_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Aucune image reçue"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Aucun fichier choisi"}), 400

        img = prepare_image(file)
        prediction = model.predict(img)[0]

        index = int(np.argmax(prediction))
        class_key = CLASS_NAMES[index]
        confidence = float(prediction[index]) * 100

        if class_key == "notumor":
            final_result = "No Tumor"
            status = "Sain"
        else:
            final_result = DISPLAY_NAMES[class_key]
            status = "Tumeur détectée"

        probabilities = {}
        for i, key in enumerate(CLASS_NAMES):
            probabilities[DISPLAY_NAMES[key]] = round(float(prediction[i]) * 100, 2)

        return jsonify({
            "status": status,
            "result": final_result,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
