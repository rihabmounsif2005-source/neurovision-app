from flask import Flask, render_template, request
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "brain_tumor_model.keras"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

LABELS_FR = {
    "glioma": "Gliome",
    "meningioma": "Méningiome",
    "notumor": "Pas de tumeur",
    "pituitary": "Tumeur hypophysaire",
}

history = []

model = tf.keras.models.load_model(MODEL_PATH)


def get_stats():
    total = len(history)

    counts = {
        "Gliome": 0,
        "Méningiome": 0,
        "Tumeur hypophysaire": 0,
        "Pas de tumeur": 0,
        "Metastatic": 0
    }

    for item in history:
        if item["class_name"] in counts:
            counts[item["class_name"]] += 1

    denominator = total if total > 0 else 1

    percentages = {
        "Gliome": round((counts["Gliome"] / denominator) * 100),
        "Méningiome": round((counts["Méningiome"] / denominator) * 100),
        "Tumeur hypophysaire": round((counts["Tumeur hypophysaire"] / denominator) * 100),
        "Pas de tumeur": round((counts["Pas de tumeur"] / denominator) * 100),
        "Metastatic": round((counts["Metastatic"] / denominator) * 100)
    }

    return {
        "total": total,
        "counts": counts,
        "percentages": percentages
    }


@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        history=history[-3:],
        stats=get_stats()
    )


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if file is None or file.filename == "":
        return render_template(
            "index.html",
            prediction=None,
            history=history[-3:],
            stats=get_stats()
        )

    image_path = UPLOAD_DIR / "uploaded_image.jpg"
    file.save(image_path)

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    arr = np.array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr)
    index = int(np.argmax(preds[0]))

    predicted_class = CLASS_NAMES[index]
    confidence = float(preds[0][index] * 100)

    class_name = LABELS_FR[predicted_class]
    result_text = "PAS DE TUMEUR" if predicted_class == "notumor" else "TUMEUR DÉTECTÉE"

    prediction = {
        "result": result_text,
        "class_name": class_name,
        "confidence": round(confidence, 1),
        "image": "uploads/uploaded_image.jpg"
    }

    history.append({
        "id": f"#{1250 + len(history)}",
        "result": result_text,
        "class_name": class_name,
        "confidence": round(confidence, 1)
    })

    return render_template(
        "index.html",
        prediction=prediction,
        history=history[-3:],
        stats=get_stats()
    )


if __name__ == "__main__":
    app.run(debug=True)