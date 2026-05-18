from flask import Flask, render_template, request
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

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


def preprocess_image(image_path):
    """
    ✅ CORRECTION PRINCIPALE du bug "toujours notumor".
    
    L'ancienne version utilisait preprocess_input de MobileNetV2
    qui normalise entre -1 et +1. Si ton modèle a été entraîné
    avec rescale=1./255, les valeurs d'entrée étaient complètement
    différentes → le modèle retournait toujours la même classe.
    
    On utilise maintenant /255.0 (normalisation standard).
    Si ça ne marche toujours pas, décommente OPTION B ci-dessous.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    # ✅ OPTION A : Normalisation /255 — la plus courante pour les modèles custom
    arr = arr / 255.0

    # ❌ OPTION B : MobileNetV2 preprocess_input (-1 à +1)
    # Décommente UNIQUEMENT si ton modèle a été entraîné avec preprocess_input natif
    # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    # arr = preprocess_input(arr)

    return arr


def get_stats():
    total = len(history)

    # ✅ CORRECTION : "Metastatic" supprimé — cette classe n'existe pas dans ton modèle
    counts = {
        "Gliome": 0,
        "Méningiome": 0,
        "Tumeur hypophysaire": 0,
        "Pas de tumeur": 0,
    }

    for item in history:
        if item["class_name"] in counts:
            counts[item["class_name"]] += 1

    denominator = total if total > 0 else 1
    percentages = {k: round((v / denominator) * 100) for k, v in counts.items()}

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

    # ✅ CORRECTION : Crée le dossier uploads s'il n'existe pas
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_path = UPLOAD_DIR / "uploaded_image.jpg"
    file.save(image_path)

    # ✅ Preprocessing corrigé
    arr = preprocess_image(image_path)

    preds = model.predict(arr)
    index = int(np.argmax(preds[0]))

    # ✅ DEBUG : Affiche toutes les probabilités dans le terminal pour diagnostic
    print("\n" + "=" * 50)
    print("📊 PRÉDICTIONS BRUTES :")
    for i, cls in enumerate(CLASS_NAMES):
        bar = "█" * int(preds[0][i] * 20)
        print(f"  {cls:15s}: {preds[0][i]*100:6.2f}%  {bar}")
    print(f"✅ Classe choisie : {CLASS_NAMES[index]} ({preds[0][index]*100:.1f}%)")
    print("=" * 50 + "\n")

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
