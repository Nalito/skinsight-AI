from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import base64
from huggingface_hub import hf_hub_download
#from recommender import get_recommendation

app = Flask(__name__)

# Enable CORS
from flask_cors import CORS
CORS(app)


# Load the TensorFlow model once at startup
#GITHUB_URL = "https://github.com/Nalito/skinsight-AI/releases/download/feliz/model_res.h5"
#local_file = tf.keras.utils.get_file("model_res.h5", origin=GITHUB_URL)
local_path = hf_hub_download(
    repo_id="Nalito-dev/skinsight",
    filename="model_res.h5"
)
model = tf.keras.models.load_model(local_path)
#model = tf.keras.models.load_model(local_file)

# Define your class names
class_names = ['Eczema', 'Folliculitis', 'Insect Bite', 'Tinea', 'Urticaria']
recommendations = ["Use topical corticosteroids (e.g., hydrocortisone) and regular emollient-based moisturizers; identify and avoid personal irritants and allergens.", "Apply topical antibiotic ointments such as mupirocin or clindamycin; consider antiseptic washes (e.g., chlorhexidine) and warm compresses.", "Use topical low-potency corticosteroid creams and oral antihistamines (e.g., cetirizine) to relieve itching and inflammation.", "Apply topical antifungals like terbinafine or clotrimazole; for widespread or refractory cases, use oral terbinafine.", "Administer second-generation oral antihistamines (e.g., cetirizine or loratadine) daily; add an H2-blocker if symptoms persist."]

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    arr = img_to_array(image) / 255.0
    return np.expand_dims(arr, axis=0)

def encode_image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{data}"

@app.route('/')
def home():
    return "Welcome to the skInsight!"

@app.route('/predict', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_data = None
    recommendation = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # Open and preprocess
            image = Image.open(file).convert('RGB')
            img_data = encode_image_to_base64(image)
            x = preprocess_image(image)

            # Predict
            preds = model.predict(x)
            idx = np.argmax(preds, axis=1)[0]
            prediction = class_names[idx]
            confidence = float(np.max(preds))
            recommendation = recommendations[idx]

    return jsonify({"prediction": prediction, "confidence": confidence, "recommendation": recommendation}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
