from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import io
from flask_cors import CORS
from disease_info import disease_info  

app = Flask(__name__)
CORS(app, origins=["https://agriverse-22076.web.app"])

# Load model & class indices once at startup
model = load_model('plant_disease_model.keras')

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()} 

def prepare_image(img_bytes):
    img = image.load_img(io.BytesIO(img_bytes), target_size=(160, 160))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

@app.route('/')
def home():
    return "AgriVerse Backend API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_bytes = file.read()
        x = prepare_image(img_bytes)

        preds = model.predict(x)
        pred_class_idx = np.argmax(preds, axis=1)[0]
        result_key = labels[pred_class_idx]

        # Lookup advice from disease_info dictionary
        info = disease_info.get(result_key, {
            "name": result_key,
            "advice": "No advice available for this disease."
        })

        return jsonify({
            "prediction": info["name"],
            "advice": info["advice"]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host='0.0.0.0', port=port, debug=True)
