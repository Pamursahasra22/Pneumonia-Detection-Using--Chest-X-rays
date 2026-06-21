import os

# 1. SUPPRESS TENSORFLOW LOGS (Must be at the very top)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)

# Load the model once
MODEL_PATH = 'pneumonia_model.h5'
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully: {MODEL_PATH}")
else:
    print(f"❌ Error: {MODEL_PATH} not found. Run your training script first.")

def prepare_image(img_data):
    # Decode base64 image from JS
    encoded_data = img_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess
    img = cv2.resize(img, (150, 150))
    img = img.reshape(-1, 150, 150, 1) / 255.0
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image = prepare_image(data['image'])
        
        # verbose=0 removes the "1/1 [====]" log line
        prediction = model.predict(image, verbose=0)[0][0]
        
        # Logic: Model outputs closer to 1.0 for NORMAL, closer to 0.0 for PNEUMONIA
        normal_prob = float(prediction * 100)
        pneumonia_prob = float((1 - prediction) * 100)
        
        diagnosis = "NORMAL" if prediction > 0.5 else "PNEUMONIA"
        
        print(f"DEBUG: Predicted {diagnosis} (Confidence: {max(normal_prob, pneumonia_prob):.2f}%)")
        
        return jsonify({
            "diagnosis": diagnosis,
            "normal": round(normal_prob, 2),
            "pneumonia": round(pneumonia_prob, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Set debug=False to prevent the "History Restored" terminal crashes and double-loading
    print("🚀 PneumaScan Server starting at http://127.0.0.1:8080")
    app.run(host='127.0.0.1', port=8080, debug=False)