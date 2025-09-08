from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import traceback

app = Flask(__name__, static_folder='static')
CORS(app, origins="*")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
MODEL_PATH = './keras_model.h5'
LABELS_PATH = './labels.txt'
np.set_printoptions(suppress=True)

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # This will prevent the app from starting if the model is missing
    raise

try:
    with open(LABELS_PATH, 'r') as f:
        class_names = [line.strip()[2:] for line in f.readlines()]
    print("✅ Labels loaded:", class_names)
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    # This will prevent the app from starting if the labels are missing
    raise

# Custom error handler for internal server errors
@app.errorhandler(500)
def handle_internal_server_error(e):
    traceback.print_exc()
    return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

@app.route('/')
def serve_main():
    return send_from_directory('.', 'mainpage.html')

@app.route('/about.html')
def serve_about():
    return send_from_directory('.', 'about.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("Error: No file part in the request.")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("Error: No file selected.")
        return jsonify({"error": "No file selected"}), 400

    if not file.mimetype in ['image/jpeg', 'image/png', 'image/gif']:
        print(f"Error: Invalid file type {file.mimetype}")
        return jsonify({"error": "Invalid file type"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        print(f"✅ File saved temporarily to {file_path}")

        print("Attempting to open and preprocess image...")
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)
        print("✅ Image pre-processed successfully.")

        print("Running model prediction...")
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])
        print("✅ Prediction complete.")

        return jsonify({
            "animal": class_name,
            "confidence": confidence_score,
            "info": f"This appears to be a {class_name.lower()}."
        })

    except Exception as e:
        print(f"❌ An error occurred during file processing or prediction: {e}")
        traceback.print_exc() # Print full traceback to logs
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print("✅ Temporary file removed.")

if __name__ == '__main__':
    app.run(debug=True)
