import requests
import json
from flask import Flask, request, render_template
from flask_cors import CORS  # ✅ CORS import
from keras.layers import TFSMLayer
import numpy as np
from PIL import Image
import io

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# ✅ Load the model
model = TFSMLayer("optimized_model", call_endpoint="serving_default")
print("✅ Model loaded successfully!")

# ✅ Define class labels
class_names = [
    "angular_leafspot", "anthracnose_fruit_rot", "blossom_blight",
    "gray_mold", "leaf_spot", "powdery_mildew_fruit", "powdery_mildew_leaf"
]

# ✅ ThingsBoard setup
ACCESS_TOKEN = 'EtaceB6Oeypyb8gBbyPc'  # Replace with your actual token
THINGSBOARD_HOST = 'http://43.204.100.33:8080'

def send_to_thingsboard(payload):
    url = f'{THINGSBOARD_HOST}/api/v1/{ACCESS_TOKEN}/telemetry'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.status_code, response.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files['file']

        # ✅ Preprocess image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # ✅ Run inference
        prediction_dict = model(image)
        prediction_tensor = list(prediction_dict.values())[0]

        try:
            prediction = prediction_tensor.numpy()
        except AttributeError:
            prediction = prediction_tensor

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # ✅ Send to ThingsBoard
        payload = {
            'predicted_class': predicted_class,
            'predicted_label': class_names[predicted_class],
            'confidence': confidence
        }
        status_code, response_text = send_to_thingsboard(payload)

        return render_template(
            "index.html",
            prediction=f"Predicted Disease: {class_names[predicted_class]}",
            confidence=f"Confidence: {confidence:.2%}",
            tb_status=f"ThingsBoard Status: {status_code}"
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


