from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the Keras model
model = load_model("model.h5")  # 👈 replace with your actual filename

# Define the class labels
class_labels = [ "angular_leafspot", "anthracnose_fruit_rot", "blossom_blight",
    "gray_mold", "leaf_spot", "powdery_mildew_fruit", "powdery_mildew_leaf"
 ]  # Update as needed

@app.route("/", methods=["GET"])
def home():
    return "<h1>🍓 Strawberry Leaf Disease Classifier API</h1>"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    
    # Preprocess the image
    image = image.resize((224, 224))  # Or the size your model expects
    image = img_to_array(image)
    image = image / 255.0  # Normalization (if model trained on normalized images)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]

    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)

