import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model_path = os.path.join("model", "Brain_Tumor_Model.h5")
model = tf.keras.models.load_model(model_path)

# Define class labels (adjust if different)
class_labels = {0: "No Tumor", 1: "Tumor Detected"}

def preprocess_image(image_path):
    """Load and preprocess the image for model prediction."""
    try:
        img = cv2.imread(image_path)

        if img is None:
            return None, "Error: Image not found or corrupted!"

        img = cv2.resize(img, (256, 256))

        # Ensure 3-channel RGB image
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = img.astype("float32") / 255.0  # Normalize to [0,1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        return img, None
    except Exception as e:
        return None, f"Error preprocessing image: {str(e)}"

def predict_tumor(image_path):
    """Predicts if an image has a tumor and returns the result with confidence."""
    img, error = preprocess_image(image_path)
    if error:
        return error, None, None

    try:
        pred = model.predict(img)

        # Handle Softmax Output (Multi-Class)
        if pred.shape[1] > 1:
            predicted_class = np.argmax(pred[0])
            confidence = pred[0][predicted_class] * 100

        # Handle Sigmoid Output (Binary Classification)
        else:
            predicted_class = int(pred[0][0] > 0.5)
            confidence = pred[0][0] * 100 if predicted_class == 1 else (1 - pred[0][0]) * 100

        return class_labels[predicted_class], confidence, None

    except Exception as e:
        return None, None, f"Error predicting: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file part in request!")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", message="No file selected!")

        # Save uploaded file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        print(f"Uploaded file path: {filepath}")  # Debugging print

        # Ensure file is saved correctly
        if not os.path.exists(filepath):
            return render_template("index.html", message="Error saving file!")

        # Predict tumor presence
        result, confidence, error = predict_tumor(filepath)

        if error:
            return render_template("index.html", message=error)

        return render_template(
            "index.html",
            message=f"Prediction: {result} (Confidence: {confidence:.2f}%)",
            image_url=url_for("static", filename=f"uploads/{file.filename}"),
        )

    return render_template("index.html", message="Upload an image for prediction.")

if __name__ == "__main__":
    app.run(debug=True)
