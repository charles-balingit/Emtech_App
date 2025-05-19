import os
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template_string, redirect, url_for

app = Flask(__name__)

# Load the TensorFlow model from 'model' directory
MODEL_PATH = "model"
model = tf.keras.models.load_model('iris_model.h5')

# Assume model expects images of this size
IMAGE_SIZE = (224, 224)  # typical size, change if your model expects something else

# Example class names - the user should replace this with actual class names for their model
CLASS_NAMES = ["class_0", "class_1", "class_2", "class_3", "class_4"]

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>TensorFlow Image Model Deployment</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background: #f5f7fa;
            color: #333;
            text-align: center;
        }
        h1 {
            color: #0078d7;
        }
        form {
            margin: 20px auto;
            padding: 20px;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 400px;
        }
        input[type=file] {
            margin: 10px 0;
        }
        button {
            background-color: #0078d7;
            color: #fff;
            border: none;
            padding: 10px 20px;
            margin-top: 10px;
            cursor: pointer;
            font-size: 1em;
            border-radius: 4px;
        }
        button:hover {
            background-color: #005a9e;
        }
        .result {
            margin-top: 30px;
            background: #e1f0ff;
            display: inline-block;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.2);
            font-weight: bold;
            font-size: 1.1em;
        }
        .uploaded-image {
            max-width: 300px;
            margin-top: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
    </style>
</head>
<body>
    <h1>TensorFlow Image Model Deployment</h1>
    <form action="{{ url_for('predict') }}" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required />
        <br/>
        <button type="submit">Predict</button>
    </form>

    {% if result %}
    <div class="result">Prediction: {{ result }}</div>
    <br/>
    <img src="{{ image_url }}" alt="Uploaded Image" class="uploaded-image"/>
    {% endif %}
</body>
</html>
"""

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the input image to the format expected by the model."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    return img_array

@app.route("/", methods=["GET"])
def index():
    return render_template_string(html_template)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    try:
        img = Image.open(file.stream)
    except Exception:
        return "Invalid image file", 400

    # Preprocess for model
    input_arr = preprocess_image(img)

    # Predict
    predictions = model.predict(input_arr)
    pred_index = np.argmax(predictions, axis=1)[0]
    pred_class = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else f"Class {pred_index}"

    # Convert image to base64 for inline display
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    import base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')
    image_url = f"data:image/png;base64,{img_base64}"

    return render_template_string(html_template, result=pred_class, image_url=image_url)

if __name__ == "__main__":
    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)

