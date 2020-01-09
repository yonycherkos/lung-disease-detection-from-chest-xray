import numpy as np
from PIL import Image
import base64
import os
import io
from flask import Flask
from flask import request
from flask import jsonify
from flask import send_file
from flask_cors import CORS

import prediction

# loading model
output_dir = "output/"
model_architecture_path = os.path.join(output_dir, "model_architecure/xray_imaging_architecture.json")
model_weights_path = os.path.join(output_dir, "model_weights/xray_imaging_weights.h5")
print(" * Loading Keras model...")
prediction.get_model(model_architecture_path, model_weights_path)

app = Flask(__name__)
CORS(app
@app.route("/predict", methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    predictions = prediction.predict(prediction.model, image)
    response = {
        "prediction": predictions
    }
    return response


if __name__ == "__main__":
    app.run()
