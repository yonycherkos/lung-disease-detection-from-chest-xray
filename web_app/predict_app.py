import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
import os
import io
from flask import Flask
from flask import request
from flask import jsonify
from flask import send_file
from flask_cors import CORS
import json

import prediction

class_names = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema","Fibrosis",
      "Hernia","Infiltration","Mass","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"]

image_path = "./output/prediction_graph.png"

# loading model
output_dir = "output/"
model_architecture_path = os.path.join(output_dir, "model_architecure/xray_imaging_architecture.json")
model_weights_path = os.path.join(output_dir, "model_weights/xray_imaging_weights.h5")
print(" * Loading Keras model...")
prediction.get_model(model_architecture_path, model_weights_path)

def plot_prediction_graph(predictions):
    index = np.arange(len(class_names))
    plt.bar(index, predictions)
    plt.xlabel('class_names', fontsize=5)
    plt.ylabel('predictions', fontsize=5)
    plt.xticks(index, class_names, fontsize=5, rotation=30)
    plt.title('prediction result')
    plt.savefig(image_path)

def image_to_string(image_path):
    with open(image_path, 'rb') as file:
        byte_content = file.read()
    base64_bytes = base64.b64encode(byte_content)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

app = Flask(__name__)
CORS(app)
@app.route("/predict", methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    predictions = prediction.predict(prediction.model, image)

    percentage_predictions = np.round(np.array(predictions[0])*100, 2).tolist()
    plot_prediction_graph(percentage_predictions)
    base64_string = image_to_string(image_path)

    response = {
        "prediction": predictions, "prediction_graph": base64_string
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run()
