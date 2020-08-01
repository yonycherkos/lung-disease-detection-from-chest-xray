# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import cv2
from tensorflow.keras.models import load_model
from helper import config, utils, heatmap


def preprocess_image(image, target_size=(224, 224)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image_array = np.array(image)
    image_array = image_array / 255.
    image_tensor = np.expand_dims(image_array, axis=0)
    return image_tensor


def predict(model, processed_image):
    # predict on preprocessed image
    print("[INFO] make prediction on sample image")
    prediction = model.predict(processed_image).tolist()
    prediction = np.round(prediction, 3)

    # sort prediction result based on confidence score
    prediction_map = utils.sort_prediction(prediction)

    return prediction_map


if __name__ == "__main__":
    # define argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())

    # load trained model
    print("[INFO] loading trained model...")
    model = load_model(config.MODEL_PATH)

    # process image before prediction
    # TODO: why not use keras imutils_imagenet to preprocess the input image.
    image = cv2.imread(args["image"])
    processed_image = preprocess_image(image)

    # prediction on the processed image
    prediction_map = predict(model, processed_image)
    for (key, value) in prediction_map.items():
        print(f"{key}: {value}")

    # generate heatmap
    heatmap_img = heatmap.create_heatmap(model, image, processed_image)
    cv2.imshow("heatmap image", heatmap_img)
    cv2.waitkey(0)
