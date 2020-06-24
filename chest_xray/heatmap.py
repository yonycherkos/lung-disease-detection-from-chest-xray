# import the necessary packages
import numpy as np
import cv2
from tensorflow.keras.backend import function
from utils import sort_prediction

def create_heatmap(model, image, processed_image, target_size=(224, 224)):
    # build tensorflow computational graph
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = model.layers[-3]  # "bn" layer
    get_output = function([model.layers[0].input], [
                          final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output(processed_image)
    conv_outputs = conv_outputs[0, :, :, :]

    # compute class activation map(CAM)
    class_idx = np.argmax(predictions.tolist()[0])
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i, weight in enumerate(class_weights[:, class_idx]):
        cam += weight * conv_outputs[:, :, i]

    # merge the original image and the class activation map
    cam /= np.max(cam)
    cam = cv2.resize(cam, target_size)
    heatmap_img = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_img[np.where(cam < 0.2)] = 0
    heatmap_img = heatmap_img * 0.5 + image

    # extract top3 prediction class name
    prediction_map = sort_prediction(predictions)
    top3_prediction_keys = list(prediction_map.keys())[:3]
    top3_prediction_values = list(prediction_map.values())[:3]
    texts = [
        f"{top3_prediction_keys[0]}: {top3_prediction_values[0]}",
        f"{top3_prediction_keys[1]}: {top3_prediction_values[1]}",
        f"{top3_prediction_keys[2]}: {top3_prediction_values[2]}"
    ]

    # add label to the heatmap
    cv2.putText(heatmap_img, text=texts[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=1)
    cv2.putText(heatmap_img, text=texts[1], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 255), thickness=1)
    cv2.putText(heatmap_img, text=texts[2], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=1)

    return heatmap_img
