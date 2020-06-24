# import the necessary packages
import numpy as np
import config


def get_class_counts(df, class_names):
    total_count = df.shape[0]
    labels = df[class_names]
    class_counts = np.sum(labels, axis=0)
    class_counts_dict = dict(zip(class_names, class_counts))
    return total_count, class_counts_dict


def compute_class_weight(total_count, class_counts_dict):
    class_weight = {}
    for (i, class_count) in class_counts_dict.values():
        class_weight[i] = (total_count - class_count)/total_count
    return class_weight


def sort_prediction(prediction):
    prediction_map = dict(zip(config.CLASS_NAMES, prediction))
    sorted_prediction_map = {sorted_class_name: prediction_map[sorted_class_name] for sorted_class_name in sorted(
        prediction_map, key=prediction_map.get, reverse=True)}

    return sorted_prediction_map
