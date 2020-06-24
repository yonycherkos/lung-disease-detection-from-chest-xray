# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import utils
import config


class Test():
    def __init__(self):
        """[test the model performance with the given test configurations]

        Arguments:
            test_config {[dict]} -- [test configurations]
        """
        self.test_df = pd.read_csv(config.TEST_METADATA_PATH)
        self.test_steps = int(len(self.test_df) // config.BATCH_SIZE)

    def data_generator(self):
        """[generate test data generator]

        Returns:
            [ImageDataGenerator] -- [generated test data]
        """
        test_aug = ImageDataGenerator(rescale=1./255)

        test_datagen = test_aug.flow_from_datafram(self.test_df,
                                                   directory=config.TEST_PATH,
                                                   x_col="Image Index",
                                                   y_col=config.CLASS_NAMES,
                                                   target_size=(
                                                       224, 224),
                                                   class_mode='categorical',
                                                   batch_size=config.BATCH_SIZE,
                                                   shuffle=False)

        return test_datagen

    # TODO: is it necessary to calculate auroc score?
    def calculate_auroc(self, y_pred, y):
        test_log_path = os.path.join(config.OUTPUT_PATH, "test.log")
        aurocs = []
        with open(test_log_path, "w") as f:
            for idx, class_name in enumerate(config.CLASS_NAMES):
                try:
                    auroc_score = roc_auc_score(y[:, idx], y_pred[:, idx])
                    aurocs.append(auroc_score)
                except ValueError:
                    auroc_score = 0
                f.write(f"{class_name}: {auroc_score}\n")
            mean_auroc = np.mean(aurocs)
            f.write("-------------------------\n")
            f.write(f"mean auroc: {mean_auroc}\n")

        return (aurocs, mean_auroc)

    def test(self, model, test_datagen):
        """[test model performance for the given test data generator]

        Arguments:
            test_generator {[ImageDatagenrator]} -- [test image data generator]

        Returns:
            [tuple] -- [mean accuracy and mean auroc score]
        """

        # use the trained model to  make predictions on the test data
        preds = model.predict(test_datagen, verbose=1)

        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        predIdxs = np.argmax(preds, axis=1)

        # compute the classification report
        # TODO: write classification report to a file.
        report = classification_report(
            test_datagen.classes, predIdxs, target_names=test_datagen.class_indices.keys())

        # compute the confusion matrix and and use it to derive the raw
        # accuracy, sensitivity, and specificity
        cm = confusion_matrix(test_datagen.classes, predIdxs)
        total = sum(sum(cm))
        accuracy = (cm[0, 0] + cm[1, 1]) / total
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        # calcualte auroc score
        y = np.array(test_datagen.labels)
        aurocs, mean_auroc = self.calculate_auroc(preds, y)

        # evaluation result dictionary
        evaluation_result = {
            "report": report,
            "cm": cm,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "aurocs": aurocs,
            "mean_auroc": mean_auroc
        }

        return evaluation_result


if __name__ == "__main__":
    # load trained model
    print("[INFO] loading trained model ....")
    model = load_model(config.MODEL_PATH)

    # create and initialize Test object
    test = Test()

    # test data generator
    test_datagen = test.data_generator()

    # testing
    print("[INFO] evaluating model performance...")
    evaluation_result = test.test(model, test_datagen)

    # show the classification report
    print(evaluation_result["report"])

    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(evaluation_result["cm"])
    print("accuracy: {:.4f}".format(evaluation_result["accuracy"]))
    print("sensitivity: {:.4f}".format(evaluation_result["sensitivity"]))
    print("specificity: {:.4f}".format(evaluation_result["specificity"]))

    # show the mean auroc score
    print(evaluation_result["aurocs"])
    print("-----------------------------------------------------")
    print("mean_auroc {}".format(evaluation_result["mean_auroc"]))
