# import the necessary packages
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.keras.utils import multi_gpu_model
from sklearn.model_selection import roc_auc_score
import utils
import config


class Train():
    def __init__(self):
        """[train the model with the given train configurations.]
        """
        self.train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
        self.val_df = pd.read_csv(config.VAL_METADATA_PATH)

        self.train_steps = int(len(self.train_df) // config.BATCH_SIZE)
        self.val_steps = int(len(self.val_df) // config.BATCH_SIZE)

    def build_model(self, show_summary=False):
        """[Finetune a pre-trained densenet model]

        Keyword Arguments:
            show_summary {bool} -- [show model summary] (default: {False})
        """
        img_input = Input(shape=(224, 224, 3))
        base_model = DenseNet121(include_top=False,
                                 weights="imagenet",
                                 input_tensor=img_input,
                                 input_shape=(224, 224, 3),
                                 pooling="avg"
                                 )
        # TODO: add additional dense layers.
        output = Dense(len(config.CLASS_NAMES),
                       activation="sigmoid", name="output")(base_model.output)

        model = Model(inputs=img_input, outputs=output)
        if show_summary:
            print(model.summary())

        return model

    def data_generator(self):
        """[Generate train and val data generators]

        Returns:
            [tuple(ImageDataGenerator)] -- [train and val image datagenerator]
        """
        train_aug = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=False)  # TODO: do we need apply imagenet mean and std.
        val_aug = ImageDataGenerator(rescale=1./255)

        train_datagen = train_aug.flow_from_datafram(self.train_df,
                                                     directory=None,  # can be none if x_col is full image path
                                                     x_col=self.train_df["Image Path"],
                                                     y_col=config.CLASS_NAMES,
                                                     target_size=(
                                                         224, 224),
                                                     class_mode='categorical',
                                                     batch_size=config.BATCH_SIZE,
                                                     shuffle=True)

        val_datagen = val_aug.flow_from_datafram(self.val_df,
                                                 directory=None,
                                                 x_col=self.train_df["Image Path"],
                                                 y_col=config.CLASS_NAMES,
                                                 target_size=(
                                                     224, 224),
                                                 class_mode='categorical',
                                                 batch_size=config.BATCH_SIZE,
                                                 shuffle=False)

        return (train_datagen, val_datagen)

    def callbacks(self):
        """[Configure training callbacks]

        Returns:
            [List] -- [list of callbacks]
        """
        checkpoint = ModelCheckpoint(config.MODEL_PATH,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     save_weights_only=False)

        reduceLR = ReduceLROnPlateau(monitor='val_loss',
                                     factor=0.1,
                                     patience=1,
                                     verbose=1,
                                     mode="min",
                                     min_lr=config.MIN_LR)

        tensorboard = TensorBoard(log_dir=config.LOG_DIR)

        earlyStop = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=3,
                                  verbose=1,
                                  restore_best_weights=True)

        callbacks = [checkpoint, reduceLR, tensorboard, earlyStop]

        return callbacks

    def train(self, model, train_datagen, val_datagen, callbacks):
        """[Train a with the given train configurations,
            if there is an existed trained model, resume training from where it has stop training.
            if not create new model and compile it.
            compute class weights to solve the data imbalance.
            fit the train and val generator to the model.
            ]

        Arguments:
            model {[Model]} -- [keras functional model]
            train_datagen {[ImageDatagenerator]} -- [train data generator]
            val_datagen {[ImageDatagenerator]} -- [val data generator]
            callbacks {[List]} -- [list of callbacks]
        """
        # resume training if prevously trained model exists
        if os.path.exists(config.MODEL_PATH):
            # load trained model
            print("[INFO] load trained model...")
            model = load_model(config.MODEL_PATH)
        else:
            print("[INFO] create new model...")
            # make directories to store the training outputs,.
            output_paths = [config.MODEL_PATH, config.LOG_DIR]
            for ouput_path in output_paths:
                os.makedirs(ouput_path)

            # model = self.build_model()
            print("[INFO] compile the model")
            model.compile(optimizer=Adam(lr=config.INTIAL_LR),
                          loss="binary_crossentropy",
                          metrics=["accuracy"])

        # compute class weights
        total_count, class_count_dict = utils.get_class_counts(
            self.train_df, config.CLASS_NAMES)
        class_weight = utils.compute_class_weight(
            total_count, class_count_dict)

        # check multiple gpu availability
        # TODO: how to train model on multiple gpu?
        gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
        # gpus = len(tf.config.experimental.list_physical_devices("GPU"))
        if gpus > 1:
            print(f"[INFO] multi_gpu_model is used! gpus={gpus}")
            model = multi_gpu_model(model, gpus)
        else:
            print("[INFO] there is no gpu in this device")

        # fit the train and validation datagen to the model
        print("[INFO] training the model..")
        model.fit(train_datagen,
                  epochs=config.EPOCHS,
                  verbose=1,
                  callbacks=callbacks,
                  # TODO: need to be tuple (x_val, y_vall)
                  validation_data=val_datagen,
                  shuffle=True,
                  class_weight=class_weight,
                  steps_per_epoch=self.train_steps,
                  validation_steps=self.val_steps
                  )

        # save trained model explicitly
        print("[INFO] save the trained model")
        model.save(config.MODEL_PATH)


if __name__ == "__main__":
    # create and initialize Train object
    train = Train()

    # build the model
    model = train.build_model(show_summary=True)

    # train and val datagenrator
    (train_datagen, val_datagen) = train.data_generator()

    # training callbacks
    callbacks = train.callbacks()

    # train the modela
    train.train(model, train_datagen, val_datagen, callbacks)
