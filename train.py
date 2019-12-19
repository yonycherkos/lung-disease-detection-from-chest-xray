import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.models import load_model
from keras.utils import multi_gpu_model
from model import create_model

class train():
    """docstring for train."""

    def __init__(self):
        # training configurations
        self.train_path = "input/chest_xrays/train.csv"
        self.train_path = "input/chest_xrays/val.csv"
        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
                            'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

        self.target_size=(224,224,3)
        self.batch_size=32
        self.epochs=100
        self.initial_learning_rate = 0.001

        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)

        self.train_steps = np.ceil(len(train_df)/batch_size)
        self.validation_steps = np.ceil(len(val_df)/batch_size)

        # check output_dir, create it if not exists
        self.output_dir = "./output"
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, "models"))
            os.makedirs(os.path.join(self.output_dir, "weights"))
            os.makedirs(os.path.join(self.output_dir, "logs"))

        output_dir = "./output"
        model_path = os.path.join(output_dir, "models/chest_xray_imaging_model.h5")
        weights_path = os.path.join(output_dir, "weights/chest_xray_imaging_weights.h5")
        log_dir = os.path.join(output_dir, "logs")

    def train_callbacks():
        checkpoint = ModelCheckpoint(self.model_path,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     save_weights_only = False)

        reduceLR = ReduceLROnPlateau(monitor='val_loss',
                                     factor=0.1,
                                     patience=1,
                                     verbose=1,
                                     mode="min",
                                     min_lr=1e-8)

        tensorboard = TensorBoard(log_dir=self.log_dir)

        earlystop = EarlyStopping(monitor = 'val_loss',
                                  min_delta = 0,
                                  patience = 3,
                                  verbose = 1,
                                  restore_best_weights = True)

        callbacks = [checkpoint, reduceLR, tensorboard, earlystop]
        return callbacks

    def train():
        # resum training if exists
        if os.path.exists(self.model_path):
            # load model checkpoint
            print(".... loading trained model ....")
            model = load_model(self.model_path) # load the architecture, weigths, triaining configuration, and training status
        else:
            print("create new model")
            model = create_model()
            model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy",  metrics=["accuracy"])

        # check multiple gpu availability
        gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
        if gpus > 1:
            print(f"** multi_gpu_model is used! gpus={gpus} **")
            model = multi_gpu_model(model, gpus)
        else:
            print("there is no gpu in this device")

        # create data generator
        train_datagen = DataGenerator(dataset_csv_file = self.train_path,
                                      class_names = self.class_names,
                                      batch_size = self.batch_size,
                                      target_size = self.target_size,
                                      shuffle=True)

        validation_datagen = DataGenerator(dataset_csv_file = self.val_path,
                                          class_names = self.class_names,
                                          batch_size = self.batch_size,
                                          target_size = self.target_size,
                                          shuffle=True)

        callbacks = self.train_callbacks()
        # train the model
        model.fit_generator(train_datagen,
                            steps_per_epoch=train_steps,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=validation_datagen,
                            validation_steps=validation_steps,
                            class_weight=None,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=False,
                            shuffle=True,
                            initial_epoch=0)

# define custom datagenerator
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_csv_file, class_names, batch_size=32, target_size=(224,224,3), shuffle=True):
        'Initialization'
        self.dataset_df = pd.read_csv(dataset_csv_file)
        self.indexes = np.array(self.dataset_df['Image Index'])
        self.image_paths = self.dataset_df['path']
        self.labels = self.dataset_df[class_names]
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(image_path) for image_path in batch_image_paths])
        batch_x = self.normalize_batch_images(batch_x)
        batch_y = self.labels[index * self.batch_size: (index + 1) * self.batch_size]
        return batch_x, batch_y


    def load_image(self, image_path):
        image_path = os.path.join(os.getcwd(), image_path)
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = np.resize(image_array, self.target_size)
        return image_array

    def normalize_batch_images(self, batch_x):
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

if __name__ == '__main__':
    train = train()
    train.train()
