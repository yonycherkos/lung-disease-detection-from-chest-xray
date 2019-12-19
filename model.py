from tensorflow import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras.applications.densenet import DenseNet121
from keras.optimizers import Adam

# fine-tune densenet model
def create_model(input_shape=(224, 224, 3), num_classes=14, show_summary=False):
    img_input = Input(input_shape)
    densenet_model = DenseNet121(include_top=False,
                                 weights="imagenet",
                                 input_tensor=img_input,
                                 input_shape=input_shape,
                                 pooling="avg"
                                )
    x = densenet_model.output
    predictions = Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=img_input, outputs=predictions)
    if show_summary:
        model.summary()

    return model
