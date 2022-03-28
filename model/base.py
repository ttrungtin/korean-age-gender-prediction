from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, Flatten, Dense
from tensorflow.keras.models import Model


def create_model(input_shape=(160, 160, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, strides=2, padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    outputs = Dense(1, name='age')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
