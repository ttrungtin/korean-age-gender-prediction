from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, Flatten, Dense
from tensorflow.keras.models import Model


def create_model_all(input_shape=(160, 160, 3), num_classes=10):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, strides=2, padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(10)(x)

    outputs_cate = Dense(num_classes, activation='softmax', name='cate')(x)
    outputs_reg = Dense(1, name='reg')(x)

    model = Model(inputs=inputs, outputs=[outputs_cate, outputs_reg])
    return model


def create_model_cate(input_shape=(160, 160, 3), num_classes=10):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, strides=2, padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(10)(x)

    outputs_cate = Dense(num_classes, activation='softmax', name='cate')(x)

    model = Model(inputs=inputs, outputs=[outputs_cate])
    return model


def create_model_reg(input_shape=(160, 160, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, strides=2, padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(10)(x)

    outputs_reg = Dense(1, name='reg')(x)

    model = Model(inputs=inputs, outputs=[outputs_reg])
    return model