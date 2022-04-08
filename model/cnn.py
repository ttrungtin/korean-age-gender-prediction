from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, Flatten, Dense, ReLU, \
    AveragePooling2D
from tensorflow.keras.models import Model


def create_model_all(input_shape=(160, 160, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=3, padding='valid', strides=1)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(32, kernel_size=3, padding='valid', strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(32, kernel_size=3, padding='valid', strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(32, kernel_size=3, padding='valid', strides=1)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(32, kernel_size=1, padding='valid', strides=1)(inputs)

    x = Flatten()(x)
    output_cate = Dense(10, activation='softmax', name='cate')(x)
    output_reg = Dense(1, name='reg')(x)

    model = Model(inputs=inputs, outputs=[output_cate, output_reg])
    return model