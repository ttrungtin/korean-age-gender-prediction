from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Flatten, Dense, ReLU, \
    AveragePooling2D, MaxPooling2D, Concatenate
from tensorflow.keras.models import Model


def base(input_shape=(64, 64, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=3, padding='valid', strides=1, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=2, strides=2)(x)

    x = Conv2D(32, kernel_size=3, padding='valid', strides=1)(x)
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

    x = Conv2D(32, kernel_size=1, padding='valid', strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = Flatten()(x)

    model = Model(inputs=[inputs], outputs=[x])

    return model


def create_model_all(input_shape=(64, 64, 3), num_classes=10):
    base_model = base(input_shape=input_shape)

    x1 = Input(shape=input_shape)
    x2 = Input(shape=input_shape)
    x3 = Input(shape=input_shape)

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)

    concat = Concatenate(axis=-1)([y1, y2, y3])

    output_cate = Dense(num_classes, activation='softmax', name='cate')(concat)
    output_reg = Dense(1, name='reg')(output_cate)

    return Model(inputs=[x1, x2, x3], outputs=[output_cate, output_reg])
