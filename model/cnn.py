from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, Flatten, Dense, ReLU, \
    AveragePooling2D, Dropout
from tensorflow.keras.models import Model


def create_model_all(input_shape=(160, 160, 3), num_classes=10, image_net_pre_train=False):
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

    x = Conv2D(32, kernel_size=3, padding='valid', strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(32, kernel_size=1, padding='valid', strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Flatten()(x)

    if image_net_pre_train:
        output = Dense(num_classes, name='cate')(x)
        model = Model(inputs=inputs, outputs=[output])
        return model

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    output_cate = Dense(num_classes, activation='softmax', name='cate')(x)
    output_reg = Dense(1, name='reg')(output_cate)
    model = Model(inputs=inputs, outputs=[output_cate, output_reg])
    return model


def create_top_all(pre_train_model, num_classes):
    top = pre_train_model.layers[-2].output  # flatten

    x = Dense(64, activation='relu')(top)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    output_cate = Dense(num_classes, activation='softmax', name='cate')(x)
    output_reg = Dense(1, name='reg')(x)
    model = Model(inputs=pre_train_model.inputs, outputs=[output_cate, output_reg])
    return model