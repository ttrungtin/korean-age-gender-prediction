import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Dense, ReLU, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

Activation = ReLU


class GeM(Layer):

    def __init__(self, init_p=3., dynamic_p=False, **kwargs):
        # https://arxiv.org/pdf/1711.02512.pdf
        # add this behind relu
        if init_p <= 0:
            raise Exception("fatal p")
        super(GeM, self).__init__(**kwargs)
        self.init_p = init_p
        self.epsilon = 1e-8
        self.dynamic_p = dynamic_p

    def build(self, input_shape):
        super(GeM, self).build(input_shape)
        if self.dynamic_p:
            self.init_p = tf.Variable(self.init_p, dtype=tf.float32)

    def call(self, inputs):
        pool = tf.nn.avg_pool(tf.pow(tf.math.maximum(inputs, self.epsilon), self.init_p), inputs.shape[1:3],
                              strides=(1, 1), padding="VALID")
        pool = tf.pow(pool, 1. / self.init_p)
        return pool


def white_norm(input):
    return (input - tf.constant(127.5)) / 128.0


def BRA(input):
    bn = BatchNormalization()(input)
    activation = Activation()(bn)
    return AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(activation)


def BN_ReLU(input, name):
    return Activation()(BatchNormalization()(input))


def SE_BLOCK(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = GeM(dynamic_p=True)(input)  # GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums // r_factor)(ga_pooling)
    scale = Dense(channel_nums, activation="sigmoid")(Activation()(fc1))
    return multiply([scale, input])


def base(input_shape=(64, 64, 3), using_white_norm=True, using_SE=False):
    inputs = Input(shape=input_shape)

    if using_white_norm:
        wn = Lambda(white_norm, name="white_norm")(inputs)
        conv1 = Conv2D(32, 3, padding='valid', strides=1, use_bias=False, name="conv1")(wn)
    else:
        conv1 = Conv2D(32, 3, padding='valid', strides=1, use_bias=False, name='conv1')(inputs)

    block1 = BRA(conv1)
    block1 = SE_BLOCK(block1, using_SE)

    conv2 = Conv2D(32, 3, padding='valid', strides=1, name='conv2')(block1)
    block2 = BRA(conv2)
    block2 = SE_BLOCK(block2, using_SE)

    conv3 = Conv2D(32, 3, padding='valid', strides=1, name='conv3')(block2)
    block3 = BRA(conv3)
    block3 = SE_BLOCK(block3, using_SE)

    conv4 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv4")(block3)  # 9248
    block4 = BN_ReLU(conv4, name="BN_ReLu")  # 128
    block4 = SE_BLOCK(block4, using_SE)

    conv5 = Conv2D(32, (1, 1), padding="valid", strides=1, name="conv5")(block4)  # 1024 + 32
    conv5 = SE_BLOCK(conv5, using_SE)

    flat_conv = Flatten()(conv5)

    model = Model(inputs=[inputs], outputs=[flat_conv])
    return model


def create_model_all(num_classes=12, input_shape=(64, 64, 3), using_white_norm=True, using_SE=True):
    base_model = base(input_shape=input_shape, using_white_norm=using_white_norm, using_SE=using_SE)

    x1 = Input(shape=input_shape)
    x2 = Input(shape=input_shape)
    x3 = Input(shape=input_shape)

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)

    cfeat = Concatenate(axis=-1)([y1, y2, y3])
    bulk_feat = Dense(num_classes, use_bias=True, activity_regularizer=regularizers.l1(0.), activation="softmax", name="cate")(
        cfeat)

    age = Dense(1, name="reg")(bulk_feat)
    outputs_gen = Dense(2, activation='softmax', name='gen')(cfeat)
    return Model(inputs=[x1, x2, x3], outputs=[bulk_feat, age, outputs_gen])