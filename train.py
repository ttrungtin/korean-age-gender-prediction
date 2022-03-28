import tensorflow as tf

from model import base
from utils import load_data, create_data_gen_xy
from fast_ml.model_development import train_valid_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

'''
SOURCE:
    'wiki': 'wiki_crop',
    'imdb': 'imdb_crop_{}.feather',
    'utk': 'UTKFace',
    'cacd': 'cacd.feather',
    'facial': 'facial-age.feather',
    'asia': 'All-Age-Faces Dataset',
    'afad': 'AFAD-Full'
'''

SEED = 22
tf.random.set_seed(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':
    # PARAMS
    batch_size = 256
    epochs = 100

    # LOAD DATA -------------------------------------------------------
    dataframe = load_data("D:\\Data", source='afad')

    # SPLIT -----------------------------------------------------------
    train_rate = 0.8
    valid_rate = 0.1
    test_rate = 0.1

    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(dataframe,
                                                                                target=['age'],
                                                                                train_size=train_rate,
                                                                                valid_size=valid_rate,
                                                                                test_size=test_rate,
                                                                                random_state=SEED)

    # DATA GEN --------------------------------------------------------
    train_gen = create_data_gen_xy(X_train, y_train, batch_size)
    test_gen = create_data_gen_xy(X_test, y_test, batch_size)
    valid_gen = create_data_gen_xy(X_valid, y_valid, batch_size)

    # MODEL -----------------------------------------------------------
    model = base.create_model(input_shape=[160, 160, 3])

    # COMPILE ---------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=['mae'],
        metrics={'age': 'mse'}
    )

    # CALLBACKS -------------------------------------------------------
    callbacks = [
        ModelCheckpoint(".\\save\\base\\", monitor='val_mse', verbose=1, save_best_only=True, save_weights_only=True,
                        mode='min'),
    ]

    # FIT -------------------------------------------------------------
    history = model.fit(train_gen,
                        steps_per_epoch=len(X_train) / batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=len(X_valid) / batch_size)

    predict = model.evaluate(test_gen,
                             steps=len(X_test) / batch_size)
    print(predict)
