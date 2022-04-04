import tensorflow as tf
import os

from model import base
from utils import load_data, create_data_gen_xy, visual_results
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == '__main__':
    # PARAMS
    batch_size = 256
    epochs = 5
    save_file_path = ".\\save\\base2\\"

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
    model.summary()

    # LOAD MODEL ------------------------------------------------------
    if os.path.exists(save_file_path):
        print("Model {} loaded.".format(save_file_path))
        model.load_weights(save_file_path)

    else:
        print("Train new model.")

    # COMPILE ---------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=['categorical_crossentropy'],
        metrics=['categorical_accuracy']
    )

    # # TEST ZONE -------------------------------------------------------
    # img_ori = tf.random.uniform(shape=[1, 160, 160, 3])
    # result = model([img_ori])
    # print(result)

    # CALLBACKS -------------------------------------------------------
    callbacks = [
        ModelCheckpoint(save_file_path, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                        save_weights_only=True,
                        mode='max'),
    ]

    # FIT -------------------------------------------------------------
    history = model.fit(train_gen,
                        steps_per_epoch=len(X_train) / batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=len(X_valid) / batch_size)

    evaluate = model.evaluate(test_gen,
                              steps=len(X_test) / batch_size)

    # VISUAL
    for test_data in test_gen:
        results = model.predict(test_data[0])
        visual_results(save_file_path, test_data, results)
        break
