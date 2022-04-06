import tensorflow as tf
import os

from model import base
from utils import load_data, create_data_gen_xy, visual_results, visual_history
from fast_ml.model_development import train_valid_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

'''
SOURCE:
    'wiki': 'wiki_crop',
    'imdb': 'imdb',
    'utk': 'UTKFace',
    'cacd': 'cacd',
    'facial': 'Facial',
    'asia': 'All-Age-Faces',
    'afad': 'AFAD-Full'
    
MODEL DICT:
    cate
    reg
    all   
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
    epochs = 50
    mode = "all"
    source = 'afad'
    ver = 3
    save_file_path = ".\\save\\base{}_{}\\".format(ver, mode)

    # LOAD DATA -------------------------------------------------------
    dataframe = load_data("D:\\Data", source=source)

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
    # cate: categories | reg: regression | all: both cate+reg
    train_gen = create_data_gen_xy(X_train, y_train, batch_size, mode=mode)
    test_gen = create_data_gen_xy(X_test, y_test, batch_size, mode=mode)
    valid_gen = create_data_gen_xy(X_valid, y_valid, batch_size, mode=mode)

    # # TEST ZONE -------------------------------------------------------
    # for i in train_gen:
    #     print(i)
    #     break

    # MODEL -----------------------------------------------------------
    if mode == 'reg':
        model = base.create_model_reg(input_shape=[160, 160, 3])
    elif mode == 'cate':
        model = base.create_model_cate(input_shape=[160, 160, 3])
    else:
        model = base.create_model_all(input_shape=[160, 160, 3])

    # LOAD MODEL ------------------------------------------------------
    if os.path.exists(save_file_path):
        print("Model {} loaded.".format(save_file_path))
        model.load_weights(save_file_path)

    else:
        print("Train new model.")

    # COMPILE ---------------------------------------------------------
    if mode == 'reg':
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['mae'],
            metrics={"reg": 'mae'}
        )
    elif mode == 'cate':
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['categorical_crossentropy'],
            metrics={"cate": 'categorical_accuracy'}
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['categorical_crossentropy', 'mae'],
            metrics={"cate": 'categorical_accuracy', "reg": 'mae'}
        )

    # # TEST ZONE -------------------------------------------------------
    # img_ori = tf.random.uniform(shape=[1, 160, 160, 3])
    # result = model([img_ori])
    # print(result)

    # CALLBACKS -------------------------------------------------------
    callbacks = [
        ModelCheckpoint(save_file_path, monitor='val_loss', verbose=1, save_best_only=True,
                        save_weights_only=True,
                        mode='min'),
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

    # VISUAL ----------------------------------------------------------
    for test_data in test_gen:
        results = model.predict(test_data[0])
        visual_results(save_file_path, test_data, results, mode=mode)
        break

    # VISUAL HISTORY --------------------------------------------------
    visual_history(save_file_path, history)