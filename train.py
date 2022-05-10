import tensorflow as tf
import os

from model import base, cnn, cnn2
from utils import load_data, visual_results, visual_history, create_data_gen
from fast_ml.model_development import train_valid_test_split, train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

'''
SOURCE:
    'wiki': 'wiki_crop',
    'imdb': 'imdb',
    'utk': 'UTKFace',
    'cacd': 'cacd', <<< for benchmark
    'facial': 'Facial',
    'asia': 'All-Age-Faces',
    'afad': 'AFAD-Full' <<< for benchmark
    
MODEL DICT:
    cate
    reg
    all   
'''

SEED = 22
tf.random.set_seed(SEED)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == '__main__':
    # PARAMS ----------------------------------------------------------
    batch_size = 512
    epochs = 1
    input_shape = [160, 160, 3]

    mode = "all"
    ver = 1
    model_type = 'cnn2_base'
    save_file_path = ".\\save\\{}_{}_{}\\".format(model_type, ver, mode)
    log_path = ".\\log\\"

    use_valid = False
    num_classes = 11
    soft_label = True  # soft categorical label
    data_path = "D:\\Dataset\\Feather"
    source = 'imdb|wiki'

    image_net_pre_train = False
    image_net_num_classes = 10450
    pre_train_save_file_path = ".\\save\\cnn_base_1_all_imgnet\\"

    # LOAD DATA -------------------------------------------------------
    training_df = load_data(data_path, source=source)

    # SPLIT -----------------------------------------------------------
    if use_valid:
        train_rate, valid_rate, test_rate = [0.8, 0.1, 0.1]

        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(training_df,
                                                                                    target=['age'],
                                                                                    train_size=train_rate,
                                                                                    valid_size=valid_rate,
                                                                                    test_size=test_rate,
                                                                                    random_state=SEED)

    else:
        train_rate, test_rate = [0.8, 0.2]
        Xy_train, Xy_test = train_test_split(training_df,
                                             train_size=train_rate,
                                             test_size=test_rate,
                                             shuffle=True)

    # DATA GEN --------------------------------------------------------
    # cate: categories | reg: regression | all: both cate+reg
    if use_valid:
        train_gen = create_data_gen(X_train, y_train, batch_size=batch_size, mode=mode, num_classes=num_classes,
                                    soft_label=soft_label)
        test_gen = create_data_gen(X_test, y_test, batch_size=batch_size, mode=mode, num_classes=num_classes,
                                   soft_label=soft_label)
        valid_gen = create_data_gen(X_valid, y_valid, batch_size=batch_size, mode=mode, num_classes=num_classes,
                                    soft_label=soft_label)

    else:
        train_gen = create_data_gen(Xy_train, batch_size=batch_size, mode=mode, num_classes=num_classes,
                                    soft_label=soft_label)
        test_gen = create_data_gen(Xy_test, batch_size=batch_size, mode=mode, num_classes=num_classes,
                                   soft_label=soft_label)

    # # TEST ZONE -------------------------------------------------------
    # for i in train_gen:
    #     print(i)
    #     break

    # MODEL -----------------------------------------------------------
    if image_net_pre_train:
        model = cnn.create_model_all(input_shape=input_shape, num_classes=image_net_num_classes,
                                     image_net_pre_train=True)
    else:
        if mode == 'reg':
            model = base.create_model_reg(input_shape=input_shape)
        elif mode == 'cate':
            model = base.create_model_cate(input_shape=input_shape, num_classes=num_classes)
        else:
            model = cnn2.create_model_all(input_shape=input_shape, num_classes=num_classes)

    model.summary()

    # LOAD MODEL ------------------------------------------------------
    if image_net_pre_train:
        print("Model {} loaded.".format(pre_train_save_file_path))
        model.load_weights(pre_train_save_file_path)
    else:
        if os.path.exists(save_file_path):
            print("Model {} loaded.".format(save_file_path))
            model.load_weights(save_file_path)

        else:
            print("Train new model.")

    # CHANGE MODEL TOP
    if image_net_pre_train:
        model = cnn.create_top_all(model, num_classes=num_classes)

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
            loss={'cate': 'categorical_crossentropy'},
            metrics={"cate": 'categorical_accuracy'}
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'cate': 'categorical_crossentropy', 'reg': 'mae'},
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
        TensorBoard(log_dir=log_path, write_images=True, update_freq='epoch'),
        ReduceLROnPlateau(min_lr=0.00001)
    ]

    # FIT -------------------------------------------------------------
    if use_valid:
        history = model.fit(train_gen,
                            steps_per_epoch=len(X_train) / batch_size,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=valid_gen,
                            validation_steps=len(X_valid) / batch_size)

        evaluate = model.evaluate(test_gen,
                                  steps=len(X_test) / batch_size)

    else:
        history = model.fit(train_gen,
                            steps_per_epoch=len(Xy_train) / batch_size,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=test_gen,
                            validation_steps=len(Xy_test) / batch_size)

        evaluate = model.evaluate(test_gen,
                                  steps=len(Xy_test) / batch_size)

    # VISUAL ----------------------------------------------------------
    for test_data in test_gen:
        results = model.predict(test_data[0])
        visual_results(save_file_path, test_data, results, mode=mode)
        break

    # VISUAL HISTORY --------------------------------------------------
    visual_history(save_file_path, history)
