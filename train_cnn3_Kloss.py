import tensorflow as tf
import os

from model import base, cnn, cnn2, cnn3, c3ae_alpha, c3ae_base
from utils import load_data, visual_results, visual_history, create_data_gen, focal_loss
from fast_ml.model_development import train_valid_test_split, train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

'''
SOURCE:
    'wiki': 'wiki_crop', - CELEB
    'imdb': 'imdb', - CELEB
    'utk': 'UTKFace', 
    'cacd': 'cacd', - CELEB
    'facial': 'Facial',
    'asia': 'All-Age-Faces', - ASIAN 
    'afad': 'AFAD-Full' <<< for benchmark | ASIAN
    
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

model_dict = {
    "base": base,
    "cnn": cnn,
    "cnn2": cnn2,
    "cnn3": cnn3,
    "c3ae_base": c3ae_base
}

'''
    base4.0: 
        - default mode = all
'''

if __name__ == '__main__':
    # PARAMS ----------------------------------------------------------
    # ------------------------------
    batch_size = 512
    epochs = 100
    input_shape = [64, 64, 3]
    learning_rate = 0.001

    # ------------------------------
    model_type = 'cnn3'
    model_various = 'KLloss'
    ver = "base40"
    save_file_path = ".\\save\\{}_{}_{}\\".format(model_various, model_type, ver)
    log_path = ".\\logs\\log_{}_{}_{}\\".format(model_type, model_various, ver)

    # ------------------------------
    use_valid = False
    num_classes = 11
    soft_label = True  # soft categorical label

    # ------------------------------
    data_path = "D:\\Dataset\\Feather"
    source = 'imdb|wiki|cacd'

    # ------------------------------
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
        print("Train: {} | Test: {} | Valid: {}".format(y_train.shape, y_test.shape, y_valid.shape))

    else:
        train_rate, test_rate = [0.8, 0.2]
        Xy_train, Xy_test = train_test_split(training_df,
                                             train_size=train_rate,
                                             test_size=test_rate,
                                             shuffle=True)
        print("Train: {} | Test: {}".format(Xy_train.shape, Xy_test.shape))

    # DATA GEN --------------------------------------------------------
    if use_valid:
        train_gen = create_data_gen(X_train, y_train, batch_size=batch_size, mode="all", num_classes=num_classes,
                                    soft_label=soft_label, model_type=model_type)
        test_gen = create_data_gen(X_test, y_test, batch_size=batch_size, mode="all", num_classes=num_classes,
                                   soft_label=soft_label, model_type=model_type)
        valid_gen = create_data_gen(X_valid, y_valid, batch_size=batch_size, mode="all", num_classes=num_classes,
                                    soft_label=soft_label, model_type=model_type)

    else:
        train_gen = create_data_gen(Xy_train, batch_size=batch_size, mode="all", num_classes=num_classes,
                                    soft_label=soft_label, model_type=model_type)
        test_gen = create_data_gen(Xy_test, batch_size=batch_size, mode="all", num_classes=num_classes,
                                   soft_label=soft_label, model_type=model_type)

    # # TEST ZONE -------------------------------------------------------
    # for i in train_gen:
    #     print(i)
    #     break

    # MODEL -----------------------------------------------------------
    if image_net_pre_train:
        model = model_dict[model_type].create_model_all(input_shape=input_shape, num_classes=image_net_num_classes,
                                                        image_net_pre_train=True)
    else:
        if model_type == 'c3ae_base':
            model = model_dict[model_type].create_model_all(input_shape=input_shape, num_classes=num_classes)
        else:
            model = model_dict[model_type].create_model_all(input_shape=input_shape, num_classes=num_classes)

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
        model = model_dict[model_type].create_top_all(model, num_classes=num_classes)

    # COMPILE ---------------------------------------------------------
    # model.compile(
    #     optimizer=Adam(learning_rate=learning_rate),
    #     loss={'cate': 'categorical_crossentropy', 'reg': 'mae'},
    #     metrics={"cate": 'categorical_accuracy', "reg": 'mae'}
    # )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'cate': 'kl_divergence', 'reg': 'mae'},
        metrics={"cate": 'kullback_leibler_divergence', "reg": 'mae'}
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
        visual_results(save_file_path, test_data, results, mode="all")
        break

    # VISUAL HISTORY --------------------------------------------------
    visual_history(save_file_path, history)
