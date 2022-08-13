import pickle

import numpy as np
import tensorflow as tf
import os
import cv2

from model import base, cnn, cnn2, cnn3, c3ae_alpha, c3ae_base
from utils import load_data, visual_results, visual_history, create_data_gen, focal_loss, visual_2, visual_3, visual_4
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
    'lab': 'lab
    
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
    "c3ae_base": c3ae_base,
    "c3ae_alpha": c3ae_alpha
}

'''
    base4.0: 
        - default mode = all
'''

if __name__ == '__main__':
    # PARAMS ----------------------------------------------------------
    # ------------------------------
    batch_size = 32
    epochs = 100
    input_shape = [160, 160, 3]
    learning_rate = 0.001
    gap = 3

    # ------------------------------
    model_type = 'cnn2'
    model_various = 'KLloss_gap{}'.format(gap)
    ver = "release5"
    save_file_path = ".\\save\\{}_{}_{}_afad_asia\\".format(ver, model_various, model_type)
    log_path = ".\\logs\\log_{}_{}_{}\\".format(ver, model_type, model_various)
    print(save_file_path)
    # ------------------------------
    use_valid = False
    num_classes = 100
    soft_label = True  # soft categorical label

    # ------------------------------
    # warm_up = False
    #
    data_path = "D:\\Dataset\\Feather"
    # if warm_up:  # <<<<<<<<<<<<<<<<<<<<<<<<<< non-asian training, skip gender
    #     include_gen = True
    #     source = 'imdb|wiki'
    # else:  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< asian training, with gender
    #     include_gen = True
    #     source = 'afad'
    # # ------------------------------

    include_gen = True
    source = 'lab'

    image_net_pre_train = False
    image_net_num_classes = 10450
    pre_train_save_file_path = ".\\save\\cnn_base_1_all_imgnet\\"

    # LOAD DATA -------------------------------------------------------
    training_df = load_data(data_path, source=source)

    # # SPLIT -----------------------------------------------------------
    # if use_valid:
    #     train_rate, valid_rate, test_rate = [0.8, 0.1, 0.1]
    #
    #     X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(training_df,
    #                                                                                 target=['age'],
    #                                                                                 train_size=train_rate,
    #                                                                                 valid_size=valid_rate,
    #                                                                                 test_size=test_rate,
    #                                                                                 random_state=SEED)
    #     print("Train: {} | Test: {} | Valid: {}".format(y_train.shape, y_test.shape, y_valid.shape))
    #
    # else:
    #     train_rate, test_rate = [0.8, 0.2]
    #     Xy_train, Xy_test = train_test_split(training_df,
    #                                          train_size=train_rate,
    #                                          test_size=test_rate,
    #                                          shuffle=True)
    #     print("Train: {} | Test: {}".format(Xy_train.shape, Xy_test.shape))

    # # DATA GEN --------------------------------------------------------
    # if use_valid:
    #     train_gen = create_data_gen(X_train, y_train, batch_size=batch_size, mode="all", num_classes=num_classes,
    #                                 soft_label=soft_label, model_type=model_type)
    #     test_gen = create_data_gen(X_test, y_test, batch_size=batch_size, mode="all", num_classes=num_classes,
    #                                soft_label=soft_label, model_type=model_type)
    #     valid_gen = create_data_gen(X_valid, y_valid, batch_size=batch_size, mode="all", num_classes=num_classes,
    #                                 soft_label=soft_label, model_type=model_type)
    #
    # else:
    #     train_gen = create_data_gen(Xy_train, batch_size=batch_size, mode="all", num_classes=num_classes,
    #                                 soft_label=soft_label, model_type=model_type, include_gen=include_gen, gap=gap)
    #     test_gen = create_data_gen(Xy_test, batch_size=batch_size, mode="all", num_classes=num_classes,
    #                                soft_label=soft_label, model_type=model_type, include_gen=include_gen, gap=gap)

    train_gen = create_data_gen(training_df, batch_size=batch_size, mode="all", num_classes=num_classes,
                                    soft_label=soft_label, model_type=model_type, include_gen=include_gen, gap=gap)

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

    if include_gen:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={'cate': 'kl_divergence', 'reg': 'mae', "gen": "categorical_crossentropy"},
            metrics={"cate": 'kullback_leibler_divergence', "reg": 'mae', "gen": "accuracy"},
            # loss_weights = [30, 1, 10],
        )

        # model.compile(
        #     optimizer=Adam(learning_rate=learning_rate),
        #     loss={'reg': 'mae', "gen": "categorical_crossentropy"},
        #     metrics={"reg": 'mae', "gen": "accuracy"},
        # )

        # model.compile(
        #     optimizer=Adam(learning_rate=learning_rate),
        #     loss={'cate': 'categorical_crossentropy', 'reg': 'mae', "gen": "categorical_crossentropy"},
        #     metrics={"cate": 'categorical_accuracy', "reg": 'mae', "gen": "accuracy"}
        # )
    else:
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
        # TensorBoard(log_dir=log_path, write_images=True, update_freq='epoch'),
        ReduceLROnPlateau(monitor='val_reg_mae', factor=0.1, patience=10, min_lr=0.00001)
    ]

    # evaluate = model.evaluate(train_gen, steps=len(training_df) / batch_size)
    # re = model.predict(train_gen, steps=len(training_df) / batch_size)


    t = os.listdir("test1\\face_data\\face_data")
    d = "test1\\face_data\\face_data"

    results = []


    for folder in t:
        obj_res = []
        # print(os.path.join(d, folder))
        imgs_list = os.path.join(d, folder)
        # print(imgs_list)
        for imgdir in os.listdir(imgs_list):
            idir = os.path.join(imgs_list, imgdir)
            print(idir)
            img = cv2.imread(idir)
            img = cv2.resize(img, (160, 160))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            res = model.predict(img, verbose=0)

            age = np.round(res[1][0])
            age_reg = res[1][0]
            gen = res[2][0]
            gen = gen.argmax()
            gen = np.where(gen == 0, 'M', 'F')

            info = folder.split("_")
            age_label = info[1]
            gen_label = "M" if info[-1] == "M" else "F"

            if gen_label != gen:
                continue

            re_dict = {
                "age": age,
                "age_reg": age_reg,
                "gen": gen,
                "img": img.squeeze(),
                "object": folder,
                "age_label": age_label,
                "gen_label": gen_label
            }

            obj_res.append(re_dict)

            # print(age, gen)
            # break

        # min error
        min_error = 10000
        for o in obj_res:
            if np.abs(int(o['age_label'])-o['age_reg']) < min_error:
                min_error = np.abs(int(o['age_label'])-o['age_reg'])
                pick = o
        results.append(pick)

        # results.append(obj_res)
        # break

    # visual_2(results)
    visual_4(results)
    # visual_3(results)






    # img = cv2.imread('tin.jpg')
    # img = cv2.resize(img, (160, 160))
    # img = img / 255.0
    # img = np.expand_dims(img, axis=0)
    #
    # res = model.predict(img)
    # print(res)

    # t = os.listdir("test1\\face_data\\face_data")
    # imgs = []
    # print(t)
    # for i in t:
    #     d = "test\\" + i
    #     img = cv2.imread(d)
    #     img = cv2.resize(img, (160, 160))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = img / 255.0
    #     # print(img.shape)
    #     imgs.append(img)
    #
    # # print(len(imgs))
    #
    # np_imgs = np.zeros(shape=[6, 160, 160, 3])
    # np_imgs[0] = imgs[0]
    # np_imgs[1] = imgs[1]
    # np_imgs[2] = imgs[2]
    # np_imgs[3] = imgs[3]
    # np_imgs[4] = imgs[4]
    # np_imgs[5] = imgs[5]

    # results = model.predict(np_imgs)
    # print(results)
    #
    # visual_results("", np_imgs, results, mode="all", name=model_various)

    # # FIT -------------------------------------------------------------
    # if use_valid:
    #     history = model.fit(train_gen,
    #                         steps_per_epoch=len(X_train) / batch_size,
    #                         epochs=epochs,
    #                         callbacks=callbacks,
    #                         validation_data=valid_gen,
    #                         validation_steps=len(X_valid) / batch_size)
    #
    #     evaluate = model.evaluate(test_gen,
    #                               steps=len(X_test) / batch_size)
    #
    # else:
    #     history = model.fit(train_gen,
    #                         steps_per_epoch=len(Xy_train) / batch_size,
    #                         epochs=epochs,
    #                         callbacks=callbacks,
    #                         validation_data=test_gen,
    #                         validation_steps=len(Xy_test) / batch_size)
    #
    #     evaluate = model.evaluate(test_gen,
    #                               steps=len(Xy_test) / batch_size)
    #
    # # VISUAL ----------------------------------------------------------
    # for test_data in test_gen:
    #     results = model.predict(test_data[0])
    #     visual_results(save_file_path, test_data, results, mode="all", name=model_various)
    #     break
    #
    # # VISUAL HISTORY --------------------------------------------------
    # visual_history(save_file_path, history, name=model_various)
    #
    # with open('{}{}{}'.format(save_file_path, 'history',warm_up), 'wb') as f:
    #     pickle.dump(history.history, f)
