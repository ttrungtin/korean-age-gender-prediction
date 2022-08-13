import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

feather_dict = {
    'wiki': 'wiki_crop',
    'imdb': 'imdb',
    'utk': 'UTKFace',
    'cacd': 'cacd',
    'facial': 'Facial',
    'asia': 'All-Age-Faces',
    'afad': 'AFAD-Full',
    'lab': 'lab'
}

# age_dict = {
#     0: '0-10',
#     1: '10-20',
#     2: '20-30',
#     3: '30-40',
#     4: '40-50',
#     5: '50-60',
#     6: '60-70',
#     7: '70-80',
#     8: '80-90',
#     9: '90-100'
# }

age_dict = {
    0: '0-1',
    1: '1-3',
    2: '3-6',
    3: '6-9',
    4: '9-12',
    5: '12-15',
    6: '15-30',
    7: '30-45',
    8: '45-65',
    9: '65-85',
    10: '85-100'
}

two_point_model = {'c3ae_base', 'c3ae_alpha'}
three_box_model = {'cnn3', 'c3ae_base', 'c3ae_alpha'}


def convert_age(age, gap=10):
    min_age = age - gap
    max_age = age + gap

    # print(min_age, max_age)

    # ls_min = np.logspace(0, 0.99, num=gap)/10
    # ls_max = np.logspace(0.99, 0, num=gap)/10

    # ls_min = np.linspace(0, 0.99, num=gap)
    # ls_max = np.linspace(0.99, 0, num=gap)

    ls = np.linspace(0, 0.99, num=gap + 1)
    # ls = np.logspace(0, 0.99, num=gap + 1)/10
    ls_mirror = np.concatenate([ls[:-1], ls[::-1]])

    # print(ls_mirror)
    # print(ls_mirror.shape)

    if min_age <= 0:
        # a[0:age+1] = [0, ls_min[-min_age:]]
        # a[age-1:max_age] = ls_max[-1:]
        # a[0:len(ls[-min_age:])] = ls[-min_age-1:]

        ls_filter = ls_mirror[-min_age:]
        output = np.concatenate([ls_filter, np.zeros(shape=100 - len(ls_filter))])

    elif max_age >= 100:
        # a[min_age+1:age+1] = ls_min
        # a[age:101] = ls_max[:max_age-100+1]

        ls_filter = ls_mirror[:100 - min_age]
        output = np.concatenate([np.zeros(shape=100 - len(ls_filter)), ls_filter])

    else:
        output = np.concatenate([np.zeros(shape=min_age), ls_mirror, np.zeros(shape=100 - max_age - 1)])
        # print(output.shape)

    # plt.plot(output)
    # print(output[age - 2], output[age - 1], output[age], output[age + 1], output[age + 2])
    return output


def load_data(data_path, source):
    source = source.split('|')
    init_df = True
    dataframe = pd.DataFrame()

    for s in source:
        feather_folder = feather_dict[s]
        feather_path = os.path.join(data_path, feather_folder)

        all_feather_files = os.listdir(feather_path)

        for idx, feather in enumerate(all_feather_files):

            feather_dir = os.path.join(feather_path, feather)
            loaded_data = pd.read_feather(feather_dir)

            # For first dataframe loaded
            if init_df:
                dataframe = loaded_data
                init_df = False
            else:
                dataframe = pd.concat([dataframe, loaded_data], ignore_index=True, sort=False)

            print("Source: {} | {}: ".format(s, idx), loaded_data.shape)
    print("Total: ", dataframe.shape)

    # Filter
    dataframe = dataframe[(dataframe['age'] > 0) & (dataframe['age'] < 101)]
    dataframe = dataframe.dropna()

    return dataframe


def image_decode(row, model_type='cnn'):
    row = row[1]

    def decode(img):
        img = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = img / 255.0
        return img

    if model_type in three_box_model:
        img_list = [row.img_box1, row.img_box2, row.img_box3]
        img_list = list(map(decode, img_list))
    else:
        img_list = [row.img_ori]
        img_list = list(map(decode, img_list))

    return img_list


def age_categories_convert(age_label, num_classes, soft_label):
    # Init
    age_convert = np.zeros_like(age_label)

    # # Age group
    # age_convert[age_label < 10] = 0
    # age_convert[(age_label >= 10) & (age_label < 20)] = 1
    # age_convert[(age_label >= 20) & (age_label < 30)] = 2
    # age_convert[(age_label >= 30) & (age_label < 40)] = 3
    # age_convert[(age_label >= 40) & (age_label < 50)] = 4
    # age_convert[(age_label >= 50) & (age_label < 60)] = 5
    # age_convert[(age_label >= 60) & (age_label < 70)] = 6
    # age_convert[(age_label >= 70) & (age_label < 80)] = 7
    # age_convert[(age_label >= 80) & (age_label < 90)] = 8
    # age_convert[(age_label >= 90) & (age_label < 100)] = 9

    # Age group
    age_convert[age_label <= 1] = 0
    age_convert[(age_label > 1) & (age_label <= 3)] = 1
    age_convert[(age_label > 3) & (age_label <= 6)] = 2
    age_convert[(age_label > 6) & (age_label <= 9)] = 3
    age_convert[(age_label > 9) & (age_label <= 12)] = 4
    age_convert[(age_label > 12) & (age_label <= 15)] = 5
    age_convert[(age_label > 15) & (age_label <= 30)] = 6
    age_convert[(age_label > 30) & (age_label <= 45)] = 7
    age_convert[(age_label > 45) & (age_label <= 65)] = 8
    age_convert[(age_label > 65) & (age_label <= 85)] = 9
    age_convert[(age_label > 85) & (age_label <= 100)] = 10

    # To one-hot vector
    age_convert = to_categorical(age_convert, num_classes=num_classes)

    # Soft label
    if soft_label:
        for i in age_convert:
            max_idx = i.argmax()
            if (max_idx >= 1) and (max_idx < len(i) - 1):
                i[max_idx - 1], i[max_idx], i[max_idx + 1] = 0.1, 0.8, 0.1
            elif max_idx == 0:
                i[max_idx], i[max_idx + 1] = 0.8, 0.2
            elif max_idx == len(i) - 1:
                i[max_idx - 1], i[max_idx] = 0.2, 0.8

    # Expand dim
    age_convert = np.expand_dims(age_convert, axis=2)

    return age_convert


def smooth_label(labels, cls_num, on_value=0.99, epsilon=1e-8):
    from keras.utils.np_utils import to_categorical
    if not (0 <= on_value <= 1 and cls_num > 1):
        raise Exception("fatal params on smooth")

    if 0.99 in labels:
        labels = np.where(labels==0.99, 1, 0)

    onehot = to_categorical(labels, 2)
    return np.where(onehot > 0, on_value, (1 - on_value) / (cls_num - 1 + epsilon))


def two_point(age_label, num_classes, interval=10, elips=0.000001):
    def age_split(age):
        embed = [0 for x in range(0, num_classes)]
        right_prob = age % interval * 1.0 / interval
        left_prob = 1 - right_prob
        idx = age // interval
        if left_prob:
            embed[idx] = left_prob
        if right_prob and idx + 1 < num_classes:
            embed[idx + 1] = right_prob
        return embed

    return np.array(age_split(age_label))


# cate: categories | reg: regression | all: both cate+reg
def create_data_gen(*args, **kwds):
    mode = kwds['mode']

    num_classes = kwds['num_classes']
    soft_label = kwds['soft_label']
    batch_size = kwds['batch_size']
    include_gen = kwds['include_gen']
    model_type = kwds['model_type']
    gap = kwds['gap']

    if len(args) == 1:  # <<<< for DF
        dataframe = args[0].reset_index(drop=True)
        all_nums = len(dataframe)
    elif len(args) == 2:  # <<<<< non DF: X,y
        X = args[0].reset_index(drop=True)
        y = args[1].reset_index(drop=True)
        all_nums = len(X)

    while True:
        idxs = np.random.permutation(all_nums)
        start = 0

        while start + batch_size < all_nums:
            sub_idxs = list(idxs[start:start + batch_size])

            if len(args) == 1:
                sub_df = dataframe.iloc[sub_idxs]
                age_label = sub_df.age.to_numpy()

                if include_gen:
                    gen_label = sub_df.gen.to_numpy()

            elif len(args) == 2:
                sub_X_df = X.iloc[sub_idxs]
                sub_y_df = y.iloc[sub_idxs]
                age_label = sub_y_df.age.to_numpy()

                if include_gen:
                    gen_label = sub_y_df.gen.to_numpy()

            y_label = []
            if mode == 'cate':
                age_cate = age_categories_convert(age_label, num_classes=num_classes, soft_label=soft_label)
                y_label = [age_cate]

            elif mode == 'reg':
                age_reg = np.expand_dims(age_label, axis=1)
                y_label = [age_reg]

            elif mode == 'all':
                # if model_type in two_point_model:
                #     # age_cate = np.array([two_point(int(x), num_classes) for x in age_label])
                #     age_cate = np.array([convert_age(int(x), gap=gap) for x in age_label])
                #     age_reg = np.expand_dims(age_label, axis=1)
                #     y_label = [age_cate, age_reg]
                # else:
                #     age_cate = age_categories_convert(age_label, num_classes=num_classes, soft_label=soft_label)
                #     age_reg = np.expand_dims(age_label, axis=1)
                #     y_label = [age_cate, age_reg]

                age_cate = np.array([convert_age(int(x), gap=gap) for x in age_label])
                age_reg = np.expand_dims(age_label, axis=1)

                if include_gen:
                    gen_cate = smooth_label(gen_label, 2, 0.99)
                    y_label = [age_cate, age_reg, gen_cate]
                    # y_label = [age_reg, gen_cate]
                else:
                    y_label = [age_cate, age_reg]



            # x data
            if len(args) == 1:
                x_data = np.array([image_decode(row, model_type=model_type) for row in sub_df.iterrows()])
            elif len(args) == 2:
                x_data = np.array([image_decode(row, model_type=model_type) for row in sub_X_df.iterrows()])

            # out
            if model_type in three_box_model:
                yield [x_data[:, 0], x_data[:, 1], x_data[:, 2]], y_label
            else:
                yield [x_data.squeeze()], y_label

            # continue
            start += batch_size


def plot_results(save_file_path, images, labels, predicts, name, mode='cate', loss_mode=None):
    fig, ax = plt.subplots(3, 2, figsize=[10, 20])

    for i in range(3):
        for j in range(2):
            idx = i * 2 + j
            # img = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
            img = images[idx]
            ax[i, j].imshow(img)

            if mode == "all":
                if loss_mode == 'fl':
                    ax[i, j].title.set_text(
                        'label: {:0.2f}|{:0.2f} - pred: {:0.2f}|{:0.2f}'.format(
                            labels[0][idx], labels[1][idx],
                            predicts[0][idx], predicts[1][idx]))
                else:
                    ax[i, j].title.set_text(
                        'label: {:s}|{:0.2f} - pred: {:s}|{:0.2f}'.format(
                            labels[0][idx], labels[1][idx],
                            predicts[0][idx], predicts[1][idx]))
            elif mode == 'cate':
                ax[i, j].title.set_text(
                    'label: {} - pred: {}'.format(age_dict[labels[idx]], age_dict[predicts[idx]]))
            else:
                ax[i, j].title.set_text(
                    'label: {:0.2f} - pred: {:0.2f}'.format(labels[idx], predicts[idx]))

    fig.savefig('{}result_{}.jpg'.format(save_file_path, name))


def visual_results(save_file_path, test_data, results, name, mode='cate', loss_mode=None):
    images = test_data[0][0][:6]
    # images = test_data
    if mode == 'cate':
        labels = test_data[1][0][:6].squeeze().argmax(axis=1)

        predicts = np.argmax(results, axis=1)[:6]

        plot_results(save_file_path, images, labels, predicts, mode=mode, name=name)

    elif mode == 'reg':
        labels = test_data[1][0][:6].squeeze()

        predicts = results[:6].squeeze()

        plot_results(save_file_path, images, labels, predicts, mode=mode, name=name)

    elif mode == 'all':
        # labels_cate = test_data[1][0][:6].squeeze().argmax(axis=1)

        labels_reg = test_data[1][1][:6].squeeze()
        #
        labels_gen = test_data[1][2][:6].squeeze()
        labels_gen = labels_gen.argmax(axis=1)
        labels_gen = np.where(labels_gen == 0, 'Male', 'Female')

        # predicts_cate = np.argmax(results[0], axis=1)[:6]
        # predicts_reg = results[1][:6].squeeze()

        predicts_reg = results[1][:6].squeeze()

        predicts_gen = results[2][:6].squeeze()
        predicts_gen = predicts_gen.argmax(axis=1)
        predicts_gen = np.where(predicts_gen == 0, 'Male', 'Female')


        # labels_reg = test_data[1][0][:6].squeeze()
        # predicts_reg = results[0][:6].squeeze()
        #
        # labels_gen = test_data[1][1][:6].squeeze()
        # labels_gen = labels_gen.argmax(axis=1)
        # labels_gen = np.where(labels_gen == 0, 'Male', 'Female')
        #
        # predicts_gen = results[1][:6].squeeze()
        # predicts_gen = predicts_gen.argmax(axis=1)
        # predicts_gen = np.where(predicts_gen == 0, 'Male', 'Female')


        plot_results(save_file_path, images, [labels_gen, labels_reg], [predicts_gen, predicts_reg], mode=mode,
                     loss_mode=loss_mode, name=name)

        # plot_results(save_file_path, images,
        #              [['Male', 'Female', 'Female', 'Female', 'Male', 'Male', ], [25, 25, 25, 25, 26, 36]],
        #              [predicts_gen, predicts_reg], mode=mode,
        #              loss_mode=loss_mode, name=name)


def visual_history(save_file_path, history, name):
    epoch = history.epoch
    history = history.history

    try:
        history.pop('lr', None)
    except:
        pass

    assert len(history) % 2 == 0, 'History format error.'

    half = int(len(history) / 2)
    fig, ax = plt.subplots(half, 1, figsize=[10, 20])

    for idx, key in enumerate(history):
        if idx >= half:
            idx -= half
        ax[idx].plot(epoch, history[key], label=key)
        ax[idx].legend()

    fig.savefig('{}chart_{}.jpg'.format(save_file_path, name))


def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    # copy from https://github.com/maozezhong/focal_loss_multi_class/blob/master/focal_loss.py
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops

        # print(classes_num)
        # print(type(classes_num))

        # 1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor, zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.math.log(tf.clip_by_value(prediction_tensor, 1e-6, 1.0))

        # 2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [(total_num / ff if ff != 0 else 0.0) for ff in classes_num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff / sum_ for ff in classes_w_t1]  # scale
        # print(classes_w_t2)
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        # 3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        # 4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1 - e) * balanced_fl + e * K.categorical_crossentropy(
            K.ones_like(prediction_tensor) / nb_classes, prediction_tensor)
        return fianal_loss

    return focal_loss_fixed

# # cate: categories | reg: regression | all: both cate+reg
# def create_data_gen_df(dataframe, batch_size, mode='cate'):
#     dataframe = dataframe.reset_index(drop=True)
#     all_nums = len(dataframe)
#     while True:
#         idxs = np.random.permutation(all_nums)
#         start = 0
#
#         while start + batch_size < all_nums:
#             sub_idxs = list(idxs[start:start + batch_size])
#             sub_df = dataframe.iloc[sub_idxs]
#
#             # y label
#             age_label = sub_df.age.to_numpy()
#             y_label = []
#
#             if mode == 'cate':
#                 age_label = age_categories_convert(age_label)
#                 age_label = to_categorical(age_label, num_classes=num_classes)
#                 age_label = np.expand_dims(age_label, axis=2)
#                 y_label = [age_label]
#
#             elif mode == 'reg':
#                 age_label = np.expand_dims(age_label, axis=1)
#                 y_label = [age_label]
#
#             elif mode == 'all':
#                 age_cate = age_categories_convert(age_label)
#                 age_cate = to_categorical(age_cate, num_classes=num_classes)
#                 age_cate = np.expand_dims(age_cate, axis=2)
#
#                 age_reg = np.expand_dims(age_label, axis=1)
#
#                 y_label = [age_cate, age_reg]
#
#             # x data
#             x_data = np.array([image_decode(row) for row in sub_df.iterrows()])
#
#             # out
#             yield [x_data.squeeze()], y_label
#
#             # continue
#             start += batch_size


# # cate: categories | reg: regression | all: both cate+reg
# def create_data_gen_xy(X, y, batch_size, mode='cate'):
#     X = X.reset_index(drop=True)
#     y = y.reset_index(drop=True)
#     all_nums = len(X)
#     while True:
#         idxs = np.random.permutation(all_nums)
#         start = 0
#
#         while start + batch_size < all_nums:
#             sub_idxs = list(idxs[start:start + batch_size])
#             sub_X_df = X.iloc[sub_idxs]
#             sub_y_df = y.iloc[sub_idxs]
#
#             # y label
#             age_label = sub_y_df.age.to_numpy()
#             y_label = []
#
#             if mode == 'cate':
#                 age_label = age_categories_convert(age_label)
#                 age_label = to_categorical(age_label, num_classes=num_classes)
#                 age_label = np.expand_dims(age_label, axis=2)
#                 y_label = [age_label]
#
#             elif mode == 'reg':
#                 age_label = np.expand_dims(age_label, axis=1)
#                 y_label = [age_label]
#
#             elif mode == 'all':
#                 age_cate = age_categories_convert(age_label)
#                 age_cate = to_categorical(age_cate, num_classes=num_classes)
#                 age_cate = np.expand_dims(age_cate, axis=2)
#
#                 age_reg = np.expand_dims(age_label, axis=1)
#
#                 y_label = [age_cate, age_reg]
#
#             # x data
#             x_data = np.array([image_decode(row) for row in sub_X_df.iterrows()])
#
#             # out
#             yield [x_data.squeeze()], y_label
#
#             # continue
#             start += batch_size

# def plot_results(save_file_path, images, labels, predicts, name, mode='cate', loss_mode=None):
#     fig, ax = plt.subplots(3, 2, figsize=[10, 20])
#
#     for i in range(3):
#         for j in range(2):
#             idx = i * 2 + j
#             ax[i, j].imshow(images[idx])
#
#             if mode == "all":
#                 if loss_mode == 'fl':
#                     ax[i, j].title.set_text(
#                         'label: {:0.2f}|{:0.2f} - pred: {:0.2f}|{:0.2f}'.format(
#                             labels[0][idx], labels[1][idx],
#                             predicts[0][idx], predicts[1][idx]))
#                 else:
#                     ax[i, j].title.set_text(
#                         'label: {:s}|{:0.2f} - pred: {:s}|{:0.2f}'.format(
#                             labels[0][idx], labels[1][idx],
#                             predicts[0][idx], predicts[1][idx]))
#             elif mode == 'cate':
#                 ax[i, j].title.set_text(
#                     'label: {} - pred: {}'.format(age_dict[labels[idx]], age_dict[predicts[idx]]))
#             else:
#                 ax[i, j].title.set_text(
#                     'label: {:0.2f} - pred: {:0.2f}'.format(labels[idx], predicts[idx]))
#
#     fig.savefig('{}result_{}.jpg'.format(save_file_path, name))

def visual_2(results):

    # per folder
    for result in results:

        if len(result) == 0:
            continue

        if len(result) % 2 == 0:
            len_res = len(result)
        else:
            len_res = len(result)+1

        row = int(len_res / 2)
        col = 2

        # print(result)

        fig, ax = plt.subplots(row, col, figsize=[10, 20])


        for i in range(row):
            for j in range(col):
                try:
                    idx = i * 2 + j
                    # print(idx, i, j)
                    ax[i, j].imshow(result[idx]['img'])
                    ax[i, j].title.set_text('reg: {} | age: {} | gen: {}'.format(
                        result[idx]['age_reg'], result[idx]['age'], result[idx]['gen'])
                    )
                except:
                    pass

        fig.suptitle(result[0]['object'])
        fig.savefig('test_res\\result_{}.jpg'.format(result[0]['object']))

    return 0

def visual_3(results):

    if len(results) % 2 == 0:
        len_res = len(results)
    else:
        len_res = len(results)+1

    row = int(len_res / 2)
    col = 2


    splitA = 10
    splitB = 10
    splitC = 10
    splitD = 8

    rowA = rowB = rowC = 10
    colA = colB = colC = 2

    rowD = 4
    colD = 2

    fig, ax = plt.subplots(rowA, colA, figsize=[10, 20])
    for i in range(row):
        for j in range(col):
            try:
                idx = i * 2 + j
                # print(idx, i, j)
                ax[i, j].imshow(results[idx]['img'])
                ax[i, j].title.set_text('reg: {} | age: {} | gen: {} | name: {}'.format(
                    results[idx]['age_reg'], results[idx]['age'], results[idx]['gen'], results[idx][object])
                )
            except:
                pass
    fig.savefig('test_res\\result1.jpg')

    fig, ax = plt.subplots(rowB, colB, figsize=[10, 20])
    for i in range(row):
        for j in range(col):
            try:
                idx = i * 2 + j + 10
                # print(idx, i, j)
                ax[i, j].imshow(results[idx]['img'])
                ax[i, j].title.set_text('reg: {} | age: {} | gen: {} | name: {}'.format(
                    results[idx]['age_reg'], results[idx]['age'], results[idx]['gen'], results[idx][object])
                )
            except:
                pass
    fig.savefig('test_res\\result2.jpg')

    fig, ax = plt.subplots(rowC, colC, figsize=[10, 20])
    for i in range(row):
        for j in range(col):
            try:
                idx = i * 2 + j + 20
                # print(idx, i, j)
                ax[i, j].imshow(results[idx]['img'])
                ax[i, j].title.set_text('reg: {} | age: {} | gen: {} | name: {}'.format(
                    results[idx]['age_reg'], results[idx]['age'], results[idx]['gen'], results[idx][object])
                )
            except:
                pass
    fig.savefig('test_res\\result3.jpg')

    fig, ax = plt.subplots(rowD, colD, figsize=[10, 20])
    for i in range(row):
        for j in range(col):
            try:
                idx = i * 2 + j + 30
                # print(idx, i, j)
                ax[i, j].imshow(results[idx]['img'])
                ax[i, j].title.set_text('reg: {} | age: {} | gen: {} | name: {}'.format(
                    results[idx]['age_reg'], results[idx]['age'], results[idx]['gen'], results[idx][object])
                )
            except:
                pass
    fig.savefig('test_res\\result4.jpg')

    # for result in results:
    #
    #     # if len(result) == 0:
    #     #     continue
    #     #
    #     # if len(result) % 2 == 0:
    #     #     len_res = len(result)
    #     # else:
    #     #     len_res = len(result)+1
    #
    #     # row = 1
    #     # col = 1
    #
    #     # print(result)
    #
    #
    #     ax.imshow(result['img'])
    #     ax.title.set_text('reg: {} | age: {} | label: {} | gen: {}'.format(
    #         result['age_reg'], result['age'], result['age_label'], result['gen'])
    #     )
    #     fig.suptitle(result['object'])
    #     fig.savefig('test_res\\result_{}.jpg'.format(result['object']))

        # for i in range(row):
        #     for j in range(col):
        #         try:
        #             idx = i * 2 + j
        #             # print(idx, i, j)
        #             ax[i, j].imshow(result[idx]['img'])
        #             ax[i, j].title.set_text('reg: {} | age: {} | gen: {}'.format(
        #                 result[idx]['age_reg'], result[idx]['age'], result[idx]['gen'])
        #             )
        #         except:
        #             pass
        #
        # fig.suptitle(result[0]['object'])
        # fig.savefig('test_res\\result_{}.jpg'.format(result[0]['object']))

    return 0


def visual_4(results):
    print(len(results))
    row = 3
    col = 1
    num_fig = int(np.round((len(results) / 3)))
    idx = 0

    for i in range(num_fig):
        fig, ax = plt.subplots(row, col, figsize=[8.27, 11.69])
        for j in range(row):
            for k in range(col):
                try:
                    ax[j].imshow(results[idx]['img'])
                    ax[j].title.set_text('reg: {:0.2f} | age_pred: {} | gen: {} | name: {}'.format(
                        results[idx]['age_reg'][0], results[idx]['age'][0], results[idx]['gen'], results[idx]['object'])
                    )
                    print(results[idx]['object'])
                    idx+=1
                except:
                    print('here')
                    pass
        fig.savefig('test_res\\result_{}.jpg'.format(idx))






