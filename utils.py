import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
from tensorflow.keras.utils import to_categorical

feather_dict = {
    'wiki': 'wiki_crop',
    'imdb': 'imdb',
    'utk': 'UTKFace',
    'cacd': 'cacd',
    'facial': 'Facial',
    'asia': 'All-Age-Faces',
    'afad': 'AFAD-Full'
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


def image_decode(row):
    row = row[1]
    img_list = [row.img_ori]

    def decode(img):
        img = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img

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


# cate: categories | reg: regression | all: both cate+reg
def create_data_gen(*args, **kwds):
    mode = kwds['mode']
    num_classes = kwds['num_classes']
    soft_label = kwds['soft_label']
    batch_size = kwds['batch_size']

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
            elif len(args) == 2:
                sub_X_df = X.iloc[sub_idxs]
                sub_y_df = y.iloc[sub_idxs]
                age_label = sub_y_df.age.to_numpy()

            y_label = []
            if mode == 'cate':
                age_cate = age_categories_convert(age_label, num_classes=num_classes, soft_label=soft_label)
                y_label = [age_cate]

            elif mode == 'reg':
                age_reg = np.expand_dims(age_label, axis=1)
                y_label = [age_reg]

            elif mode == 'all':
                age_cate = age_categories_convert(age_label, num_classes=num_classes, soft_label=soft_label)
                age_reg = np.expand_dims(age_label, axis=1)
                y_label = [age_cate, age_reg]

            # x data
            if len(args) == 1:
                x_data = np.array([image_decode(row) for row in sub_df.iterrows()])
            elif len(args) == 2:
                x_data = np.array([image_decode(row) for row in sub_X_df.iterrows()])

            # out
            yield [x_data.squeeze()], y_label

            # continue
            start += batch_size


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


def plot_results(save_file_path, images, labels, predicts, mode='cate'):
    fig, ax = plt.subplots(3, 2, figsize=[10, 20])

    for i in range(3):
        for j in range(2):
            idx = i * 2 + j
            ax[i, j].imshow(images[idx])

            if mode == "all":
                ax[i, j].title.set_text(
                    'label: {:s}|{:0.2f} - pred: {:s}|{:0.2f}'.format(
                        age_dict[labels[0][idx]], labels[1][idx],
                        age_dict[predicts[0][idx]], predicts[1][idx]))
            elif mode == 'cate':
                ax[i, j].title.set_text(
                    'label: {} - pred: {}'.format(age_dict[labels[idx]], age_dict[predicts[idx]]))
            else:
                ax[i, j].title.set_text(
                    'label: {:0.2f} - pred: {:0.2f}'.format(labels[idx], predicts[idx]))

    fig.savefig('{}result.jpg'.format(save_file_path))


def visual_results(save_file_path, test_data, results, mode='cate'):
    images = test_data[0][0][:6]
    if mode == 'cate':
        labels = test_data[1][0][:6].squeeze().argmax(axis=1)

        predicts = np.argmax(results, axis=1)[:6]

        plot_results(save_file_path, images, labels, predicts, mode=mode)

    elif mode == 'reg':
        labels = test_data[1][0][:6].squeeze()

        predicts = results[:6].squeeze()

        plot_results(save_file_path, images, labels, predicts, mode=mode)

    elif mode == 'all':
        labels_cate = test_data[1][0][:6].squeeze().argmax(axis=1)
        labels_reg = test_data[1][1][:6].squeeze()

        predicts_cate = np.argmax(results[0], axis=1)[:6]
        predicts_reg = results[1][:6].squeeze()

        plot_results(save_file_path, images, [labels_cate, labels_reg], [predicts_cate, predicts_reg], mode=mode)


def visual_history(save_file_path, history):
    epoch = history.epoch
    history = history.history

    assert len(history) % 2 == 0, 'History format error.'

    half = int(len(history) / 2)
    fig, ax = plt.subplots(half, 1, figsize=[10, 20])

    for idx, key in enumerate(history):
        if idx >= half:
            idx -= half
        ax[idx].plot(epoch, history[key], label=key)
        ax[idx].legend()

    fig.savefig('{}chart.jpg'.format(save_file_path))
