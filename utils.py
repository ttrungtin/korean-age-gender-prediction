import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
from tensorflow.keras.utils import to_categorical

feather_dict = {
    'wiki': 'wiki_crop',
    'imdb': 'imdb_crop_{}.feather',
    'utk': 'UTKFace',
    'cacd': 'cacd.feather',
    'facial': 'facial-age.feather',
    'asia': 'All-Age-Faces Dataset',
    'afad': 'AFAD-Full'
}

age_dict = {
    0: '0-10',
    1: '10-20',
    2: '20-30',
    3: '30-40',
    4: '40-50',
    5: '50-60',
    6: '60-70',
    7: '70-80',
    8: '80-90',
    9: '90-100'
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


def age_categories_convert(age_label):
    age_convert = np.zeros_like(age_label)

    age_convert[age_label < 10] = 0
    age_convert[(age_label >= 10) & (age_label < 20)] = 1
    age_convert[(age_label >= 20) & (age_label < 30)] = 2
    age_convert[(age_label >= 30) & (age_label < 40)] = 3
    age_convert[(age_label >= 40) & (age_label < 50)] = 4
    age_convert[(age_label >= 50) & (age_label < 60)] = 5
    age_convert[(age_label >= 60) & (age_label < 70)] = 6
    age_convert[(age_label >= 70) & (age_label < 80)] = 7
    age_convert[(age_label >= 80) & (age_label < 90)] = 8
    age_convert[(age_label >= 90) & (age_label < 100)] = 9

    return age_convert


# cate: categories | reg: regression
def create_data_gen_df(dataframe, batch_size, mode='cate'):
    dataframe = dataframe.reset_index(drop=True)
    all_nums = len(dataframe)
    while True:
        idxs = np.random.permutation(all_nums)
        start = 0

        while start + batch_size < all_nums:
            sub_idxs = list(idxs[start:start + batch_size])
            sub_df = dataframe.iloc[sub_idxs]

            # y label
            age_label = sub_df.age.to_numpy()

            if mode == 'cate':
                age_label = age_categories_convert(age_label)
                age_label = to_categorical(age_label, num_classes=10)
                age_label = np.expand_dims(age_label, axis=2)

            elif mode == 'reg':
                age_label = np.expand_dims(age_label, axis=1)

            y_label = [age_label]

            # x data
            x_data = np.array([image_decode(row) for row in sub_df.iterrows()])

            # out
            yield [x_data.squeeze()], y_label

            # continue
            start += batch_size


# cate: categories | reg: regression
def create_data_gen_xy(X, y, batch_size, mode='cate'):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    all_nums = len(X)
    while True:
        idxs = np.random.permutation(all_nums)
        start = 0
        while start + batch_size < all_nums:
            sub_idxs = list(idxs[start:start + batch_size])
            sub_X_df = X.iloc[sub_idxs]
            sub_y_df = y.iloc[sub_idxs]

            # y label
            age_label = sub_y_df.age.to_numpy()

            if mode == 'cate':
                age_label = age_categories_convert(age_label)
                age_label = to_categorical(age_label, num_classes=10)
                age_label = np.expand_dims(age_label, axis=2)

            elif mode == 'reg':
                age_label = np.expand_dims(age_label, axis=1)

            y_label = [age_label]

            # x data
            x_data = np.array([image_decode(row) for row in sub_X_df.iterrows()])

            # out
            yield [x_data.squeeze()], y_label

            # continue
            start += batch_size


def plot_results(save_file_path, images, labels, predicts):
    fig, ax = plt.subplots(3, 2, figsize=[10, 20])

    for i in range(3):
        for j in range(2):
            idx = i * 2 + j
            ax[i, j].imshow(images[idx])
            ax[i, j].title.set_text('label: {} - pred: {}'.format(age_dict[labels[idx]], age_dict[predicts[idx]]))

    fig.savefig('{}result.jpg'.format(save_file_path))


def visual_results(save_file_path, test_data, results):
    images = test_data[0][0][:6]
    labels = test_data[1][0][:6].squeeze().argmax(axis=1)
    predicts = np.argmax(results, axis=1)[:6]
    plot_results(save_file_path, images, labels, predicts)
