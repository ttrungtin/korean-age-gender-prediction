import pandas as pd
import os
import numpy as np

from cv2 import cv2

feather_dict = {
    'wiki': 'wiki_crop',
    'imdb': 'imdb_crop_{}.feather',
    'utk': 'UTKFace',
    'cacd': 'cacd.feather',
    'facial': 'facial-age.feather',
    'asia': 'All-Age-Faces Dataset',
    'afad': 'AFAD-Full'
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


def create_data_gen_df(dataframe, batch_size):
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
            age_label = np.expand_dims(age_label, axis=1)
            y_label = [age_label]

            # x data
            x_data = np.array([image_decode(row) for row in sub_df.iterrows()])

            # out
            yield [x_data.squeeze()], y_label

            # continue
            start += batch_size


def create_data_gen_xy(X, y, batch_size):
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
            age_label = np.expand_dims(age_label, axis=1)
            y_label = [age_label]

            # x data
            x_data = np.array([image_decode(row) for row in sub_X_df.iterrows()])

            # out
            yield [x_data.squeeze()], y_label

            # continue
            start += batch_size
