import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

feather_dict = {
    'wiki': 'wiki_crop',
    'imdb': 'imdb',
    'utk': 'UTKFace',
    'cacd': 'cacd',
    'facial': 'facial',
    'asia': 'All-Age-Faces',
    'afad': 'AFAD-Full'
}

'''
    input:  data_path: the folder that contants all avaiable data folder in *.feather
            source: which data will be used
    output: a dataframe which a combination of data in all sources 
'''


def load_data_all(data_path, source):
    source = source.split('|')
    init_df = True
    dataframe = pd.DataFrame()

    for s in source:
        feather_folder = feather_dict[s]
        feather_path = os.path.join(data_path, feather_folder)

        all_feather_files = os.listdir(feather_path)

        for feather in all_feather_files:

            feather_dir = os.path.join(feather_path, feather)
            loaded_data = pd.read_feather(feather_dir)

            # For first dataframe loaded
            if init_df:
                dataframe = loaded_data
                init_df = False
            else:
                dataframe = pd.concat([dataframe, loaded_data], ignore_index=True, sort=False)

            print(loaded_data.shape)
            print(feather_dir)
    print(dataframe.shape)

    # Filter
    dataframe = dataframe[(dataframe['age'] > 0) & (dataframe['age'] < 101)]
    dataframe = dataframe.dropna()

    return dataframe


def plot_count(idx_gen, value_gen, idx_age, value_age, name):
    fig, ax = plt.subplots(1, 2, figsize=[10, 5])

    ax[0].bar(idx_gen, value_gen)
    ax[0].set_title("Gender")
    ax[1].bar(idx_age, value_age)
    ax[1].set_title("Age")

    # fig.suptitle('ADAF Dataset')
    plt.savefig('{}.jpg'.format(name))
    # plt.show()


'''
    'wiki': 'wiki_crop',
    'imdb': 'imdb',
    'utk': 'UTKFace',
    'cacd': 'cacd',
    'facial': 'facial',
    'asia': 'All-Age-Faces',
    'afad': 'AFAD-Full'
'''

if __name__ == '__main__':
    name = 'imdb'
    data = load_data_all("D:\\Dataset\\Feather", name)

    gen = data['gen'].value_counts().to_frame()
    age = data['age'].value_counts().to_frame().sort_index()

    idx_gen = ["Male", "Female"]
    idx_age = age.index

    value_gen = gen.gen
    value_age = age.age

    plot_count(idx_gen, value_gen, idx_age, value_age, name)