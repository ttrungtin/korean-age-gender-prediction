import cv2
import pandas as pd
import mtcnn
import numpy as np
import os
import pyarrow.feather as feather
import mat73

from datetime import datetime
from scipy.io import loadmat
from tqdm import tqdm

detector = mtcnn.MTCNN()


# gen 0: male | 1: female

def get_triple_boxes(box, nose):
    x, y, w, h = box
    ymin, xmin, ymax, xmax = y, x, y + h, x + w
    nose_x, nose_y = nose
    w_h_margin = abs(w - h)
    top_nose_dist = nose_y - ymin

    '''
        - bbox_ori: (x, y, w, h)
        - bbox_1,2,3: (h_min, w_min, h_max, w_max)  
    '''
    bbox_1 = np.array([xmin - w_h_margin, ymin - w_h_margin, xmax + w_h_margin, ymax + w_h_margin])
    bbox_2 = np.array([nose_x - top_nose_dist, nose_y - top_nose_dist, nose_x + top_nose_dist, nose_y + top_nose_dist])
    bbox_3 = np.array([nose_x - w // 2, nose_y - w // 2, nose_x + w // 2, nose_y + w // 2])

    return bbox_1, bbox_2, bbox_3


def processing_data(img_dir, profile, padding=200):
    # Read image & Bordering
    img = cv2.imread(img_dir)
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

    # Find faces in images
    faces = detector.detect_faces(img)

    # No face in imgae
    if len(faces) == 0:
        # print("No Face: {}".format(img_dir))
        return {
            'gen': np.nan,
            'age': np.nan,
            'img_ori': np.nan,
            'img_box1': np.nan,
            'img_box2': np.nan,
            'img_box3': np.nan
        }

    # Take only 1st face
    # print("1st Face: {}".format(img_dir))
    face = faces[0]

    if profile == 'afad':
        '''
        - Extract data
        - Ex: ['D:', 'Tempdata', 'AFAD-Full', '15', '111', '109008-0.jpg']
        - Note: 111: male | 112: female -> 0: male | 1: female
        '''
        info = img_dir.split("\\")
        gen = 0 if info[-2] == '111' else 1
        age = int(info[-3])
        bbox_ori = np.array(face['box'])
        bbox_1, bbox_2, bbox_3 = get_triple_boxes(bbox_ori, face['keypoints']['nose'])

    elif profile == 'utk':
        '''
        - Extract data
        - Ex: ['D:', 'Tempdata', 'UTKFace', '1_0_0_20161219154018476.jpg']
        - Note: age_gen_??_??.jpg -> 0: male | 1: female
        '''
        info = img_dir.split("\\")[-1].split("_")
        gen = 0 if int(info[1]) == 0 else 1
        age = int(info[0])
        bbox_ori = np.array(face['box'])
        bbox_1, bbox_2, bbox_3 = get_triple_boxes(bbox_ori, face['keypoints']['nose'])

    elif profile == 'imdb' or profile == 'wiki' or profile == 'cacd':
        '''
        - Extract data
        '''
        gen = np.nan
        age = np.nan
        bbox_ori = np.array(face['box'])
        bbox_1, bbox_2, bbox_3 = get_triple_boxes(bbox_ori, face['keypoints']['nose'])

    elif profile == 'asia':
        '''
        - Extract data
        - Ex: 00032A02.jpg -> Split A -> ['00032', '02']
        - Note: Idx < 7380 is Female, else is Male
        '''
        info = img_dir.split("\\")[-1].split('A')  # ['00032', '02.jpg']
        gen = 1 if int(info[0]) <= 7380 else 0
        age = int(info[1].split('.')[0])  # ['02', '.jpg']
        bbox_ori = np.array(face['box'])
        bbox_1, bbox_2, bbox_3 = get_triple_boxes(bbox_ori, face['keypoints']['nose'])

    elif profile == 'facial':
        '''
        - Extract data
        '''
        info = img_dir.split("\\")
        gen = np.nan
        age = int(info[-2])
        bbox_ori = np.array(face['box'])
        bbox_1, bbox_2, bbox_3 = get_triple_boxes(bbox_ori, face['keypoints']['nose'])

    # Preprocess image
    # Having: Raw image, bbox_ori, bbox_1, bbox_2, bbox_3
    # for b in [bbox_ori]:
    #     x, y, w, h = b  
    #     img = cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 1)

    # for b in [bbox_1, bbox_2, bbox_3]:
    #     x, y, w, h = b  
    #     img = cv2.rectangle(img, (x, y), (w, h), (255,255,0), 1)
    # print(bbox_ori, bbox_1, bbox_2, bbox_3)
    # cv2.imwrite("%s_%s_%s_1.jpg" % (age, gen, 1), img)

    # Original Crop
    x, y, w, h = bbox_ori
    crop_img_ori = cv2.resize(img[y:y + h, x:x + w, :], (160, 160))

    # Neighbor Crop
    crop_img_neighbor = []
    for box in [bbox_1, bbox_2, bbox_3]:
        w_min, h_min, w_max, h_max = box
        crop_img_neighbor.append(cv2.resize(img[w_min:w_max, h_min:h_max, :], (64, 64)))

    # Encode image
    _, crop_img_ori = cv2.imencode(".jpg", crop_img_ori)
    crop_img_ori = crop_img_ori.tobytes()

    for idx, neighbor in enumerate(crop_img_neighbor):
        _, neighbor = cv2.imencode(".jpg", neighbor)
        neighbor = neighbor.tobytes()
        crop_img_neighbor[idx] = neighbor

    return {
        'gen': gen,
        'age': age,
        'img_ori': crop_img_ori,
        'img_box1': crop_img_neighbor[0],
        'img_box2': crop_img_neighbor[1],
        'img_box3': crop_img_neighbor[2]
    }


def afad(main_dir, save_dir):
    all_img_dir = []
    dataset_df = []

    # List all image dir
    try:
        for sub_dir in os.listdir(main_dir):
            sub_dir_2 = os.path.join(main_dir, sub_dir)

            for gen_dir in os.listdir(sub_dir_2):
                sub_dir_3 = os.path.join(sub_dir_2, gen_dir)

                for img_dir in os.listdir(sub_dir_3):
                    img_dir = os.path.join(sub_dir_3, img_dir)

                    all_img_dir.append(img_dir)

    except NotADirectoryError as e:
        print("File appeared: {}".format(e))

    # Process one-by-one
    for idx, img_dir in enumerate(tqdm(all_img_dir)):
        try:
            processed_data = processing_data(img_dir, profile='afad')
            dataset_df.append(processed_data)

            if (idx % 50000 == 0) and (idx != 0):
                dataset_df = pd.DataFrame(dataset_df)
                dataset_df = dataset_df.dropna()
                feather.write_feather(dataset_df, "{}\\AFAD-Full_{}.feather".format(save_dir, idx))
                dataset_df = []

        except Exception as e:
            print("Unknwon error: {}".format(e))

    # Save
    dataset_df = pd.DataFrame(dataset_df)
    dataset_df = dataset_df.dropna()
    feather.write_feather(dataset_df, "{}\\AFAD-Full.feather".format(save_dir))


def utkface(main_dir, save_dir):
    all_img_dir = []
    dataset_df = []

    # List all image dir
    try:
        for img_dir in os.listdir(main_dir):
            img_dir = os.path.join(main_dir, img_dir)

            all_img_dir.append(img_dir)

    except NotADirectoryError as e:
        print("File appeared: {}".format(e))

    # Process one-by-one
    for idx, img_dir in enumerate(tqdm(all_img_dir)):
        try:
            processed_data = processing_data(img_dir, profile='utk')
            dataset_df.append(processed_data)

            if (idx % 50000 == 0) and (idx != 0):
                dataset_df = pd.DataFrame(dataset_df)
                dataset_df = dataset_df.dropna()
                feather.write_feather(dataset_df, "{}\\UTKFace_{}.feather".format(save_dir, idx))
                dataset_df = []

        except Exception as e:
            print("Unknwon error: {}".format(e))

    # Save
    dataset_df = pd.DataFrame(dataset_df)
    dataset_df = dataset_df.dropna()
    feather.write_feather(dataset_df, "{}\\UTKFace.feather".format(save_dir))


def imdb(main_dir, save_dir):
    mat_dir = os.path.join(main_dir, 'imdb.mat')
    meta = loadmat(mat_dir)

    full_dir = [os.path.abspath(os.path.join(main_dir, p[0])) for p in meta['imdb'][0, 0]["full_path"][0]]

    dob = meta['imdb'][0, 0]['dob'][0]
    taken = meta['imdb'][0, 0]['photo_taken'][0]

    '''
        - Original gender: 1 - male | 0 - female
        - Converted gender: 0 - male | 0.99 - female
    '''
    gen = meta['imdb'][0, 0]['gender'][0]
    gen = np.where(gen == 1, 0.1, 0.99)

    age = [taken[i] - datetime.fromordinal(max(int(dob[i]) - 366, 1)).year for i in range(len(dob))]

    dataset_df = []
    # Process one-by-one
    for idx, img_dir in enumerate(tqdm(full_dir)):
        try:
            processed_data = processing_data(img_dir, profile='imdb')

            processed_data['gen'] = gen[idx]
            processed_data['age'] = age[idx]

            dataset_df.append(processed_data)

            if (idx % 50000 == 0) and (idx != 0):
                dataset_df = pd.DataFrame(dataset_df)
                dataset_df = dataset_df.dropna()
                feather.write_feather(dataset_df, "{}\\imdb_crop_{}.feather".format(save_dir, idx))
                dataset_df = []

        except Exception as e:
            print("Unknwon error: {}".format(e))

    # Save
    dataset_df = pd.DataFrame(dataset_df)
    dataset_df = dataset_df.dropna()
    feather.write_feather(dataset_df, "{}\\imdb_crop.feather".format(save_dir))


def wiki(main_dir, save_dir):
    mat_dir = os.path.join(main_dir, 'wiki.mat')
    meta = loadmat(mat_dir)

    full_dir = [os.path.abspath(os.path.join(main_dir, p[0])) for p in meta['wiki'][0, 0]["full_path"][0]]

    dob = meta['wiki'][0, 0]['dob'][0]
    taken = meta['wiki'][0, 0]['photo_taken'][0]

    '''
        - Original gender: 1 - male | 0 - female
        - Converted gender: 0 - male | 0.99 - female
    '''
    gen = meta['wiki'][0, 0]['gender'][0]
    gen = np.where(gen == 1, 0.1, 0.99)

    age = [taken[i] - datetime.fromordinal(max(int(dob[i]) - 366, 1)).year for i in range(len(dob))]

    dataset_df = []
    # Process one-by-one
    for idx, img_dir in enumerate(tqdm(full_dir)):
        try:
            processed_data = processing_data(img_dir, profile='wiki')

            processed_data['gen'] = gen[idx]
            processed_data['age'] = age[idx]

            dataset_df.append(processed_data)

            if (idx % 50000 == 0) and (idx != 0):
                dataset_df = pd.DataFrame(dataset_df)
                dataset_df = dataset_df.dropna()
                feather.write_feather(dataset_df, "{}\\wiki_crop_{}.feather".format(save_dir, idx))
                dataset_df = []

        except Exception as e:
            print("Unknwon error: {}".format(e))

    # Save
    dataset_df = pd.DataFrame(dataset_df)
    dataset_df = dataset_df.dropna()
    feather.write_feather(dataset_df, "{}\\wiki_crop.feather".format(save_dir))


def asia(main_dir, save_dir):
    all_img_dir = []
    dataset_df = []

    # List all image dir
    try:
        for img_dir in os.listdir(main_dir):
            img_dir = os.path.join(main_dir, img_dir)

            all_img_dir.append(img_dir)

    except NotADirectoryError as e:
        print("File appeared: {}".format(e))

    # Process one-by-one
    for idx, img_dir in enumerate(tqdm(all_img_dir)):
        try:
            processed_data = processing_data(img_dir, profile='asia')
            dataset_df.append(processed_data)

            if (idx % 50000 == 0) and (idx != 0):
                dataset_df = pd.DataFrame(dataset_df)
                dataset_df = dataset_df.dropna()
                feather.write_feather(dataset_df, "{}\\asia_aglined_{}.feather".format(save_dir, idx))
                dataset_df = []

        except Exception as e:
            print("Unknwon error: {}".format(e))

    # Save
    dataset_df = pd.DataFrame(dataset_df)
    dataset_df = dataset_df.dropna()
    feather.write_feather(dataset_df, "{}\\asia_aglined.feather".format(save_dir))


def cacd(main_dir):
    mat_dir = os.path.join(main_dir, 'celebrity2000.mat')
    meta = mat73.loadmat(mat_dir)

    full_dir = [os.path.abspath(os.path.join("D:\\Tempdata\\cacd", "CACD2000", p[0])) for p in
                meta['celebrityImageData']['name']]
    age = [a for a in meta['celebrityImageData']['age']]

    dataset_df = []
    # Process one-by-one
    for idx, img_dir in enumerate(tqdm(full_dir)):
        try:
            processed_data = processing_data(img_dir, profile='cacd')

            processed_data['age'] = age[idx]
            processed_data['gen'] = -1

            dataset_df.append(processed_data)

            if (idx % 50000 == 0) and (idx != 0):
                dataset_df = pd.DataFrame(dataset_df)
                dataset_df = dataset_df.dropna()
                feather.write_feather(dataset_df, "cacd_{}.feather".format(idx))
                dataset_df = []

        except Exception as e:
            print("Unknwon error: {}".format(e))

    # Save
    dataset_df = pd.DataFrame(dataset_df)
    dataset_df = dataset_df.dropna()
    feather.write_feather(dataset_df, "cacd.feather")


def facial(main_dir):
    all_img_dir = []
    dataset_df = []

    # List all image dir
    try:
        for age_dir in os.listdir(main_dir):
            sub_dir = os.path.join(main_dir, age_dir)

            for img_dir in os.listdir(sub_dir):
                img_dir = os.path.join(sub_dir, img_dir)

                all_img_dir.append(img_dir)

    except NotADirectoryError as e:
        print("File appeared: {}".format(e))

    # Process one-by-one
    for idx, img_dir in enumerate(tqdm(all_img_dir)):
        try:
            processed_data = processing_data(img_dir, profile='facial')

            processed_data['gen'] = -1

            dataset_df.append(processed_data)

            if (idx % 50000 == 0) and (idx != 0):
                dataset_df = pd.DataFrame(dataset_df)
                dataset_df = dataset_df.dropna()
                feather.write_feather(dataset_df, "facial_age_{}.feather".format(idx))
                dataset_df = []

        except Exception as e:
            print("Unknwon error: {}".format(e))

    # Save
    dataset_df = pd.DataFrame(dataset_df)
    dataset_df = dataset_df.dropna()
    feather.write_feather(dataset_df, "facial_age.feather")


if __name__ == '__main__':
    # afad("D:\\Dataset\\Raw\\AFAD-Full", "D:\\Dataset\\Feather\\AFAD-Full") # CHECK DONE !
    # utkface("D:\\Dataset\\Raw\\UTKFace", "D:\\Dataset\\Feather\\UTKFace") # CHECK DONE !
    # imdb("D:\\Dataset\\Raw\\imdb_crop", "D:\\Dataset\\Feather\\imdb_crop") # CHECK DONE !
    # wiki("D:\\Dataset\\Raw\\wiki_crop", "D:\\Dataset\\Feather\\wiki_crop") # CHECK DONE !
    # asia("D:\\Dataset\\Raw\\All-Age-Faces Dataset\\aglined faces",
    #      "D:\\Dataset\\Feather\\All-Age-Faces Dataset\\aglined faces")
    # cacd("D:\\Tempdata\\cacd") # no gender
    facial("D:\\Tempdata\\facial-age")  # no gender

# img = np.frombuffer(dataset_df['img'][0], np.uint8)
# img = cv2.imdecode(img, cv2.IMREAD_COLOR)
# print(img.shape)
