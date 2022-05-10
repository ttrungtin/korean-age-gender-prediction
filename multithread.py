import multiprocessing
import time
from tqdm import tqdm
import numpy as np
import os

def check_file(idx, img_dir):
    # print("Start thread {}".format(idx))
    for i in tqdm(img_dir):
        try:
            f = open(i, 'rb')
        except Exception as e:
            print(e)

if __name__ == '__main__':

    # Append all images dir
    main_dir = "D:\\Dataset\\Raw\\imagenet21k_resized\\imagenet21k_train"
    all_img_dir = []

    # List all image dir
    try:
        for age_dir in tqdm(os.listdir(main_dir)):
            sub_dir = os.path.join(main_dir, age_dir)

            for img_dir in os.listdir(sub_dir):
                img_dir = os.path.join(sub_dir, img_dir)

                all_img_dir.append(img_dir)

    except NotADirectoryError as e:
        print("File appeared: {}".format(e))

    print(len(all_img_dir))


    # Split index
    num_workder = 362
    all_img_dir = np.array(all_img_dir)
    all_img_dir_splited = np.split(all_img_dir, num_workder)


    # Multi-threads check

    tic = time.time()

    process_list = []
    for i in range(num_workder):
        p =  multiprocessing.Process(target= check_file, args=[i, all_img_dir_splited[i]])
        p.start()
        process_list.append(p)

    print(process_list)

    for process in process_list:
        process.join()

    toc = time.time()

    print('Done in {:.4f} seconds'.format(toc-tic))