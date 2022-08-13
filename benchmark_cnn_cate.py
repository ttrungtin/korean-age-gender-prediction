import os
from builtins import int

from utils import load_data, create_data_gen
from model import base, cnn, cnn2, cnn3, c3ae_base, c3ae_alpha
from tensorflow.keras.optimizers import Adam

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

model_dict = {
    "base": base,
    "cnn": cnn,
    "cnn2": cnn2,
    "cnn3": cnn3,
    "c3ae_base": c3ae_base,
    "c3ae_alpha": c3ae_alpha
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # PARAMS
    batch_size = 256
    epochs = 100
    input_shape = [160, 160, 3]

    model_type = 'cnn'
    model_various = 'alpha'
    ver = "base40"
    save_file_path = ".\\save\\{}_{}_{}\\".format(model_various, model_type, ver)
    log_path = ".\\logs\\log_{}_{}_{}\\".format(model_type, model_various, ver)

    use_valid = False
    num_classes = 11
    soft_label = True  # soft categorical label
    data_path = "D:\\Dataset\\Feather"
    source = 'afad'

    # LOAD DATA
    benckmark_df = load_data(data_path, source)

    # DATA GEN
    benck_gen = create_data_gen(benckmark_df, batch_size=batch_size, mode="all", num_classes=num_classes,
                                soft_label=soft_label, model_type=model_type)

    # # TEST ZONE
    # for i in benck_gen:
    #     print(i)
    #     break

    # MODEL
    model = model_dict[model_type].create_model_all(input_shape=input_shape, num_classes=num_classes)

    # LOAD MODEL
    print("Model {} loaded.".format(save_file_path))
    model.load_weights(save_file_path)

    # COMPILE
    model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'cate': 'categorical_crossentropy', 'reg': 'mae'},
            metrics={"cate": 'categorical_accuracy', "reg": 'mae'}
        )

    # EVALUATE
    print("EVALUATING...")
    model.evaluate(benck_gen, steps=len(benckmark_df)/batch_size)
