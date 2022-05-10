import tensorflow as tf
import os

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from model import cnn, cnn2
from utils import visual_history

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':
    # PARAMS ------------------------------------------------
    batch_size = 512
    epochs = 10
    image_size = (160, 160)
    seed = 22
    learning_rate = 0.1

    mode = 'all'
    ver = 1
    model_type = 'cnn2_base'
    save_file_path = ".\\save\\{}_{}_{}_imgnet\\".format(model_type, ver, mode)
    log_path = ".\\log\\"

    train_path = "D:\\Dataset\\Raw\\imagenet21k_resized\\imagenet21k_train"
    valid_path = "D:\\Dataset\\Raw\\imagenet21k_resized\\imagenet21k_val"

    # LOAD DATA ---------------------------------------------
    train_ds = image_dataset_from_directory(
        directory=train_path,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed
    )

    val_ds = image_dataset_from_directory(
        directory=valid_path,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed
    )

    # DATA CACHE ---------------------------------------------
    class_names = train_ds.class_names
    # autotune = tf.data.AUTOTUNE
    # train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    # val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    # NORM ---------------------------------------------------

    normalization_layer = Rescaling(1. / 255)
    norm_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    norm_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # MODEL -------------------------------------------------
    model = cnn2.create_model_all(input_shape=[160, 160, 3],
                                  num_classes=len(class_names),
                                  image_net_pre_train=True)

    model.summary()

    # LOAD MODEL ------------------------------------------------------
    if os.path.exists(save_file_path):
        print("Model {} loaded.".format(save_file_path))
        model.load_weights(save_file_path)

    else:
        print("Train new model.")

    # COMPILE -----------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'cate': SparseCategoricalCrossentropy(from_logits=True)},
        metrics={'cate': 'accuracy'}
    )

    # # TEST ZONE -------------------------------------------------------
    # img_ori = tf.random.uniform(shape=[1, 160, 160, 3])
    # result = model([img_ori])
    # print(result)

    # CALLBACKS ---------------------------------------------
    callbacks = [
        ModelCheckpoint(save_file_path, monitor='val_loss', verbose=1, save_best_only=True,
                        save_weights_only=True,
                        mode='min'),
        TensorBoard(log_dir=log_path, write_images=True, update_freq='epoch'),
        ReduceLROnPlateau(min_lr=0.00001)
    ]

    # FIT ---------------------------------------------------
    history = model.fit(norm_train_ds,
                        validation_data=norm_val_ds,
                        epochs=epochs,
                        callbacks=callbacks)

    # VISUAL HISTORY --------------------------------------------------
    visual_history(save_file_path, history)
