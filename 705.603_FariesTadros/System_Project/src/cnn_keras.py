from pyclbr import Class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

import tensorboard
tensorboard.__version__

logdir="logs/fit/"
file_writer = tf.summary.create_file_writer(logdir)


# ignoring warnings
import warnings
warnings.simplefilter("ignore")

import os, cv2, json
from PIL import Image

WORK_DIR = '../data/cassava-leaf-disease-classification'
#os.listdir(WORK_DIR)

class MyCNN:
    def __init__(self, name, age):
        self.workdir = WORK_DIR
        self.data_dir = WORK_DIR

    def read_data(WORK_DIR):
        with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json")) as file:
            print(json.dumps(json.loads(file.read()), indent=4))

        train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
        train_labels.head()

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize = (6, 4))

        for i in ['top', 'right', 'left']:
            ax.spines[i].set_visible(False)
        ax.spines['bottom'].set_color('black')

        fig = sns.countplot(train_labels.label, edgecolor = 'black',
                    palette = reversed(sns.color_palette("viridis", 5)))
        plt.xlabel('Classes', fontfamily = 'serif', size = 15)
        plt.ylabel('Count', fontfamily = 'serif', size = 15)
        plt.xticks(fontfamily = 'serif', size = 12)
        plt.yticks(fontfamily = 'serif', size = 12)
        ax.grid(axis = 'y', linestyle = '--', alpha = 0.9)
        #plt.show()
        tf.summary.image("Classes", fig, step=0)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        return train_labels

    def data_gen(train_labels):
        # Main parameters
        BATCH_SIZE = 8
        STEPS_PER_EPOCH = len(train_labels)*0.8 / BATCH_SIZE
        VALIDATION_STEPS = len(train_labels)*0.2 / BATCH_SIZE
        EPOCHS = 1
        TARGET_SIZE = 512


        train_labels.label = train_labels.label.astype('str')

        train_datagen = ImageDataGenerator(validation_split = 0.2,
                                            preprocessing_function = None,
                                            rotation_range = 45,
                                            zoom_range = 0.2,
                                            horizontal_flip = True,
                                            vertical_flip = True,
                                            fill_mode = 'nearest',
                                            shear_range = 0.1,
                                            height_shift_range = 0.1,
                                            width_shift_range = 0.1)

        train_generator = train_datagen.flow_from_dataframe(train_labels,
                                directory = os.path.join(WORK_DIR, "train_images"),
                                subset = "training",
                                x_col = "image_id",
                                y_col = "label",
                                target_size = (TARGET_SIZE, TARGET_SIZE),
                                batch_size = BATCH_SIZE,
                                class_mode = "sparse")


        validation_datagen = ImageDataGenerator(validation_split = 0.2)

        validation_generator = validation_datagen.flow_from_dataframe(train_labels,
                                directory = os.path.join(WORK_DIR, "train_images"),
                                subset = "validation",
                                x_col = "image_id",
                                y_col = "label",
                                target_size = (TARGET_SIZE, TARGET_SIZE),
                                batch_size = BATCH_SIZE,
                                class_mode = "sparse")
        return train_generator,validation_generator,TARGET_SIZE



    def create_model(TARGET_SIZE):
        conv_base = EfficientNetB0(include_top = False, weights = None,
                                input_shape = (TARGET_SIZE, TARGET_SIZE, 3))
        model = conv_base.output
        model = layers.GlobalAveragePooling2D()(model)
        model = layers.Dense(5, activation = "softmax")(model)
        model = models.Model(conv_base.input, model)

        model.compile(optimizer = Adam(lr = 0.001),
                    loss = "sparse_categorical_crossentropy",
                    metrics = ["acc"])
        return model

    def train_model(model,train_generator,STEPS_PER_EPOCH,EPOCHS,validation_generator,VALIDATION_STEPS,TARGET_SIZE):

        model.summary()


        model_save = ModelCheckpoint('./EffNetB0_512_8_best_weights.h5', 
                                    save_best_only = True, 
                                    save_weights_only = True,
                                    monitor = 'val_loss', 
                                    mode = 'min', verbose = 1)
        early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                                patience = 5, mode = 'min', verbose = 1,
                                restore_best_weights = True)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                    patience = 2, min_delta = 0.001, 
                                    mode = 'min', verbose = 1)

        # Define the Keras TensorBoard callback.

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        history = model.fit(
            train_generator,
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs = EPOCHS,
            validation_data = validation_generator,
            validation_steps = VALIDATION_STEPS,
            callbacks=[model_save]
        )

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.set_style("white")
        plt.suptitle('Train history', size = 15)

        ax1.plot(epochs, acc, "bo", label = "Training acc")
        ax1.plot(epochs, val_acc, "b", label = "Validation acc")
        ax1.set_title("Training and validation acc")
        ax1.legend()

        ax2.plot(epochs, loss, "bo", label = "Training loss", color = 'red')
        ax2.plot(epochs, val_loss, "b", label = "Validation loss", color = 'red')
        ax2.set_title("Training and validation loss")
        ax2.legend()

        plt.show()
        model.save('.')
        return model

if __name__ == "__main__":
    CNN = MyCNN
    WORK_DIR = './System_Project/data/cassava-leaf-disease-classification'

    train_labels = CNN.read_data(WORK_DIR)
    train_generator,validation_generator,TARGET_SIZE = CNN.data_gen(train_labels)
    BATCH_SIZE = 8
    STEPS_PER_EPOCH = len(train_labels)*0.8 / BATCH_SIZE
    VALIDATION_STEPS = len(train_labels)*0.2 / BATCH_SIZE
    EPOCHS = 1
    TARGET_SIZE = 512
    model = CNN.create_model(TARGET_SIZE)
    train = CNN.train_model(model,train_generator,STEPS_PER_EPOCH,EPOCHS,validation_generator,VALIDATION_STEPS,TARGET_SIZE)

    preds = []
    image = Image.open(os.path.join(WORK_DIR,  "test_images", "2216849948.jpg"))
    image = image.resize((TARGET_SIZE, TARGET_SIZE))
    image = np.expand_dims(image, axis = 0)
    preds.append(np.argmax(model.predict(image)))

    print(preds)


