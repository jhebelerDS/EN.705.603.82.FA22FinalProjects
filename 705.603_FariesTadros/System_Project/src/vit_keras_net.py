import pandas as pd
import numpy as np
import tensorflow as tf

import tensorflow_addons as tfa
import glob, warnings
import matplotlib.pyplot as pltP
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import vit_keras
from vit_keras import vit
from vit_keras import visualize
import os
warnings.filterwarnings('ignore')
print('TensorFlow Version ' + tf.__version__)

import tensorboard
tensorboard.__version__

logdir="logs/fit/"
file_writer = tf.summary.create_file_writer(logdir)


IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

TRAIN_PATH = './System_Project/data/cassava-leaf-disease-classification/train_images'
TEST_PATH = './System_Project/data/cassava-leaf-disease-classification/test_images'

class MyVit:
    def __init__(self):
        super().__init__()
        self.IMAGE_SIZE = 224

    @staticmethod
    def load_data(TRAIN_PATH,TEST_PATH):
        file_path = os.path.abspath( __file__ )
        full_path = file_path[0:-20]+'data/cassava-leaf-disease-classification/train.csv'
        DF_TRAIN = pd.read_csv(full_path, dtype='str')
        TEST_IMAGES = glob.glob(TEST_PATH + '/*.jpg')
        DF_TEST = pd.DataFrame(TEST_IMAGES, columns = ['image_path'])

        classes = {0 : "Cassava Bacterial Blight (CBB)",
                1 : "Cassava Brown Streak Disease (CBSD)",
                2 : "Cassava Green Mottle (CGM)",
                3 : "Cassava Mosaic Disease (CMD)",
                4 : "Healthy"}
        return DF_TRAIN,DF_TEST,classes

    def data_augment(self,image):
        p_spatial = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
        p_rotate = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
        p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
        p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
        p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
        
        # Flips
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        if p_spatial > .75:
            image = tf.image.transpose(image)
            
        # Rotates
        if p_rotate > .75:
            image = tf.image.rot90(image, k = 3) # rotate 270º
        elif p_rotate > .5:
            image = tf.image.rot90(image, k = 2) # rotate 180º
        elif p_rotate > .25:
            image = tf.image.rot90(image, k = 1) # rotate 90º
            
        # Pixel-level transforms
        if p_pixel_1 >= .4:
            image = tf.image.random_saturation(image, lower = .7, upper = 1.3)
        if p_pixel_2 >= .4:
            image = tf.image.random_contrast(image, lower = .8, upper = 1.2)
        if p_pixel_3 >= .4:
            image = tf.image.random_brightness(image, max_delta = .1)
            
        return image


    def data_loader(self,DF_TRAIN,DF_TEST,TRAIN_PATH):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                                samplewise_center = True,
                                                                samplewise_std_normalization = True,
                                                                validation_split = 0.2,
                                                                preprocessing_function = self.data_augment)

        train_gen = datagen.flow_from_dataframe(dataframe = DF_TRAIN,
                                                directory = TRAIN_PATH,
                                                x_col = 'image_id',
                                                y_col = 'label',
                                                subset = 'training',
                                                batch_size = BATCH_SIZE,
                                                seed = 1,
                                                color_mode = 'rgb',
                                                shuffle = True,
                                                class_mode = 'categorical',
                                                target_size = (self.IMAGE_SIZE, self.IMAGE_SIZE))

        valid_gen = datagen.flow_from_dataframe(dataframe = DF_TRAIN,
                                                directory = TRAIN_PATH,
                                                x_col = 'image_id',
                                                y_col = 'label',
                                                subset = 'validation',
                                                batch_size = BATCH_SIZE,
                                                seed = 1,
                                                color_mode = 'rgb',
                                                shuffle = False,
                                                class_mode = 'categorical',
                                                target_size = (self.IMAGE_SIZE, self.IMAGE_SIZE))

        test_gen = datagen.flow_from_dataframe(dataframe = DF_TEST,
                                            x_col = 'image_path',
                                            y_col = None,
                                            batch_size = BATCH_SIZE,
                                            seed = 1,
                                            color_mode = 'rgb',
                                            shuffle = False,
                                            class_mode = None,
                                            target_size = (self.IMAGE_SIZE, self.IMAGE_SIZE))
        return train_gen,valid_gen,test_gen

    @staticmethod
    def vis_images(train_gen,IMAGE_SIZE):
        images = [train_gen[0][0][i] for i in range(16)]
        fig, axes = plt.subplots(3, 5, figsize = (10, 10))

        axes = axes.flatten()

        for img, ax in zip(images, axes):
            ax.imshow(img.reshape(IMAGE_SIZE, IMAGE_SIZE, 3))
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_model():
        vit_model = vit.vit_b32(
            image_size = IMAGE_SIZE,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = 5)
        return vit_model

    @staticmethod
    def vis_att(vit_model,test_gen): 
        x = test_gen.next()
        image = x[0]
        attention_map = visualize.attention_map(model = vit_model, image = image)

        # Plot results
        fig, (ax1, ax2) = plt.subplots(ncols = 2)
        ax1.axis('off')
        ax2.axis('off')
        ax1.set_title('Original')
        ax2.set_title('Attention Map')
        _ = ax1.imshow(image)
        _ = ax2.imshow(attention_map)
        return attention_map


    def train_model(self,vit_model,train_gen,valid_gen,EPOCHS):
        

        model = tf.keras.Sequential([
                vit_model,
                tf.keras.layers.Flatten(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(11, activation = tfa.activations.gelu),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(5, 'softmax')
            ],
            name = 'vision_transformer')

        model.summary()

        learning_rate = 1e-4

        optimizer = tfa.optimizers.RectifiedAdam(learning_rate = learning_rate)

        model.compile(optimizer = optimizer, 
                    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2), 
                    metrics = ['acc'])

        STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
        STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_acc',
                                                        factor = 0.2,
                                                        patience = 2,
                                                        verbose = 1,
                                                        min_delta = 1e-4,
                                                        min_lr = 1e-6,
                                                        mode = 'max')

        earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_acc',
                                                        min_delta = 1e-4,
                                                        patience = 5,
                                                        mode = 'max',
                                                        restore_best_weights = True,
                                                        verbose = 1)

        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = './model.hdf5',
                                                        monitor = 'val_acc', 
                                                        verbose = 1, 
                                                        save_best_only = True,
                                                        save_weights_only = True,
                                                        mode = 'max')

        callbacks = [earlystopping, reduce_lr, checkpointer]
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        history = model.fit(x = train_gen,
                steps_per_epoch = STEP_SIZE_TRAIN,
                validation_data = valid_gen,
                validation_steps = STEP_SIZE_VALID,
                epochs = EPOCHS,
                callbacks = tensorboard_callback)


        predicted_classes = np.argmax(model.predict(valid_gen, steps = valid_gen.n // valid_gen.batch_size + 1), axis = 1)
        true_classes = valid_gen.classes
        class_labels = list(valid_gen.class_indices.keys())  

        confusionmatrix = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize = (16, 16))
        sns.heatmap(confusionmatrix, cmap = 'Blues', annot = True, cbar = True)

        print(classification_report(true_classes, predicted_classes))

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


if __name__ == "__main__":
    ViT=MyVit()
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 1

    TRAIN_PATH = './System_Project/data/cassava-leaf-disease-classification/train_images'
    TEST_PATH = './System_Project/data/cassava-leaf-disease-classification/test_images'
    DF_TRAIN,DF_TEST,classes = ViT.load_data(TRAIN_PATH,TEST_PATH)
    train_gen,valid_gen,test_gen=ViT.data_loader(DF_TRAIN,DF_TEST,TRAIN_PATH)
    vit_model = ViT.create_model()
    ViT.train_model(vit_model,train_gen,valid_gen,EPOCHS)