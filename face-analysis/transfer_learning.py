# In[1]:
get_ipython().system('export TF_ENABLE_ONEDNN_OPTS=1')
get_ipython().system('export TF_GPU_ALLOCATOR=cuda_malloc_async')

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from glob import glob
sys.path.append('../code/tfkeras-vggface/')
sys.path.append('../code/LCNN/')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import  Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tfkeras_vggface.vggface import VGGFace
from tfkeras_vggface import utils
from tensorflow.keras.preprocessing import image

from src.model import layers as lcnn_layers
from src.model import lcnn

#import wandb
#wandb.init(project="sasa", entity="asabuncuoglu13")


BATCH_SIZE = 32
IMG_SIZE = (160, 160)

#wandb.define_metric("val_acc_after_finetune")
#wandb.define_metric("val_loss_after_finetune")
#wandb.define_metric("test_loss")
#wandb.define_metric("test_acc")
#wandb.define_metric("val_loss")
#wandb.define_metric("val_acc")

print("Experiment with MobileNet with LCNN Final Layers")
print("================================================")

#df = pd.DataFrame({ 'val-acc' : 0.00, 'vall-acc-finetune' : 0.00, 'test-acc' : 0.00 }, index = ['path'])

# In[2]:
data_dirs = []
total_val_accuracy = []
total_test_accuracy = []
for i in range(7,9):
  data_dirs.append(glob(os.path.join('../face-levels/%d/' % i, "*")))

for i in range(2):
  for data_dir in data_dirs[i]:

    print("DATASET: %s" % data_dir)
    print("================================================")
    """
    wandb.config = {
      "dataset": data_dir,
      "model": "MobileNet with LCNN",
      "learning_rate": 0.001,
      "epochs": 100,
      "batch_size": BATCH_SIZE
    }
    """

    train_dataset = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=IMG_SIZE,
      batch_size=BATCH_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=IMG_SIZE,
      batch_size=BATCH_SIZE)

    class_names = train_dataset.class_names

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    IMG_SHAPE = IMG_SIZE + (3,)

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    #base_model = VGGFace(include_top=False, input_shape=IMG_SHAPE, pooling='max')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)

    base_model.trainable = False

    prediction_layer = Sequential([
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(len(class_names))
    ])

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.Dropout(0.2)(x)

    conv_2d_8 = lcnn.MaxOutConv2D(x, 64, kernel_size=1, strides=1, padding="same")
    batch_norm_8 = layers.BatchNormalization()(conv_2d_8)

    conv_2d_9 = lcnn.MaxOutConv2D(batch_norm_8, 64, kernel_size=3, strides=1, padding="same")
    maxpool_9 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_2d_9)
    flatten = layers.Flatten()(maxpool_9)

    dense_10 = lcnn.MaxOutDense(flatten, 160)
    batch_norm_10 = layers.BatchNormalization()(dense_10)
    dropout_10 = layers.Dropout(0.75)(batch_norm_10)

    outputs = layers.Dense(len(class_names), activation="softmax")(dropout_10)

    #outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    initial_epochs = 50

    loss0, accuracy0 = model.evaluate(validation_dataset)

    #print("initial loss: {:.2f}".format(loss0))
    #print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset,
                        verbose=0)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    before_fine_acc = val_acc
    print('Validation accuracy before fine tuning: ', val_acc[-1]) 
    #wandb.log({"val_loss": val_loss, "val_acc": val_acc})


    base_model.trainable = True

    fine_tune_at = 100

    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False


    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                  metrics=['accuracy'])


    fine_tune_epochs = 50
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                            epochs=total_epochs,
                            initial_epoch=history.epoch[-1],
                            validation_data=validation_dataset,
                            verbose=0)


    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    print('Validation accuracy after fine tuning: ', val_acc[-1])

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    #wandb.log({"val_loss_after_finetune": val_loss, "val_acc_after_finetune": val_acc})

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy : ', accuracy)
    #wandb.log({"test_loss": loss, "test_acc": accuracy})
    
    #df.loc[data_dir] = [before_fine_acc, val_acc, accuracy]

#wandb.finish()
#print("Total Val Accuracy: {:.2f}".format(sum(total_val_accuracy) / len(total_val_accuracy)))
#print("Total Test Accuracy: {:.2f}".format(sum(total_test_accuracy) / len(total_test_accuracy)))
#df.to_csv("vggface_with_lcnn.csv")
#df.to_csv("mobilenet_with_lcnn.csv")

