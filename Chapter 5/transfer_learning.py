import os
import time
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds 
import matplotlib.pyplot as plt

SPLIT_WEIGHTS = (8, 1, 1)


splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS) 

# https://www.tensorflow.org/datasets/catalog/horses_or_humans
(raw_train, raw_validation, raw_test), metadata = tfds.load('horses_or_humans', split=list(splits), with_info=True, as_supervised=True)

get_label_name = metadata.features['label'].int2str

def show_images(dataset):
    for image, label in dataset.take(10):
        plt.figure()
        plt.imshow(image) 
        plt.title(get_label_name(label))

show_images(raw_train)

IMG_SIZE = 160 # All images will be resized to 160x160

# why not using the preprocess_input function which comes with mobile net V2 application?
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) 
    return image, label

train = raw_train.map(format_example) 
validation = raw_validation.map(format_example) 
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 2000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE) 
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

base_model.trainable = False

base_model.summary()

# Model: "mobilenetv2_1.00_160"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_1 (InputLayer)            [(None, 160, 160, 3) 0                                            
# __________________________________________________________________________________________________
# Conv1_pad (ZeroPadding2D)       (None, 161, 161, 3)  0           input_1[0][0]   
# ...
# Conv_1 (Conv2D)                 (None, 5, 5, 1280)   409600      block_16_project_BN[0][0]        
# __________________________________________________________________________________________________
# Conv_1_bn (BatchNormalization)  (None, 5, 5, 1280)   5120        Conv_1[0][0]                     
# __________________________________________________________________________________________________
# out_relu (ReLU)                 (None, 5, 5, 1280)   0           Conv_1_bn[0][0]                  
# ==================================================================================================
# Total params: 2,257,984
# Trainable params: 2,223,872
# Non-trainable params: 34,112

# https://www.tensorflow.org/datasets/overview
for image_batch, label_batch in train_batches.take(1): 
    pass
print (image_batch.shape)

feature_batch = base_model(image_batch) 
print(feature_batch.shape)

# extend the network
# pooling layer
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() 
feature_batch_average = global_average_layer(feature_batch) 
print(feature_batch_average.shape)

# output layer
prediction_layer = tf.keras.layers.Dense(1) 
prediction_batch = prediction_layer(feature_batch_average) 
print(prediction_batch.shape)

# here we use the sequential API but one could also use the functional API instead.
model = tf.keras.Sequential([ base_model, global_average_layer, prediction_layer])

base_learning_rate = 0.0001 
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_ rate), 
    loss='binary_crossentropy', metrics=['accuracy'])
