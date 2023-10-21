###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################
from datetime import datetime

import numpy as np
import configparser
from numpy import moveaxis
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as K
# from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
import keras
import sys
sys.path.insert(0, './preprocessing/')
from help_functions import *
# function to obtain data for training/testing (validation)
from extract_patches import get_data_training

# Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = tf.keras.layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = tf.keras.layers.Dropout(0.2)(conv1)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    #
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = tf.keras.layers.Dropout(0.2)(conv2)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    #
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = tf.keras.layers.Dropout(0.2)(conv3)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = tf.keras.layers.concatenate([conv2,up1],axis=1)
    conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = tf.keras.layers.Dropout(0.2)(conv4)
    conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = tf.keras.layers.concatenate([conv1,up2], axis=1)
    conv5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = tf.keras.layers.Dropout(0.2)(conv5)
    conv5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = tf.keras.layers.Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = tf.keras.layers.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = tf.keras.layers.Permute((2,1))(conv6)
    ############
    conv7 = tf.keras.layers.Activation('softmax')(conv6)

    model = tf.keras.Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model




def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

                  padding='same', data_format='last'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = tf.keras.layers.Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = tf.keras.layers.Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = tf.keras.layers.BatchNormalization()(layer1)
                layer1 = tf.keras.layers.Activation('relu')(layer1)
            layer1 = tf.keras.layers.Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                tf.keras.layers.add([layer1, layer]))
            if batch_normalization:
                layer1 = tf.keras.layers.BatchNormalization()(layer1)
            layer1 = tf.keras.layers.Activation('relu')(layer1)
        layer = layer1

    out_layer = tf.keras.layers.add([layer, skip_layer])
    return out_layer

def up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = tf.keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = tf.keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate

#Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def r2_unet(img_w, img_h, n_label, data_format='channels_last'):
    inputs = tf.keras.layers.Input((img_w, img_h,1))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = tf.keras.layers.Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv6 = tf.keras.layers.Reshape((2, patch_height * patch_width))(conv6)
    conv6 = tf.keras.layers.Permute((2, 1))(conv6)
    conv7 = tf.keras.layers.Activation('softmax')(conv6)
    model = tf.keras.Model(inputs=inputs, outputs=conv7)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model

def attention_up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = tf.keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = tf.keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = tf.keras.layers.Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = tf.keras.layers.Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = tf.keras.layers.Activation('relu')(tf.keras.layers.add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = tf.keras.layers.Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = tf.keras.layers.Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = tf.keras.layers.multiply([x, rate])

    return att_x

#Attention R2U-Net
def att_r2_unet(img_w, img_h, n_label, data_format='channels_last'):
    inputs = tf.keras.layers.Input((img_w, img_h, 1))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = tf.keras.layers.Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = tf.keras.layers.Reshape((2, patch_height * patch_width))(conv6)
    conv8 = tf.keras.layers.Permute((2, 1))(conv7)
    conv9 = tf.keras.layers.Activation('sigmoid')(conv8)
    model = tf.keras.Model(inputs=inputs, outputs=conv9)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-6), loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model




#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration_stare.txt')
#patch to the datasets
path_data = config.get('datapaths', 'path_local')
#Experiment name
name_experiment = config.get('experimentname', 'name')
#training settings
N_epochs = int(config.get('trainingsettings', 'N_epochs'))
batch_size = int(config.get('trainingsettings', 'batch_size'))



#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('datapaths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('datapaths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('dataattributes', 'patch_height')),
    patch_width = int(config.get('dataattributes', 'patch_width')),
    N_subimgs = int(config.get('trainingsettings', 'N_subimgs')),
    inside_FOV = config.getboolean('trainingsettings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)


#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]

# change channels first to channels last format
patches_imgs_train = moveaxis(patches_imgs_train, 1, 3)
# patches_masks_train = moveaxis(patches_masks_train, 1, 3)
print("Patch img shape: ", patches_imgs_train.shape)
print("Patch mask shape: ", patches_masks_train.shape)
# model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
# model = r2_unet(patch_width,patch_height,2)
model = att_r2_unet(patch_width, patch_height,2)
print("Check: final output of the network:")
print(model.output_shape)
# plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
# open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
print(patches_masks_train.shape)
# Output batch loss every 1000 batches
# logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


















#
