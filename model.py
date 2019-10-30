import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Activation, add
from tensorflow.keras.layers import ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.resnet50 import ResNet50

def identity_block(input_tensor, kernel_size, filters, l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value):
    filters1, filters2, filters3 = filters
    bn_axis = 3

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_weight_decay))(input_tensor)

    x = BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)
    x = Dropout(dropout_value, seed = seed_value)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_weight_decay))(x)

    x = BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)
    x = Dropout(dropout_value, seed = seed_value)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_weight_decay))(x)

    x = BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)
    x = Dropout(dropout_value, seed = seed_value)(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = 3

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)
    x = Dropout(dropout_value, seed=seed_value)(x)
    x = Activation('relu')(x)


    x = Conv2D(filters2, kernel_size, strides=strides, padding='same',
                    kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_weight_decay))(x)
    x = BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)
    x = Dropout(dropout_value, seed=seed_value)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_weight_decay))(x)
    x = BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)
    x = Dropout(dropout_value, seed=seed_value)(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2_weight_decay))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis,
                                         momentum=batch_norm_decay,
                                         epsilon=batch_norm_epsilon)(shortcut)
    x = Dropout(dropout_value, seed=seed_value)(x)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def resnet50(input_shape, l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value):

    img_input = Input(shape=input_shape)

    x = img_input
    bn_axis = 3

    x = ZeroPadding2D(padding=(3, 3))(x)

    x = Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_weight_decay))(x)
    x = BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value, strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [64, 64, 256], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)


    x = conv_block(x, 3, [128, 128, 512], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [128, 128, 512], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [128, 128, 512], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [128, 128, 512], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)

    x = conv_block(x, 3, [256, 256, 1024], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [256, 256, 1024], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [256, 256, 1024], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [256, 256, 1024], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [256, 256, 1024], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [256, 256, 1024], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)

    x = conv_block(x, 3, [512, 512, 2048], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [512, 512, 2048], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)
    x = identity_block(x, 3, [512, 512, 2048], l2_weight_decay, batch_norm_decay, batch_norm_epsilon, dropout_value, seed_value)

    # average pool, 1000-d fc, softmax
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid',
        kernel_regularizer=l2(l2_weight_decay),
        bias_regularizer=l2(l2_weight_decay))(x)
    model = Model(img_input, x, name='resnet50')

    # Create model.
    return model

def transfer_layers(model, num_layers, trainable=False):
    image_net_model = ResNet50(weights='imagenet', include_top=False)

    num_layers_keep = min(len(image_net_model.layers), 20)
    add = 0
    for i in range(num_layers_keep):
        while model.layers[i + add].name.startswith("dropout"):
            add += 1
        model.layers[i + add] = image_net_model.layers[i]
        model.layers[i + add].trainable = trainable

    return model
