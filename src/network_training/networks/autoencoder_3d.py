# Neural Network Packages
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from keras import backend as K

def autoencoder_3d():
    '''
    '''
    input_img = Input(shape=(20, 20, 20, 1))

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling3D((2, 2, 2), padding='same', data_format='channels_last')(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((2, 2, 2), padding='same', data_format='channels_last')(x)

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

    return autoencoder
