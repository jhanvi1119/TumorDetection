import cv2
import os
from PIL import Image
import keras
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import tensorflow.image as tfi
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Layer, Conv2D, Dropout, UpSampling2D, concatenate, Add, Multiply, Input, MaxPool2D, BatchNormalization
from tensorflow.keras.models import load_model

def breast(input_image):
    class EncoderBlock(Layer):

        def __init__(self, filters, rate, pooling=True, **kwargs):
            super(EncoderBlock, self).__init__(**kwargs)

            self.filters = filters
            self.rate = rate
            self.pooling = pooling

            self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
            self.drop = Dropout(rate)
            self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
            self.pool = MaxPool2D()

        def call(self, X):
            x = self.c1(X)
            x = self.drop(x)
            x = self.c2(x)
            if self.pooling:
                y = self.pool(x)
                return y, x
            else:
                return x

        def get_config(self):
            base_config = super().get_config()
            return {
                **base_config,
                "filters":self.filters,
                'rate':self.rate,
                'pooling':self.pooling
            }
        
    class DecoderBlock(Layer):

        def __init__(self, filters, rate, **kwargs):
            super(DecoderBlock, self).__init__(**kwargs)

            self.filters = filters
            self.rate = rate

            self.up = UpSampling2D()
            self.net = EncoderBlock(filters, rate, pooling=False)

        def call(self, X):
            X, skip_X = X
            x = self.up(X)
            c_ = concatenate([x, skip_X])
            x = self.net(c_)
            return x

        def get_config(self):
            base_config = super().get_config()
            return {
                **base_config,
                "filters":self.filters,
                'rate':self.rate,
            }
        
    class AttentionGate(Layer):

        def __init__(self, filters, bn, **kwargs):
            super(AttentionGate, self).__init__(**kwargs)

            self.filters = filters
            self.bn = bn

            self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
            self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')
            self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
            self.resample = UpSampling2D()
            self.BN = BatchNormalization()

        def call(self, X):
            X, skip_X = X

            x = self.normal(X)
            skip = self.down(skip_X)
            x = Add()([x, skip])
            x = self.learn(x)
            x = self.resample(x)
            f = Multiply()([x, skip_X])
            if self.bn:
                return self.BN(f)
            else:
                return f
            # return f

        def get_config(self):
            base_config = super().get_config()
            return {
                **base_config,
                "filters":self.filters,
                "bn":self.bn
            }
        
    input_image = cv2.resize(input_image,(256, 256))
    input_image = input_image / 255
    custom_objects = {'EncoderBlock': EncoderBlock, 'DecoderBlock':DecoderBlock, 'AttentionGate':AttentionGate}
    model = load_model('BreastCancerAttentionUNet.h5',custom_objects= custom_objects)
    pred_mask = model.predict(input_image[np.newaxis,...])

    intermi=np.zeros((256,256))

    output=0
    for i in range(256):
        for j in range(256):
            if np.squeeze(pred_mask)[i][j] > 0.5:
                intermi[i][j]=255
                output+=1

    if output>50: 
        colored_image=input_image
        grayscale_image=np.array(intermi,dtype=np.uint8)
        _, mask = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        grayscale_image_colored = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
        blended_image = cv2.bitwise_and(colored_image, colored_image, mask=inv_mask)
        blended_image=(blended_image*255).astype(np.uint8)
        inv_mask2 = cv2.bitwise_not(grayscale_image_colored)

        col_mask=np.zeros((256,256,3))
        for i in range(256):
            for j in range(256):
                if inv_mask2[i][j][0]==255:
                    col_mask[i][j][0]=200
        col_mask=np.array(col_mask,dtype=np.uint8)
        blended_image = cv2.add(blended_image, col_mask)
        return blended_image,"Cancer Detected!!!"
    else:
        with open('/Users/dev/programming/KrackHack/static/images/safe.jpeg', 'rb') as f:
            image_data = f.read()
        
        # Convert image data to NumPy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Decode the image data into a NumPy array
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image, "Cancer Not Detected"

