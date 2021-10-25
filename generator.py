import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,\
                                    UpSampling1D,\
                                    Conv2DTranspose,\
                                    BatchNormalization,\
                                    Dropout,\
                                    Dense,\
                                    LeakyReLU,\
                                    Reshape

from tensorflow.keras.models import Sequential


class Generator(tf.keras.models.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = Sequential([
            Dense(8 * 8 * 512, use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Reshape((8, 8, 512)),
            Conv2DTranspose(384, (5, 5), (1, 1), padding="same", use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2DTranspose(256, (5, 5), (2, 2), padding="same", use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2DTranspose(128, (5, 5), (2, 2), padding="same", use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2DTranspose(64, (5, 5), (2, 2), padding="same", use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2DTranspose(32, (5, 5), (2, 2), padding="same", use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2DTranspose(3, (5, 5), (2, 2), padding="same", use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

        ])

    def call(self, x):
        return self.model(x)
