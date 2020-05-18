from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Softmax

class FMNISTModel(keras.Model):
    # Create the different layers used by the model
    def __init__(self):
        super(FMNISTModel, self).__init__(name='fmnist_model')
        self.conv2d_1   = Conv2D(64, 3, padding='same', activation='relu',input_shape=(28,28))
        self.conv2d_2   = Conv2D(64, 3, padding='same', activation='relu')
        self.max_pool2d = MaxPooling2D((2, 2), padding='same')
        #self.batch_norm = BatchNormalization()
        self.flatten    = Flatten()
        self.dense1     = Dense(512, activation='relu')
        self.dense2     = Dense(10)
        self.dropout    = Dropout(0.3)
        self.softmax    = Softmax()

    # Chain the layers for forward propagation
    def call(self, x):
        # 1st convolution block
        x = self.conv2d_1(x)
        x = self.max_pool2d(x)
        #x = self.batch_norm(x)
        # 2nd convolution block
        x = self.conv2d_2(x)
        x = self.max_pool2d(x)
        #x = self.batch_norm(x)
        # Flatten and classify
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return self.softmax(x)
