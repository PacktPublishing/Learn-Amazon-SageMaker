import os, argparse, glob
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import multi_gpu_model, to_categorical

import subprocess, sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
# Keras-metrics brings additional metrics: precision, recall, f1
install('keras-metrics')
import keras_metrics

print ('Keras ', tf.keras.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning-rate', type=float, default=0.1)
parser.add_argument('--batch-size', type=int, default=128)

parser.add_argument('--filters1', type=int, default=64)
parser.add_argument('--filters2', type=int, default=64)
parser.add_argument('--batch-norm', type=str, default='yes')
parser.add_argument('--bn-momentum', type=float, default=0.99)
parser.add_argument('--bn-epsilon', type=float, default=0.001)
parser.add_argument('--dropout-conv', type=float, default=0.2)
parser.add_argument('--num-fc', type=int, default=1)
parser.add_argument('--dropout-fc', type=float, default=0.2)

parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

args, _ = parser.parse_known_args()
epochs = args.epochs
lr = args.learning_rate
batch_size = args.batch_size

filters1 = args.filters1
filters2 = args.filters2
batch_norm = args.batch_norm
bn_momentum = args.bn_momentum
bn_epsilon = args.bn_epsilon
dropout_conv = args.dropout_conv
dropout_fc = args.dropout_fc

gpu_count = args.gpu_count
model_dir = args.model_dir
training_dir = args.training
validation_dir = args.validation
chk_dir = '/opt/ml/checkpoints'

# Load data set
x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']

img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)

# Normalize pixel values
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255

# Convert class vectors to binary class matrices
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# Check if checkpoints are available
checkpoints = sorted(glob.glob(os.path.join(chk_dir,'fmnist-cnn-*')))

if checkpoints :
    last_checkpoint = checkpoints[-1]
    last_epoch = int(last_checkpoint.split('-')[-1])
    model = load_model(last_checkpoint)
    print('Loaded checkpoint for epoch', last_epoch)
else:
    last_epoch = 0
    # Build model    
    model = Sequential()
    # 1st convolution block
    model.add(Conv2D(filters1, kernel_size=(3,3), padding='same', input_shape=(img_rows, img_cols, 1)))
    if batch_norm == 'yes' :
        model.add(BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Dropout(dropout_conv))
    # 2nd convolution block
    model.add(Conv2D(filters2, kernel_size=(3,3), padding='valid'))
    if batch_norm == 'yes' :
        model.add(BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Dropout(dropout_conv))
    # 1st fully connected block
    model.add(Flatten())
    model.add(Dense(256))
    if batch_norm == 'yes' :
        model.add(BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_fc))
    # 2nd fully connected block
    model.add(Dense(64))
    if batch_norm == 'yes' :
        model.add(BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_fc))
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
print(model.summary())

if gpu_count > 1:
    model = multi_gpu_model(model, gpus=gpu_count)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy',
                       keras_metrics.precision(), 
                       keras_metrics.recall(),
                       keras_metrics.f1_score()])

# Define callback to save best epoch
chk_name = 'fmnist-cnn-{epoch:04d}'
checkpointer = ModelCheckpoint(filepath=os.path.join(chk_dir,chk_name),
                               monitor='val_accuracy')

# Define callback for early stopping
early_stopping = EarlyStopping(monitor='val_accuracy',
                          min_delta=0,
                          patience=25,
                          verbose=1, mode='auto')

# Define custom callback to log best validation metric
class LogBestMetric(Callback):
    def on_train_begin(self, logs={}):
        self.val_accuracy = []
    def on_train_end(self, logs={}):
        print("Best val_accuracy:", max(self.val_accuracy))
    def on_epoch_end(self, batch, logs={}):
        self.val_accuracy.append(logs.get('val_accuracy'))

best_val_metric = LogBestMetric()

model.fit(x=x_train, 
          y=y_train, 
          batch_size=batch_size, 
          validation_data=(x_val, y_val), 
          epochs=epochs,
          callbacks=[checkpointer, early_stopping, best_val_metric],
          verbose=1,
          initial_epoch=last_epoch)

# save model for Tensorflow Serving
save_model(model, os.path.join(model_dir, '1'), save_format='tf')
