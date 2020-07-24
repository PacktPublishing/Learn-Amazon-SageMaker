import tensorflow as tf
import numpy as np
import argparse, os

from model import FMNISTModel

print("TensorFlow version", tf.__version__)

# Process command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=128)

parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

args, _ = parser.parse_known_args()

epochs     = args.epochs
lr         = args.learning_rate
batch_size = args.batch_size

gpu_count  = args.gpu_count
model_dir  = args.model_dir
training_dir   = args.training
validation_dir = args.validation

# Load data set
x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']

# Add extra dimension for channel: (28,28) --> (28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_val   = x_val[..., tf.newaxis]

# Prepare training and validation iterators
#  - define batch size
#  - normalize pixel values to [0,1]
#  - one-hot encode labels
preprocess = lambda x, y: (tf.divide(tf.cast(x, tf.float32), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10)))

if (gpu_count > 1):
    batch_size *= gpu_count
    
train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
train = train.map(preprocess)
train = train.repeat()

val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
val = val.map(preprocess)
val = val.repeat()

# Build model
model = FMNISTModel()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
train_steps = x_train.shape[0] / batch_size
val_steps   = x_val.shape[0] / batch_size

model.fit(train, epochs=epochs, steps_per_epoch=train_steps, validation_data=val, validation_steps=val_steps)

# save model for Tensorflow Serving
model.save(os.path.join(model_dir, '1'))
   