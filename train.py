# import glob, os
# import time
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import numpy as np
from models import Generator, Discriminator, FeatureExtractor
# import scipy.io as sio
# import pickle
# from models import FlowRegAffine
# from losses import correlation_loss
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN
# from data_generator import DataGenerator
# import matplotlib.pyplot as plt
# from utils import normalize

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

# Defaults
batch_size = 4
epochs = 2
learning_rate = 1e-4
model_loss = 'mse'
monitor = 'val_loss'
validation_split = 0.1
lr_size = 24
hr_size = 96

# Directories
train_lr_path = '/media/samir/Secondary/Datasets/DIV2K/DIV2K_train_LR_bicubic/X4/'
train_hr_path = '/media/samir/Secondary/Datasets/DIV2K/DIV2K_train_HR/'
val_lr_path = '/media/samir/Secondary/Datasets/DIV2K/DIV2K_valid_LR_bicubic/X4/'
val_hr_path = '/media/samir/Secondary/Datasets/DIV2K/DIV2K_valid_HR/'
model_path = '/media/samir/Secondary/SRGANs/SRGAN/models'
model_name = 'SRGAN_Default'

# Create output directories
if(not os.path.isdir(model_path) or not os.listdir(model_path)):
	os.makedirs(model_path + '/logs')
	os.makedirs(model_path + '/models')
	os.makedirs(model_path + '/history')
	os.makedirs(model_path + '/figures')
	os.makedirs(model_path + '/params')
	os.makedirs(model_path + '/checkpoints')

# Create train list
train_lr_imgs = glob.glob(train_lr_path + '/*.png')
train_hr_imgs = glob.glob(train_hr_path + '/*.png')
num_imgs = len(train_lr_imgs)
idx = np.arange(num_imgs)
np.random.shuffle(idx)
train_ids = idx

# Create validation list
val_lr_imgs = glob.glob(val_lr_path + '/*.png')
val_hr_imgs = glob.glob(val_hr_path + '/*.png')
v_num_imgs = len(val_lr_imgs)
idx = np.arange(v_num_imgs)
np.random.shuffle(idx)
val_ids = idx

# Create tensors
train_gen = DataGenerator(img_ids=train_ids, lr_path=train_lr_imgs, hr_path=train_hr_path)
val_gen = DataGenerator(img_ids=val_ids, lr_path=val_lr_path, hr_path=val_hr_path)

# Instantiate Model
Generator_Netor = Generator()
generator = Generator_Net.get_model()

# Model Summary
print(generator.summary)

# Set Optimizer
optimizer = Adam(learning_rate=learning_rate)

# Set loss function
loss = MeanSquaredError()

# unfinished