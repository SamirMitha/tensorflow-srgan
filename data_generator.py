import numpy as np
import scipy.io as sio
import glob, os
from skimage import transform

from utils import normalize

class DataGenerator(Sequence):

	def __init__(self, 
				lr, 
				hr, 
				batch_size=4, 
				crop=True, 
				shuffle=True, 
				lr_size=(24,24,3),
				hr_size=(96,96,3)):

		# Initializations
		super(DataGenerator, self).__init__()
		self.img_ids = img_ids
		self.lr_path = lr_path
		self.hr_path = hr_path
		self.batch_size = batch_size
		self.crop = crop
		self.shuffle = shuffle
		self.lr_size = lr_size
		self.hr_size = hr_size
		self.on_epoch_end()

	def __len__(self):

		return int(np.floor(len(self.lr) / self.batch_size))

	def __getitem__(self, index):

		# Generates indexes of the batched data
		indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

		# Get list of IDs
		batch_IDs = [self.img_ids[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(batch_IDs)
		return X, y

	def on_epoch_end(self):

		self.indexes = np.arange(len(self.img_IDs))

		if(self.shuffle):
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		lr = np.empty((self.batch_size, *self.lr_size, 1))
		hr = np.empty((self.batch_size, *self.hr_size, 1))

		# crop and fill
		# unfinished