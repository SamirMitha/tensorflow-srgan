import numpy as np
import tensorflow as tf

def normalize(img):
	norm_img = img / 255
	return norm_img


def pixel_shuffle(scale):
	return lambda x: tf.nn.depth_to_space(x, scale)


def normalize_tanh(img):
	norm_img = (img / (255 / 2)) - 1
	return norm_img


def denormalize_tanh(img):
	denorm_img = (img + 1) * (255 / 2)
	return denorm_img