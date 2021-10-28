from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Conv2D, Add, BatchNormalization, ReLU, LeakyReLU, PReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from utils import normalize, pixel_shuffle, normalize_tanh, denormalize_tanh

class Generator(Model):
	def __init__(self,
				LR_SIZE=24,
				HR_SIZE=96,
				kernel_size=[3,9],
				resnet_filters=[64, 256],
				resnet_blocks=16,
				resnet_strides=1,
				normalization_momentum=0.8
				):

	# Initializations
		super(Generator, self).__init__()
		self.LR_SIZE = LR_SIZE
		self.HR_SIZE = HR_SIZE
		self.kernel_size = kernel_size
		self.resnet_filters = resnet_filters
		self.resnet_blocks = resnet_blocks
		self.resnet_strides = resnet_strides
		self.normalization_momentum = normalization_momentum


	def get_model(self):
		return self.__forward()


	def upsampling(self, input_tensor, resnet_filters, kernel_size):
		conv1 = Conv2D(filters=resnet_filters, kernel_size=kernel_size[0], padding='same')(input_tensor)
		lambda1 = Lambda(pixel_shuffle(scale=2))(conv1)
		prelu1 = PReLU(shared_axes=[1, 2])(lambda1)
		return prelu1


	def residual_block(self, input_tensor, resnet_filters, resnet_strides, kernel_size, normalization_momentum):
		conv1 = Conv2D(filters=resnet_filters, kernel_size=kernel_size[0], padding='same')(input_tensor)
		norm1 = BatchNormalization(momentum=normalization_momentum)(conv1)
		prelu1 = PReLU(shared_axes=[1, 2])(norm1)
		conv2 = Conv2D(filters=resnet_filters, kernel_size=kernel_size[0], padding='same')(prelu1)
		norm2 = BatchNormalization(momentum=normalization_momentum)(conv2)
		add1 = Add()([input_tensor, norm2])
		return(norm2)


	def resnet(self, input_tensor, resnet_filters, resnet_blocks, resnet_strides, kernel_size, normalization_momentum):
		norm_img = Lambda(normalize)(input_tensor)
		conv1 = Conv2D(filters=resnet_filters[0], kernel_size=kernel_size[1], padding='same')(norm_img)
		prelu1 = PReLU(shared_axes=[1, 2])(conv1)
		prelun = prelu1

		for _ in range(resnet_blocks):
			prelun = self.residual_block(prelun, resnet_filters[0], resnet_strides, kernel_size, normalization_momentum)

		conv2 = Conv2D(filters=resnet_filters[0], kernel_size=kernel_size[0], padding='same')(prelun)
		norm1 = BatchNormalization()(conv2)
		add1 = Add()([prelu1, conv2])
		upsample1 = self.upsampling(add1, resnet_filters[1], kernel_size)
		upsample2 = self.upsampling(upsample1, resnet_filters[1], kernel_size)
		conv3 = Conv2D(3, kernel_size=kernel_size[1], padding='same', activation='tanh')(upsample2)
		lambda1 = Lambda(denormalize_tanh)(conv3)
		return(lambda1)


	def __forward(self):
		inputs = Input(shape=(None, None, 3))
		generator_resnet = self.resnet(inputs, self.resnet_filters, self.resnet_blocks, self.resnet_strides, self.kernel_size, self.normalization_momentum)
		outputs = generator_resnet
		model = Model(inputs=inputs, outputs=outputs)
		return (model)


class FeatureExtractor(Model):
	def get_model(self, output_layer):
		# Use 5 for VGG22
		# Use 20 for VGG54
		return self.__forward(output_layer)


	def __forward(self, output_layer):
		vgg = VGG19(input_shape=(None, None, 3), include_top=False)
		return Model(vgg.input, vgg.layers[output_layer].output)


class Discriminator(Model):
	def __init__(self,
				LR_SIZE=24,
				HR_SIZE=96,
				kernel_size=3,
				discriminator_filters=[64, 128, 256, 512],
				discriminator_strides=[1,2],
				normalization_momentum=0.8
				):

		# Initializations
		super(Discriminator, self).__init__()
		self.LR_SIZE = LR_SIZE
		self.HR_SIZE = HR_SIZE
		self.kernel_size = kernel_size
		self.discriminator_filters = discriminator_filters
		self.discriminator_strides = discriminator_strides
		self.normalization_momentum = normalization_momentum


	def get_model(self):
		return self.__forward()


	def discriminator_block(self, input_tensor, discriminator_filters, discriminator_strides, kernel_size, normalization_momentum, normalization_flag=True):
		conv1 = Conv2D(filters=discriminator_filters, kernel_size=kernel_size, padding='same')(input_tensor)
		if normalization_flag == True:
			conv1 = BatchNormalization(momentum=normalization_momentum)(conv1)
		return(conv1)


	def discriminator_network(self, input_tensor, discriminator_filters, discriminator_strides, kernel_size, normalization_momentum):
		lambda1 = Lambda(normalize_tanh)(input_tensor)
		disc_block1 = self.discriminator_block(lambda1, discriminator_filters[0], discriminator_strides[0], kernel_size, normalization_momentum, normalization_flag=False)
		disc_block2 = self.discriminator_block(disc_block1, discriminator_filters[0], discriminator_strides[1], kernel_size, normalization_momentum)

		disc_block3 = self.discriminator_block(disc_block2, discriminator_filters[1], discriminator_strides[0], kernel_size, normalization_momentum)
		disc_block4 = self.discriminator_block(disc_block3, discriminator_filters[1], discriminator_strides[1], kernel_size, normalization_momentum)

		disc_block5 = self.discriminator_block(disc_block4, discriminator_filters[2], discriminator_strides[0], kernel_size, normalization_momentum)
		disc_block6 = self.discriminator_block(disc_block5, discriminator_filters[2], discriminator_strides[1], kernel_size, normalization_momentum)

		disc_block7 = self.discriminator_block(disc_block6, discriminator_filters[3], discriminator_strides[0], kernel_size, normalization_momentum)
		disc_block8 = self.discriminator_block(disc_block7, discriminator_filters[3], discriminator_strides[1], kernel_size, normalization_momentum)

		flat1 = Flatten()(disc_block8)
		dense1 = Dense(1024)(flat1)
		leakyrelu1 = LeakyReLU(alpha=0.2)(dense1)
		dense2 = Dense(1, activation='sigmoid')(leakyrelu1)
		return (dense2)


	def __forward(self):
		inputs = Input(shape=(self.HR_SIZE, self.LR_SIZE, 3))
		discriminator_net = self.discriminator_network(inputs, self.discriminator_filters, self.discriminator_strides, self.kernel_size, self.normalization_momentum)
		outputs = discriminator_net
		model = Model(inputs=inputs, outputs=outputs)
		return(model)


if __name__ == '__main__':
	# Generator = Generator()
	# model = Generator.get_model()
	# print(model.summary())

	# FeatureExtractor = FeatureExtractor()
	# model = FeatureExtractor.get_model(20)
	# print(model.summary())

	Discriminator = Discriminator()
	model = Discriminator.get_model()
	print(model.summary())