from __future__ import print_function

import keras
from keras import layers
from keras import backend as K
from keras.layers import Flatten, concatenate, UpSampling2D, Reshape
from keras.initializers import Constant, glorot_uniform
from keras.optimizers import Adam, SGD
from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D, ZeroPadding3D
from keras.layers.core import Activation, Dense, Dropout, Reshape, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ProgbarLogger, BaseLogger

import numpy as np

class Keras_Model(object):
	def __init__(self, patch_size=28, num_instances=2, num_classes=2, learning_rate=1e-4, num_bins=None, num_features=10, batch_size=None):
		self._lr_rate=learning_rate
		self._num_features = num_features

		kernel_kwargs = {
			'kernel_initializer': glorot_uniform(),
			'kernel_regularizer': None
		}

		patch_input = Input((28, 28, 1))
		x0 = Conv2D(16, (3, 3), padding='same', bias_initializer=Constant(value=0.1), **kernel_kwargs)(patch_input)
		x0 = self.wide_residual_blocks(x0, 2, 1, 16)
		x0 = self.wide_residual_blocks(x0, 4, 1, 16, True)
		x0 = self.wide_residual_blocks(x0, 8, 1, 16, True)
		x0 = Activation('relu')(x0)
		x0 = Flatten()(x0)
		print('flatten shape:{}'.format(K.int_shape(x0)))
		patch_output = Dense(self._num_features, activation='sigmoid', use_bias=False, name='fc_sigmoid', **kernel_kwargs)(x0)
		self._patch_model = Model(inputs=patch_input, outputs=patch_output)
		'''
		patch_model : input (28,28,1) -> output (10)
		
		but why is this done? DONT TELL ME THAT THESE ARE THE NUMBER OF LATENT FEATURES!!! only 10!!!!
		'''
		
		feature_input = Input((self._num_features,))
		x1 = Dense(7*7*128, use_bias=True, bias_initializer=Constant(value=0.1), **kernel_kwargs)(feature_input)
		x1 = Reshape((7,7,128))(x1)
		x1 = self.wide_residual_blocks_reverse(x1, 8, 1, 16, True)
		x1 = self.wide_residual_blocks_reverse(x1, 4, 1, 16, True)
		x1 = self.wide_residual_blocks_reverse(x1, 2, 1, 16)
		x1 = Activation('relu')(x1)
		reconstructed = Conv2D(1, (3, 3), padding='same', bias_initializer=Constant(value=0.1), **kernel_kwargs)(x1)
		print('reconstructed shape:{}'.format(K.int_shape(reconstructed)))
		# This is basically the decoder part
		self._image_generation_model = Model(inputs=feature_input, outputs=reconstructed)

		ae_output = self._image_generation_model(patch_output)
		'''
		patch_input and ae_output here are (28,28,1)	
		'''
		self._autoencoder_model = Model(inputs=patch_input, outputs=ae_output)

		input_list = list()
		output_list = list()
		ae_output_list = list()
		#In our case num_instances is 10 ( i.e 10 images in a bag!)
		for i in range(num_instances): #Num instances is just 2 !!? WTH is this??
			temp_input = Input(shape=(patch_size, patch_size, 1))
			temp_output = self._patch_model(temp_input) #Shape is (10) ( In our case it will be the number of latent features L)
			temp_output = Reshape((1,-1))(temp_output) # shape is (1,10) ( In our case it will be the number of latent features (1,L))
			# print('temp_output shape:{}'.format(K.int_shape(temp_output)))

			temp_ae_output = self._autoencoder_model(temp_input)
			temp_ae_output = Reshape((1,patch_size, patch_size, 1))(temp_ae_output)

			input_list.append(temp_input)
			output_list.append(temp_output)
			ae_output_list.append(temp_ae_output)

		'''
		output_list is of shape (num_instances, 1, num_features) -> (2,1,10) (or rather 2 elements each of shape (1,10) -> (Bag,1,J)
		'''

		'''
		output_list (Bag,1,J) -> [(1,J),....] {bag no of times}
		'''
		concatenated = layers.concatenate(output_list,axis=1) # shape  (1,J*Bag) ( J is 10 , bag is 2 from terminal example!)
		'''
		concatenated should be of shape (1,20) -> (1, 10* num_instances) or rather (1, num_features* num_instances)
		'''
		print('concatenated shape:{}'.format(K.int_shape(concatenated)))

		ae_concatenated = layers.concatenate(ae_output_list,axis=1)
		print('ae_concatenated shape:{}'.format(K.int_shape(ae_concatenated)))

		y = layers.Lambda(self.kde, arguments={'num_nodes':num_bins,'sigma':0.1,'batch_size':batch_size, 'num_features':self._num_features})(concatenated)
		print('y shape:{}'.format(K.int_shape(y)))

		y1 = Dense(384, activation='relu', name='fc_relu1', **kernel_kwargs)(y)
		y1 = Dense(192, activation='relu', name='fc_relu2', **kernel_kwargs)(y1)
		out = Dense(num_classes, activation='softmax', name='fc_softmax', **kernel_kwargs)(y1)


		self._classification_model = Model(inputs=input_list, outputs=[out,ae_concatenated])
		
		self._ucc_model = Model(inputs=input_list, outputs=out)
	
		optimizer=Adam(lr=learning_rate)
		self._classification_model.compile(optimizer=optimizer, loss=['categorical_crossentropy','mse'], metrics=['accuracy'], loss_weights=[0.5, 0.5])
		
		self._distribution_model = Model(inputs=input_list, outputs=y)

		self._features_model = Model(inputs=input_list, outputs=concatenated)


	@property
	def metrics_names(self):
		return self._classification_model.metrics_names
	
	@property
	def yaml_file(self):
		return self._classification_model.to_yaml()

	'''
	(1, J*Bag)
		
	num_nodes = 11 ( num_classes +1)
	sigma = 0.1
	batch_size = 512!!
	num_features = J (In the case of mnist it is 10)
	'''
	def kde(self, data, num_nodes=None, sigma=None, batch_size=None, num_features=None):
		# print('kde data shape:{}'.format(K.int_shape(data)))
		# print('num_nodes:{}'.format(num_nodes))
		# print('sigma:{}'.format(sigma))

		linspace = np.linspace(0, 1, num=num_nodes) # 11 points between 0,1 [0,.........,1] (11)
		data_ = [batch_size, K.int_shape(data)[1], 1] #shape (B, J*Bag, 1)
		tile = np.tile(linspace, data_) # shape (B, J*Bag, 1*11)
		k_sample_points = K.constant(tile) # shape (B, J*Bag, 1*11)
		# print('kde k_sample_points shape:{}'.format(K.int_shape(k_sample_points)))

		k_alfa = K.constant(1/np.sqrt(2*np.pi*np.square(sigma))) #alpha
		k_beta = K.constant(-1/(2*np.square(sigma))) #beta

		out_list = list()
		for i in range(num_features): #J
			# shape (B, J*Bag, 1) -> (B, Bag, J) ( lets assume its this!)
			i_ = data[:, :, i] # with the assuumption shape (B, Bag) -> jth feature of the Bag of batch no B
			shape_data_ = (-1, K.int_shape(data)[1], 1)
			temp_data = K.reshape(i_, shape_data_) #figure it out from the next line!
			# print('temp_data shape:{}'.format(K.int_shape(temp_data)))

			k_diff = k_sample_points - K.tile(temp_data,[1,1,num_nodes]) #(B, J*Bag, 1*11) - (B, J* Bag, 1*11)
			k_diff_2 = K.square(k_diff) #(B, J*Bag, 1*11)
			# print('k_diff_2 shape:{}'.format(K.int_shape(k_diff_2)))

			k_result = k_alfa * K.exp(k_beta*k_diff_2) #(B, J*Bag, 1*11)

			k_out_unnormalized = K.sum(k_result,axis=1) #(B, J*Bag, 1*11) -> (B, 1*11)

			k_sum = K.sum(k_out_unnormalized, axis=1)#(B, 1*11) -> (B)
			k_norm_coeff = K.reshape(k_sum, (-1, 1)) # (B,1)

			unnormalized_ = [1, K.int_shape(k_out_unnormalized)[1]] # [1, 1*11]
			k_tile = K.tile(k_norm_coeff, unnormalized_) #(B, 11)
			k_out = k_out_unnormalized / k_tile #(B,11)
			# print('k_out shape:{}'.format(K.int_shape(k_out)))

			out_list.append(k_out)

		'''
		out_list : [(B,11),.....J times] -> (J, B, 11)
		'''
		concat_out = K.concatenate(out_list,axis=-1) # shape here (B, J*11)
		# print('concat_out shape:{}'.format(K.int_shape(concat_out)))
		
		return concat_out


	def residual_zero_padding_block(self, x0, filters, first_unit=False, down_sample=False):
		residual_kwargs = {
			'kernel_size': (3, 3),
			'padding': 'same',
			'use_bias': True,
			'bias_initializer':Constant(value=0.1),
			'kernel_initializer': glorot_uniform(),
			'kernel_regularizer': None
		}
		skip_kwargs = {
			'kernel_size': (1, 1),
			'padding': 'valid',
			'use_bias': True,
			'kernel_initializer': glorot_uniform(),
			'kernel_regularizer': None
		}

		if first_unit:
			x0 = Activation('relu')(x0)
			if down_sample:
				residual_kwargs['strides'] = (2, 2)
				skip_kwargs['strides'] = (2, 2)
			x1 = Conv2D(filters, **residual_kwargs)(x0)
			x1 = Activation('relu')(x1)
			residual_kwargs['strides'] = (1, 1)
			x1 = Conv2D(filters, **residual_kwargs)(x1)
			x0_img_shape = x0.shape.as_list()[1:-1]
			x1_img_shape = x1.shape.as_list()[1:-1]
			print('x0_img_shape:{}'.format(x0_img_shape))
			print('x1_img_shape:{}'.format(x1_img_shape))
			print('x0 shape:{}'.format(K.int_shape(x0)))
			print('x1 shape:{}'.format(K.int_shape(x1)))
			x0_filters = x0.shape.as_list()[-1]
			x1_filters = x1.shape.as_list()[-1]
			if x0_img_shape != x1_img_shape:
				x0 = Conv2D(x0_filters, **skip_kwargs)(x0)
			if x0_filters != x1_filters:
				target_shape = (x1_img_shape[0], x1_img_shape[1], x0_filters, 1)
				x0 = Reshape(target_shape)(x0)
				zero_padding_size = x1_filters - x0_filters
				x0 = ZeroPadding3D(((0, 0), (0, 0), (0, zero_padding_size)))(x0)
				target_shape = (x1_img_shape[0], x1_img_shape[1], x1_filters)
				x0 = Reshape(target_shape)(x0)
		else:
			x1 = Activation('relu')(x0)
			x1 = Conv2D(filters, **residual_kwargs)(x1)
			x1 = Activation('relu')(x1)
			x1 = Conv2D(filters, **residual_kwargs)(x1)
		x0 = Add()([x0, x1])
		return x0


	def wide_residual_blocks(self, x0, filters, n, k, down_sample=False):
		for i in range(n):
			if i == 0:
				x0 = self.residual_zero_padding_block(x0, filters * k, True, down_sample)
			else:
				x0 = self.residual_zero_padding_block(x0, filters * k)
		return x0

	def wide_residual_blocks_reverse(self, x0, filters, n, k, up_sample=False):
		for i in range(n):
			if i == 0:
				x0 = self.residual_zero_padding_block_reverse(x0, filters * k, True, up_sample)
			else:
				x0 = self.residual_zero_padding_block_reverse(x0, filters * k)
		return x0

	def residual_zero_padding_block_reverse(self, x0, filters, first_unit=False, up_sample=False):
		residual_kwargs = {
			'kernel_size': (3, 3),
			'padding': 'same',
			'use_bias': True,
			'bias_initializer':Constant(value=0.1),
			'kernel_initializer': glorot_uniform(),
			'kernel_regularizer': None
		}
		skip_kwargs = {
			'kernel_size': (1, 1),
			'padding': 'valid',
			'use_bias': True,
			'kernel_initializer': glorot_uniform(),
			'kernel_regularizer': None
		}

		if first_unit:
			x0 = Activation('relu')(x0)
			if up_sample:
				x0 = UpSampling2D((2, 2))(x0)
			x1 = Conv2D(filters, **residual_kwargs)(x0)
			x1 = Activation('relu')(x1)
			x1 = Conv2D(filters, **residual_kwargs)(x1)
			print('x0 shape:{}'.format(K.int_shape(x0)))
			print('x1 shape:{}'.format(K.int_shape(x1)))
			x0_filters = x0.shape.as_list()[-1]
			x1_filters = x1.shape.as_list()[-1]
			if x0_filters != x1_filters:
				x0 = Conv2D(filters, **skip_kwargs)(x0)			
		else:
			x1 = Activation('relu')(x0)
			x1 = Conv2D(filters, **residual_kwargs)(x1)
			x1 = Activation('relu')(x1)
			x1 = Conv2D(filters, **residual_kwargs)(x1)
		x0 = Add()([x0, x1])
		
		return x0

	def train_on_batch_data(self, batch_inputs=None, batch_outputs=None):
		stats = self._classification_model.train_on_batch(batch_inputs, batch_outputs)

		return stats

	def test_on_batch_data(self, batch_inputs=None, batch_outputs=None):
		stats = self._classification_model.test_on_batch(batch_inputs, batch_outputs)

		return stats

	def predict_on_batch_data(self, batch_inputs=None):
		predicted_label = self._classification_model.predict_on_batch(batch_inputs)

		return predicted_label

	def predict_ucc_on_batch_data(self, batch_inputs=None):
		predicted_label = self._ucc_model.predict_on_batch(batch_inputs)

		return predicted_label

	def predict_on_batch_data_ae(self, batch_inputs=None):
		predicted_label = self._autoencoder_model.predict_on_batch(batch_inputs)

		return predicted_label

	def generate_image_from_feature(self, batch_inputs=None):
		predicted_label = self._image_generation_model.predict_on_batch(batch_inputs)

		return predicted_label

	def predict_on_batch_data_distribution(self, batch_inputs=None):
		predicted_dist = self._distribution_model.predict_on_batch(batch_inputs)

		return predicted_dist

	def predict_on_batch_data_features(self, batch_inputs=None):
		predicted_features = self._features_model.predict_on_batch(batch_inputs)

		return predicted_features

	def predict_on_batch_data_patches(self, batch_inputs=None):
		predicted_patches = self._patch_model.predict_on_batch(batch_inputs)

		return predicted_patches

	def save_model_weights(self, model_weight_save_path=None):
		self._classification_model.save_weights(model_weight_save_path)
		self._autoencoder_model.save_weights(model_weight_save_path[:-3]+'__ae.h5')

	def load_saved_weights(self, weights_path=None):
		self._classification_model.load_weights(weights_path)
		self._autoencoder_model.load_weights(weights_path[:-3]+'__ae.h5')


