from keras.models import Model
from keras.layers import Input, PReLU, Activation, Lambda
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization as BN
from keras import backend as K
from keras import metrics

from random import shuffle
import numpy as np
import math

def encoder(input_dim, dims, batch_norm):
	nb_layers = len(dims)
	x = Input(shape = (input_dim,))
	for i in range(nb_layers):
		if i == 0:
			h = PReLU()(Dense(dims[i])(x))
			if batch_norm:
				h = BN()(h)
		else:
			h = PReLU()(Dense(dims[i])(h))
			if batch_norm:
				h = BN()(h)
	return Model(inputs = [x], outputs = [h])

def decoder(output_dim, z_dim, dims, batch_norm):
	x = Input(shape = (z_dim,))
	nb_layers = len(dims)
	for i in range(nb_layers):
		if i == 0:
			h = PReLU()(Dense(dims[-(i+1)])(x))
			if batch_norm:
				h = BN()(h)
		else:
			h = PReLU()(Dense(dims[-(i+1)])(h))
			if batch_norm:
				h = BN()(h)
	recon = Dense(output_dim, activation = 'sigmoid')(h)
	return Model(inputs = [x], outputs = [recon])

# Sample generator
def gen(args):
	z_dim = 16
	mean, var = args
	eps = K.random_normal(shape = (K.shape(mean)[0], z_dim), mean = 0., stddev = 1.0)
	return mean + K.exp(var / 2) * eps

def mmd_loss(self, qz, pz):
	sigma = 2 * self.scale
	n = tf.cast(self.batch_size, tf.int32)
	nf = tf.cast(self.batch_size, tf.float32)

	norm_pz = tf.reduce_sum(tf.square(pz), axis = 1, keep_dims = True)
	prod_pz = tf.matmul(pz, pz, transpose_b = True)
	dist_pz = norm_pz + tf.transpose(norm_pz) - 2. * prod_pz
	
	norm_qz = tf.reduce_sum(tf.square(qz), axis = 1, keep_dims = True)
	prod_qz = tf.matmul(qz, qz, transpose_b = True)
	dist_qz = norm_qz + tf.transpose(norm_qz) - 2. * prod_qz

	prod = tf.matmul(qz, pz, transpose_b = True)
	dist = norm_qz + tf.transpose(norm_pz) - 2. * prod
	
	base = 2 * self.z_dim * sigma
	stat = 0.
	for s in [.1, .2, .5, 1., 2., 5., 10.]:
		C = base * s
		res1 = C / (C + dist_qz) + C / (C + dist_pz)
		res1 = tf.multiply(res1, 1. - tf.eye(n))
		res1 = tf.reduce_sum(res1) / (nf * nf - nf)
		res2 = tf.reduce_sum(C / (C + dist)) * 2. / (nf * nf)
		stat += res1 - res2
	return stat

def mmd(z, sample, scale, z_dim):
	sigma = 2. * scale
	#n = K.shape(x)[0]
	n = 50
	norm_pz = K.sum(K.square(z), axis = 1, keepdims = True)
	prod_pz = K.dot(z, K.transpose(z))
	dist_pz = norm_pz + K.transpose(norm_pz) - 2. * prod_pz
	norm_qz = K.sum(K.square(sample), axis = 1, keepdims = True)
	prod_qz = K.dot(sample, K.transpose(sample))
	dist_qz = norm_qz + K.transpose(norm_qz) - 2. * prod_qz
	prod_z = K.dot(z, K.transpose(sample))
	dist = norm_qz + K.transpose(norm_pz) - 2. * prod_z
	base = 2. * z_dim * sigma
	for s in [.1, .2, .5, 1., 2., 5., 10.]:
		C = base * s
		res1 = C / (C + dist_qz) + C / (C + dist_pz)
		#res1 = tf.multiply(res1, 1. - tf.eye(n))
		res1 = res1 * (K.ones(shape = (n, n), dtype = 'float32') - K.eye(n, dtype = 'float32'))
		res1 = K.sum(res1) / (n * n - n)
		res2 = K.sum(C / (C + dist)) * 2. / (n * n)
		if s == .1:
			stat = res1 - res2
		else:
			stat += res1 - res2
	return stat

def WAE(input_dim, dims, z_dim, batch_norm = True, scale = 1.):
	enc = encoder(input_dim, dims, batch_norm)
	dec = decoder(input_dim, z_dim, dims, batch_norm)

	x = Input(shape = (input_dim,))
	sample = Input(shape = (z_dim,))
	h = enc(x)
	z_mean = Dense(z_dim)(h)
	z_var = Dense(z_dim)(h)
	z = Lambda(gen, output_shape = (z_dim,))([z_mean, z_var])
	x_recon = dec(z)
	x_sample = dec(sample)

	Pre = Model(inputs = [x], outputs = x_recon)
	wae = Model(inputs = [x, sample], outputs = x_recon)
	gene = Model(inputs = [sample], outputs = x_sample)

	# For mmd loss
	sigma = 2. * scale
	#n = K.shape(x)[0]
	n = 50
	norm_pz = K.sum(K.square(z), axis = 1, keepdims = True)
	prod_pz = K.dot(z, K.transpose(z))
	dist_pz = norm_pz + K.transpose(norm_pz) - 2. * prod_pz
	norm_qz = K.sum(K.square(sample), axis = 1, keepdims = True)
	prod_qz = K.dot(sample, K.transpose(sample))
	dist_qz = norm_qz + K.transpose(norm_qz) - 2. * prod_qz
	prod_z = K.dot(z, K.transpose(sample))
	dist = norm_qz + K.transpose(norm_pz) - 2. * prod_z
	base = 2. * z_dim * sigma
	for s in [.1, .2, .5, 1., 2., 5., 10.]:
		C = base * s
		res1 = C / (C + dist_qz) + C / (C + dist_pz)
		#res1 = tf.multiply(res1, 1. - tf.eye(n))
		res1 = res1 * (K.ones(shape = (n, n), dtype = 'float32') - K.eye(n, dtype = 'float32'))
		res1 = K.sum(res1) / (n * n - n)
		res2 = K.sum(C / (C + dist)) * 2. / (n * n)
		if s == .1:
			stat = res1 - res2
		else:
			stat += res1 - res2
	recon_loss = input_dim * metrics.binary_crossentropy(x, x_recon)
	train_loss = recon_loss + 10. * stat

	Pre.add_loss(recon_loss)
	wae.add_loss(train_loss)

	return Pre, wae, gene

# Generatoe for training
def gen_sample(data, batch_size, z_dim, scale = 1., sphere = False):
	nb_data = len(data)
	nb_batch = math.ceil(nb_data / batch_size)
	indices = [i for i in range(nb_data)]
	while True:
		shuffle(indices)
		for i in range(nb_batch):
			idx = indices[i * batch_size:min(nb_data, (i+1) * batch_size)]
			x = data[idx]

			mean = np.zeros(z_dim)
			cov = np.identity(z_dim)
			noise = np.random.multivariate_normal(mean, cov, len(x)).astype(np.float32)
			if sphere:
				noise = noise / np.sqrt(np.sum(noise * noise, axis = 1))[:, np.newaxis]
			yield [x, noise], None

