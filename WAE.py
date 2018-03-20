import tensorflow as tf
import numpy as np
import math

from random import shuffle
from time import time
from tools import save_pickle, load_pickle

# This model is modified from 
# https://github.com/tolstikhin/wae
# The original paper can be found at
# https://openreview.net/pdf?id=HkL7n1-0b

def glorot_init(in_dim, out_dim):
	c = np.sqrt(6.0 / (in_dim + out_dim))
	return tf.random_uniform([in_dim, out_dim], -c, c)

class WAE_MMD:
	def __init__(self, dims, loss, activation, z_dim, phase, scale, 
			batch_size = 64, batch_norm = True, noise = None, drop_out = None, 
			lr = 1e-4, eps = 1e-8, params = None):
		self.dims = dims
		self.noise = noise
		self.loss = loss
		self.act = activation
		self.z_dim = z_dim
		self.phase = phase
		self.batch_size = batch_size
		self.lr = lr
		self.eps = eps
		#self.wae_lambda = wae_lambda
		self.scale = scale
		self.drop_out = drop_out
		self.bn = batch_norm
		self.nb_layers = len(dims)-1
		self._set_weight(params)
		self._set_graph()
		self.fixed_noise = self.gen_sample(100)

	def activation(self, x):
		if self.act == 'sigmoid':
			return tf.nn.sigmoid(x)
		elif self.act == 'tanh':
			return tf.nn.tanh(x)
		elif self.act == 'linear':
			return x
		elif self.act == 'lrelu':
			return tf.maximum(x, 0.2 * x)

	def gen_sample(self, size, sphere = False):
		mean = np.zeros(self.z_dim)
		cov = np.identity(self.z_dim)
		noise = np.random.multivariate_normal(mean, cov, size).astype(np.float32)
		if sphere:
			noise = noise / np.sqrt(np.sum(noise * noise, axis = 1))[:, np.newaxis]
		#noise = noise[:, np.newaxis]
		return self.scale * noise

	def _set_weight(self, params):
		# Three parts: Encoder, Decoder, Generation
		self.encoder = list()
		self.decoder = list()
		self.en_bn = list()
		self.de_bn = list()

		# Encoder and Decoder
		for i in range(self.nb_layers):
			if params:
				self.encoder.append([tf.Variable(params[0][i][j]) for j in range(len(params[0][i]))])
				self.decoder.append([tf.Variable(params[1][i][j]) for j in range(len(params[1][i]))])
			else:
				W = tf.Variable(glorot_init(self.dims[i], self.dims[i+1]))
				b = tf.Variable(tf.random_uniform([self.dims[i+1]], 0.0, 0.0))
				self.encoder.append([W, b])
				de_W = tf.Variable(glorot_init(self.dims[-(i+1)], self.dims[-(i+2)]))
				de_b = tf.Variable(tf.random_uniform([self.dims[-(i+2)]], 0.0, 0.0))
				self.decoder.append([de_W, de_b])
				if self.bn:
					scale = tf.Variable(tf.ones([self.dims[i+1]]))
					shift = tf.Variable(tf.zeros([self.dims[i+1]]))
					self.encoder[i].extend([scale, shift])
					de_scale = tf.Variable(tf.ones([self.dims[-(i+2)]]))
					de_shift = tf.Variable(tf.zeros([self.dims[-(i+2)]]))
					self.decoder[i].extend([de_scale, de_shift])
			self.en_bn.append(tf.train.ExponentialMovingAverage(decay = 0.5))
			self.de_bn.append(tf.train.ExponentialMovingAverage(decay = 0.5))

		# Generator
		if params:
			self.gen = [tf.Variable(params[2][i]) for i in range(len(params[2]))]
		else:
			W_mean = tf.Variable(glorot_init(self.dims[-1], self.z_dim))
			b_mean = tf.Variable(tf.random_uniform([self.z_dim], 0.0, 0.0))
			W_log = tf.Variable(glorot_init(self.dims[-1], self.z_dim))
			b_log = tf.Variable(tf.random_uniform([self.z_dim], 0.0, 0.0))
			W_gen = tf.Variable(glorot_init(self.z_dim, self.dims[-1]))
			b_gen = tf.Variable(tf.random_uniform([self.dims[-1]], 0.0, 0.0))
			self.gen = [W_mean, b_mean, W_log, b_log, W_gen, b_gen]
			if self.bn:
				scale = tf.Variable(tf.ones([self.dims[-1]]))
				shift = tf.Variable(tf.zeros([self.dims[-1]]))
				self.gen.extend([scale, shift])
		self.gen_bn = tf.train.ExponentialMovingAverage(decay = 0.5)

	def _set_graph(self):
		# Inputs
		self.input = tf.placeholder(tf.float32, shape = [None, self.dims[0]])
		self.sample = tf.placeholder(tf.float32, shape = [None, self.z_dim])
		self.wae_lambda = tf.placeholder(tf.float32, shape = [None])
		self.pre_lambda = tf.placeholder(tf.float32, shape = [None])

		# Encode
		for i in range(self.nb_layers):
			if i == 0:
				self.h = self.activation(tf.add(tf.matmul(self.input, self.encoder[i][0]), self.encoder[i][1]))
			else:
				self.h = self.activation(tf.add(tf.matmul(self.h, self.encoder[i][0]), self.encoder[i][1]))
			if self.bn:
				if self.phase == 'train':
					fc_mean, fc_var = tf.nn.moments(self.h, axes = [0])
					self.en_bn[i].apply([fc_mean, fc_var])
				self.h = tf.nn.batch_normalization(self.h, self.en_bn[i].average(fc_mean), self.en_bn[i].average(fc_var), self.encoder[i][2], self.encoder[i][3], self.eps)


		# Generation
		self.z_mean = tf.add(tf.matmul(self.h, self.gen[0]), self.gen[1]) 
		self.z_log = tf.add(tf.matmul(self.h, self.gen[2]), self.gen[3])
		self.z_eps = tf.random_normal([self.batch_size, self.z_dim], 0.0, 1.0)
		#self.z = self.z_mean + tf.multiply(tf.sqrt(self.eps + tf.exp(self.z_log)), self.z_eps)
		self.z = self.z_mean + tf.multiply(self.eps + tf.exp(self.z_log / 2.), self.z_eps)
		self.de_h = tf.add(tf.matmul(self.z, self.gen[4]), self.gen[5])
		self.gen_h = tf.add(tf.matmul(self.sample, self.gen[4]), self.gen[5])
		if self.bn:
			if self.phase == 'train':
				fc_mean, fc_var = tf.nn.moments(self.de_h, axes = [0])
				self.gen_bn.apply([fc_mean, fc_var])
				self.de_h = tf.nn.batch_normalization(self.de_h, self.gen_bn.average(fc_mean), self.gen_bn.average(fc_var), self.gen[6], self.gen[7], self.eps)
				self.gen_h = tf.nn.batch_normalization(self.gen_h, self.gen_bn.average(fc_mean), self.gen_bn.average(fc_var), self.gen[6], self.gen[7], self.eps)


		# Decode
		for i in range(self.nb_layers - 1):
			self.de_h = self.activation(tf.add(tf.matmul(self.de_h, self.decoder[i][0]), self.decoder[i][1]))
			self.gen_h = self.activation(tf.add(tf.matmul(self.gen_h, self.decoder[i][0]), self.decoder[i][1]))
			if self.bn:
				if self.phase == 'train':
					fc_mean, fc_var = tf.nn.moments(self.de_h, axes = [0])
					self.de_bn[i].apply([fc_mean, fc_var])
				self.de_h = tf.nn.batch_normalization(self.de_h, self.de_bn[i].average(fc_mean), self.de_bn[i].average(fc_var), self.decoder[i][2], self.decoder[i][3], self.eps)
				self.gen_h = tf.nn.batch_normalization(self.gen_h, self.de_bn[i].average(fc_mean), self.de_bn[i].average(fc_var), self.decoder[i][2], self.decoder[i][3], self.eps)

		self.x_recon = tf.nn.sigmoid(tf.add(tf.matmul(self.de_h, self.decoder[-1][0]), self.decoder[-1][1]))
		self.x_gen = tf.nn.sigmoid(tf.add(tf.matmul(self.gen_h, self.decoder[-1][0]), self.decoder[-1][1]))

		# Pre-train Loss
		# From the original model
		self.mean_pz = tf.reduce_mean(self.sample, axis = 0, keep_dims = True)
		self.mean_qz = tf.reduce_mean(self.z, axis = 0, keep_dims = True)
		self.mean_loss = tf.reduce_mean(tf.square(self.mean_pz - self.mean_qz))
		self.cov_pz = tf.matmul(self.sample - self.mean_pz, self.sample - self.mean_pz, transpose_a = True) / self.batch_size
		self.cov_qz = tf.matmul(self.z - self.mean_qz, self.z - self.mean_qz, transpose_a = True) / self.z_dim / self.batch_size
		self.cov_loss = tf.reduce_mean(tf.square(self.cov_pz - self.cov_qz))
		self.pretrain_loss = self.mean_loss + self.cov_loss

		# Loss
		if self.loss == 'CE':
			self.recon_loss = -tf.reduce_mean(tf.multiply(self.input, tf.log(tf.maximum(self.x_recon, self.eps)))) - tf.reduce_mean(tf.multiply(1. - self.input, tf.log(tf.maximum(1. - self.x_recon, self.eps))))
		elif self.loss == 'WCE':
			self.b_w = tf.to_float(tf.size(self.input)) / tf.to_float(tf.reduce_sum(self.input))
			self.recon_loss = -self.b_w * tf.reduce_mean(tf.multiply(self.input, tf.log(tf.maximum(self.x_recon, self.eps)))) - tf.reduce_mean(tf.multiply(1. - self.input, tf.log(tf.maximum(self.eps, 1. - self.x_recon))))
		elif self.loss == 'L2':
			self.recon_loss = tf.reduce_sum(tf.square(self.x_recon - self.input), axis = 1)
			self.recon_loss = tf.reduce_mean(tf.sqrt(self.eps + self.recon_loss))
		elif self.loss == 'MSE':
			self.recon_loss = tf.reduce_sum(tf.square(self.x_recon - self.input), axis = 1)
			self.recon_loss = tf.reduce_mean(self.recon_loss)
		elif self.loss == 'L1':
			self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.x_recon - self.input), axis = 1))
		self.penalty = self.mmd_loss(pz = self.z, qz = self.sample)
		self.objective = self.recon_loss + self.wae_lambda * self.penalty + self.pre_lambda * self.pretrain_loss
		#self.pretrain_loss = self.recon_loss + 0.1 * (self.mean_loss + self.cov_loss)

		# Session & Optimizer
		self.optimizer = tf.train.AdamOptimizer(self.lr)
		#if self.phase == 'train':
		#	self.train_op = self.optimizer.minimize(self.objective)
		#if self.phase == 'pre':
		#	self.train_op = self.optimizer.minimize(self.pretrain_loss)
		self.train_op = self.optimizer.minimize(self.objective)
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def mmd_loss(self, qz, pz):
		sigma = 2 * self.scale
		n = tf.cast(self.batch_size, tf.int32)
		nf = tf.cast(self.batch_size, tf.float32)
		#denom = n * (n - 1) / 2

		norm_pz = tf.reduce_sum(tf.square(pz), axis = 1, keep_dims = True)
		prod_pz = tf.matmul(pz, pz, transpose_b = True)
		dist_pz = norm_pz + tf.transpose(norm_pz) - 2. * prod_pz

		norm_qz = tf.reduce_sum(tf.square(qz), axis = 1, keep_dims = True)
		prod_qz = tf.matmul(qz, qz, transpose_b = True)
		dist_qz = norm_qz + tf.transpose(norm_qz) - 2. * prod_qz

		prod = tf.matmul(qz, pz, transpose_b = True)
		dist = norm_qz + tf.transpose(norm_pz) - 2. * prod
		
		# Use inverse multiquadratic kernel 
		# k(x,y) = C / (C + || x - y ||^2)
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

	def save_weight(self, path):
		W = self.sess.run([self.encoder, self.decoder, self.gen])
		save_pickle(W, path)

	def fit(self, data, nb_epoch, w = [5, 0],  sample_path = None):
		#self._set_graph()
		self.phase = 'train'
		nb_data = len(data)
		indices = [i for i in range(nb_data)]
		nb_batch = math.ceil(nb_data / self.batch_size)
		loss_rec = [[], [], [], []]
		for epoch in range(nb_epoch):
			print('Epoch {}/{}:'.format(epoch + 1, nb_epoch))
			shuffle(indices)
			b_time = time()
			for i in range(nb_batch):
				idx = indices[i * self.batch_size:min((i+1) * self.batch_size, nb_data)]
				data_batch = data[idx]
				sample_batch = self.gen_sample(self.batch_size)
				_, loss, recon_loss, penalty, pre = self.sess.run([self.train_op, self.objective, self.recon_loss, self.penalty, self.pretrain_loss], feed_dict = {self.input: data_batch, self.sample: sample_batch, self.wae_lambda: [w[0]], self.pre_lambda: [w[1]]})
				loss_rec[0].append(loss)
				loss_rec[1].append(recon_loss)
				loss_rec[2].append(penalty)
				loss_rec[3].append(pre)
				print('%d/%d - Loss: %.5f, Recon.: %.5f, Pen.: %.5f, Pre.: %.5f     ' %(i+1, nb_batch, np.mean(loss_rec[0][-100:]), np.mean(loss_rec[1][-100:]), np.mean(loss_rec[2][-100:]), np.mean(loss_rec[3][-100:])), end = '\r')
			print('%.1f sec. - Loss: %.5f, Recon.: %.5f, Pen.: %.5f, Pre.: %.5f     ' %(time() - b_time, np.mean(loss_rec[0][-100:]), np.mean(loss_rec[1][-100:]), np.mean(loss_rec[2][-100:]), np.mean(loss_rec[3][-100:])))
			if sample_path and (epoch+1) % 10 == 0:
				self._gen_ins(sample_path + 'sample_{}.npy'.format(epoch + 1))

	def _get_h(self, data, batch_size = 2000):
		self.phase = 'gen' 
		nb_data = len(data)
		nb_batch = math.ceil(nb_data / batch_size)
		pred = list()
		for i in range(nb_batch):
			data_batch = data[i * batch_size: min(nb_data, (i+1) * batch_size)]
			ans = self.sess.run([self.h], feed_dict = {self.input: data_batch})
			pred.extend(list(ans[0]))
		return np.asarray(pred)

	def _gen_ins(self, title):
		self.phase = 'gen'
		image = self.sess.run([self.x_gen], feed_dict = {self.sample: self.fixed_noise})
		np.save(title, image)
		
if __name__ == "__main__":
	from keras.datasets import mnist

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.astype('float32')
	X_train = np.reshape(X_train, (X_train.shape[0], 784))
	X_train /= 255.
	model = WAE_MMD(
		dims = [784, 512, 64], 
		loss = 'L2', 
		activation = 'lrelu', 
		z_dim = 8, 
		phase = 'train', 
		scale = 1,
		batch_size = 50)
	print('Pre-train')
	model.fit(data = X_train, nb_epoch = 20, w = [0, 0.05])
	#print('Pre-train with regularizer')
	#model.fit(data = X_train, nb_epoch = 30, w = [5, 0])
	model.save_weight('pre_weight.pkl')

	model = WAE_MMD(
		dims = [784, 512, 64], 
		loss = 'L2', 
		activation = 'lrelu', 
		z_dim = 8, 
		phase = 'train', 
		scale = 1,
		batch_size = 50,
		params = load_pickle('pre_weight.pkl'))
	model.fit(data = X_train, nb_epoch = 300, w = [5000, 0], sample_path ='samples/mnist_')

