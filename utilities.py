import matplotlib.pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf

from SCAE.capsules import primary
from SCAE.capsules.attention import SetTransformer
from SCAE.capsules.models.scae import ImageAutoencoder, ImageCapsule
from SCAE.tools.model import _ModelCollector, ScaeBasement, _stacked_capsule_autoencoder as _scae
from SCAE.tools.utilities import DatasetHelper
from SCAE.train import Configs

Configs.GTSRB_DATASET_PATH = './SCAE/datasets/GTSRB-for-SCAE_Attack/GTSRB'


def _stacked_capsule_autoencoder(
		canvas_size,
		template_size=11,
		n_part_caps=16,
		n_part_caps_dims=6,
		n_part_special_features=16,
		part_encoder_noise_scale=0.,
		n_channels=1,
		colorize_templates=False,
		use_alpha_channel=False,
		template_nonlin='relu1',
		color_nonlin='relu1',
		n_obj_caps=10,
		n_obj_caps_params=32,
		obj_decoder_noise_type=None,
		obj_decoder_noise_scale=0.,
		num_classes=10,
		prior_within_example_sparsity_weight=1.,
		prior_between_example_sparsity_weight=1.,
		posterior_within_example_sparsity_weight=10.,
		posterior_between_example_sparsity_weight=10.,
		set_transformer_n_layers=3,
		set_transformer_n_heads=1,
		set_transformer_n_dims=16,
		set_transformer_n_output_dims=256,
		part_cnn_strides=None,
		prep='none',
		scope='stacked_capsule_autoencoders'
):
	if part_cnn_strides is None:
		part_cnn_strides = [2, 2, 1, 1]

	"""Builds the SCAE."""
	with tf.variable_scope(scope, 'stacked_capsule_autoencoders'):
		img_size = [canvas_size] * 2
		template_size = [template_size] * 2

		cnn_encoder = snt.nets.ConvNet2D(
			output_channels=[128] * 4,
			kernel_shapes=[3],
			strides=part_cnn_strides,
			paddings=[snt.VALID],
			activate_final=True
		)

		part_encoder = primary.CapsuleImageEncoder(
			cnn_encoder,
			n_part_caps,
			n_part_caps_dims,
			n_features=n_part_special_features,
			similarity_transform=False,
			encoder_type='conv_att',
			noise_scale=part_encoder_noise_scale
		)

		part_decoder = primary.TemplateBasedImageDecoder(
			output_size=img_size,
			template_size=template_size,
			n_channels=n_channels,
			learn_output_scale=False,
			colorize_templates=colorize_templates,
			use_alpha_channel=use_alpha_channel,
			template_nonlin=template_nonlin,
			color_nonlin=color_nonlin,
		)

		obj_encoder = SetTransformer(
			n_layers=set_transformer_n_layers,
			n_heads=set_transformer_n_heads,
			n_dims=set_transformer_n_dims,
			n_output_dims=set_transformer_n_output_dims,
			n_outputs=n_obj_caps,
			layer_norm=True,
			dropout_rate=0.)

		obj_decoder = ImageCapsule(
			n_obj_caps,
			2,
			n_part_caps,
			n_caps_params=n_obj_caps_params,
			n_hiddens=128,
			learn_vote_scale=True,
			deformations=True,
			noise_type=obj_decoder_noise_type,
			noise_scale=obj_decoder_noise_scale,
			similarity_transform=False
		)

		model = ImageAutoencoder(
			primary_encoder=part_encoder,
			primary_decoder=part_decoder,
			encoder=obj_encoder,
			decoder=obj_decoder,
			input_key='image',
			label_key='label',
			target_key='target',
			n_classes=num_classes,
			dynamic_l2_weight=10,
			caps_ll_weight=1.,
			vote_type='enc',
			pres_type='enc',
			stop_grad_caps_inpt=True,
			stop_grad_caps_target=True,
			prior_sparsity_loss_type='l2',
			prior_within_example_sparsity_weight=prior_within_example_sparsity_weight,
			prior_between_example_sparsity_weight=prior_between_example_sparsity_weight,
			posterior_sparsity_loss_type='entropy',
			posterior_within_example_sparsity_weight=posterior_within_example_sparsity_weight,
			posterior_between_example_sparsity_weight=posterior_between_example_sparsity_weight,
			prep=prep
		)

	return model


class ScaeAdvTrain(_ModelCollector):
	"""
		SCAE model collector with graph that is not finalized during initialization.
		Instead, finalization can be done by calling function finalize().
		After initialization, supportive models can be applied to the graph of this SCAE.
	"""

	def __init__(
			self,
			input_size,
			template_size=11,
			n_part_caps=16,
			n_part_caps_dims=6,
			n_part_special_features=16,
			part_encoder_noise_scale=0.,
			colorize_templates=False,
			use_alpha_channel=False,
			template_nonlin='relu1',
			color_nonlin='relu1',
			n_obj_caps=10,
			n_obj_caps_params=32,
			obj_decoder_noise_type=None,
			obj_decoder_noise_scale=0.,
			num_classes=10,
			prior_within_example_sparsity_weight=1.,
			prior_between_example_sparsity_weight=1.,
			posterior_within_example_sparsity_weight=10.,
			posterior_between_example_sparsity_weight=10.,
			set_transformer_n_layers=3,
			set_transformer_n_heads=1,
			set_transformer_n_dims=16,
			set_transformer_n_output_dims=256,
			part_cnn_strides=None,
			prep='none',
			is_training=True,
			learning_rate=1e-4,
			use_lr_schedule=True,
			scope='SCAE',
			snapshot=None
	):
		if input_size is None:
			input_size = [20, 224, 224, 3]
		if part_cnn_strides is None:
			part_cnn_strides = [2, 2, 1, 1]

		self._input_size = input_size
		self._template_size = template_size
		self._n_part_caps = n_part_caps
		self._n_part_caps_dims = n_part_caps_dims
		self._n_part_special_features = n_part_special_features
		self._part_encoder_noise_scale = part_encoder_noise_scale
		self._n_channels = input_size[-1]
		self._colorize_templates = colorize_templates
		self._use_alpha_channel = use_alpha_channel
		self._template_nonlin = template_nonlin
		self._color_nonlin = color_nonlin
		self._n_obj_caps = n_obj_caps
		self._n_obj_caps_params = n_obj_caps_params
		self._obj_decoder_noise_type = obj_decoder_noise_type
		self._obj_decoder_noise_scale = obj_decoder_noise_scale
		self._num_classes = num_classes
		self._prior_within_example_sparsity_weight = prior_within_example_sparsity_weight
		self._prior_between_example_sparsity_weight = prior_between_example_sparsity_weight
		self._posterior_within_example_sparsity_weight = posterior_within_example_sparsity_weight
		self._posterior_between_example_sparsity_weight = posterior_between_example_sparsity_weight
		self._set_transformer_n_layers = set_transformer_n_layers
		self._set_transformer_n_heads = set_transformer_n_heads
		self._set_transformer_n_dims = set_transformer_n_dims
		self._set_transformer_n_output_dims = set_transformer_n_output_dims,
		self._part_cnn_strides = part_cnn_strides
		self._prep = prep
		self._is_training = is_training
		self._learning_rate = learning_rate
		self._use_lr_schedule = use_lr_schedule
		self._scope = scope
		self._snapshot = snapshot

		graph = tf.Graph()

		with graph.as_default():
			self._sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

			self._input = tf.placeholder(tf.float32, input_size)
			self._model = _stacked_capsule_autoencoder(input_size[1],  # Assume width equals height
			                                           template_size,
			                                           n_part_caps,
			                                           n_part_caps_dims,
			                                           n_part_special_features,
			                                           part_encoder_noise_scale,
			                                           input_size[-1],
			                                           colorize_templates,
			                                           use_alpha_channel,
			                                           template_nonlin,
			                                           color_nonlin,
			                                           n_obj_caps,
			                                           n_obj_caps_params,
			                                           obj_decoder_noise_type,
			                                           obj_decoder_noise_scale,
			                                           num_classes,
			                                           prior_within_example_sparsity_weight,
			                                           prior_between_example_sparsity_weight,
			                                           posterior_within_example_sparsity_weight,
			                                           posterior_between_example_sparsity_weight,
			                                           set_transformer_n_layers,
			                                           set_transformer_n_heads,
			                                           set_transformer_n_dims,
			                                           set_transformer_n_output_dims,
			                                           part_cnn_strides,
			                                           prep,
			                                           scope)

			if is_training:
				self._label = tf.placeholder(tf.int64, [input_size[0]])
				self._target = tf.placeholder(tf.float32, input_size)
				data = {'image': self._input, 'label': self._label, 'target': self._target}
				self._res = self._model(data)

				self._loss = self._model._loss(data, self._res)

				res_clean = self._model({'image': self._target})
				pri_clean = res_clean.caps_presence_prob
				pos_clean = res_clean.posterior_mixing_probs
				is_input_target_diff = tf.clip_by_value(tf.reduce_sum(tf.cast(
					tf.not_equal(self._target, self._input), tf.float32)), 0, 1)
				self._loss += is_input_target_diff * (tf.nn.l2_loss(pri_clean - self._res.caps_presence_prob) +
				                                      tf.nn.l2_loss(pos_clean - self._res.posterior_mixing_probs))

				if use_lr_schedule:
					global_step = tf.train.get_or_create_global_step()
					learning_rate = tf.train.exponential_decay(
						global_step=global_step,
						learning_rate=learning_rate,
						decay_steps=1e4,
						decay_rate=.96
					)
					global_step.initializer.run(session=self._sess)

				eps = 1e-2 / float(input_size[0]) ** 2
				optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=.9, epsilon=eps)

				self._train_step = optimizer.minimize(self._loss, var_list=tf.trainable_variables(scope=scope))
				self._sess.run(tf.initialize_variables(var_list=optimizer.variables()))
			else:
				data = {'image': self._input}
				self._res = self._model(data)

			self._saver = tf.train.Saver(var_list=tf.trainable_variables(scope=scope))

			if snapshot:
				print('Restoring from snapshot: {}'.format(snapshot))
				self._saver.restore(self._sess, snapshot)
			else:
				self._sess.run(tf.initialize_variables(var_list=tf.trainable_variables(scope=scope)))

	def run(self, images, to_collect, labels=None, targets=None):
		try:
			if labels is not None:
				images, num_images, labels = self._valid_shape(images, labels)
				targets, _ = self._valid_shape(targets if targets is not None else images)
				return self._sess.run(to_collect, feed_dict={
					self._input: images,
					self._label: labels,
					self._target: targets
				})[:num_images]

			images, num_images = self._valid_shape(images)
			return self._sess.run(to_collect, feed_dict={
				self._input: images
			})[:num_images]

		except tf.errors.InvalidArgumentError:
			pass

		raise NotImplementedError('Model is in training mode. Labels and targets must be provided.')

	def train_step(self, images, labels, targets=None):
		if not self._is_training:
			raise NotImplementedError('Model is not in training mode.')

		return self._sess.run(self._train_step, feed_dict={
			self._input: images,
			self._label: labels,
			self._target: targets if targets is not None else images
		})

	# ---------------------------------------- Inherit from ScaeBasement ----------------------------------------
	def __call__(self, images):
		return ScaeBasement.__call__(self, images)

	def _valid_shape(self, images, labels=None):
		return ScaeBasement._valid_shape(self, images, labels)

	def finalize(self):
		return ScaeBasement.finalize(self)

	def simple_test(self, dataset: DatasetHelper):
		return ScaeBasement.simple_test(self, dataset)

	def save_model(self, path):
		return ScaeBasement.save_model(self, path)


class ScaeDistTrain(_ModelCollector):
	def __init__(
			self,
			scae: ScaeBasement,
			scope_teacher='SCAE',
			snapshot_teacher=None,
			loss_lambda=0.5,
			temperature=100
	):
		self._sess = scae._sess
		self._valid_shape = scae._valid_shape

		with self._sess.graph.as_default():
			self._model = _scae(scae._input_size[1],  # Assume width equals height
			                    scae._template_size,
			                    scae._n_part_caps,
			                    scae._n_part_caps_dims,
			                    scae._n_part_special_features,
			                    scae._part_encoder_noise_scale,
			                    scae._n_channels,
			                    scae._colorize_templates,
			                    scae._use_alpha_channel,
			                    scae._template_nonlin,
			                    scae._color_nonlin,
			                    scae._n_obj_caps,
			                    scae._n_obj_caps_params,
			                    scae._obj_decoder_noise_type,
			                    scae._obj_decoder_noise_scale,
			                    scae._num_classes,
			                    scae._prior_within_example_sparsity_weight,
			                    scae._prior_between_example_sparsity_weight,
			                    scae._posterior_within_example_sparsity_weight,
			                    scae._posterior_between_example_sparsity_weight,
			                    scae._set_transformer_n_layers,
			                    scae._set_transformer_n_heads,
			                    scae._set_transformer_n_dims,
			                    scae._set_transformer_n_output_dims,
			                    scae._part_cnn_strides,
			                    scae._prep,
			                    scope_teacher)

			self._input_stu = scae._input
			self._input_tch = tf.placeholder(tf.float32, scae._input_size)
			self._label = scae._label

			self._res_stu = scae._res
			self._res_tch = self._model({'image': self._input_tch})

			# alias
			self._input = self._input_tch
			self._res = self._res_tch
			self._is_training = False

			learning_rate = scae._learning_rate
			if scae._use_lr_schedule:
				global_step = tf.train.get_or_create_global_step()
				learning_rate = tf.train.exponential_decay(
					global_step=global_step,
					learning_rate=learning_rate,
					decay_steps=1e4,
					decay_rate=.96
				)
				global_step.initializer.run(session=self._sess)

			loss_pri = tf.nn.l2_loss(
				self._res_stu.caps_presence_prob - self._res_tch.caps_presence_prob)
			loss_pos = tf.nn.l2_loss(
				self._res_stu.posterior_mixing_probs - self._res_tch.posterior_mixing_probs)

			self._loss = (1 - loss_lambda) * scae._loss + \
			             loss_lambda * temperature ** 2 * (loss_pri + loss_pos)

			eps = 1e-2 / float(scae._input_size[0]) ** 2
			optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=.9, epsilon=eps)

			self._train_step = optimizer.minimize(self._loss, var_list=tf.trainable_variables(scope=scae._scope))
			self._sess.run(tf.initialize_variables(var_list=optimizer.variables()))

			saver = tf.train.Saver(var_list=tf.trainable_variables(scope=scope_teacher))
			print('Restoring teacher from snapshot: {}'.format(snapshot_teacher))
			saver.restore(self._sess, snapshot_teacher)

	def run(self, images, to_collect, labels=None, images_student=None):
		try:
			if images_student is not None:
				assert len(images) == len(images_student)
				images_student, _ = self._valid_shape(images_student)

			if labels is not None:
				images, num_images, labels = self._valid_shape(images, labels)
				return self._sess.run(to_collect, feed_dict={
					self._input_tch: images,
					self._input_stu: images_student if images_student is not None else images,
					self._label: labels
				})[:num_images]

			images, num_images = self._valid_shape(images)
			return self._sess.run(to_collect, feed_dict={
				self._input_tch: images,
				self._input_stu: images_student if images_student is not None else images,
			})[:num_images]

		except tf.errors.InvalidArgumentError:
			pass

		raise NotImplementedError('Request outputs need labels to be provided.')

	def train_step(self, images_teacher, images_student, labels):
		return self._sess.run(self._train_step, feed_dict={
			self._input_tch: images_teacher,
			self._input_stu: images_student,
			self._label: labels
		})

	# ---------------------------------------- Inherit from ScaeBasement ----------------------------------------
	def __call__(self, images):
		return ScaeBasement.__call__(self, images)

	def simple_test(self, dataset: DatasetHelper):
		return ScaeBasement.simple_test(self, dataset)


def draw_pdf(xmax, labels, *data):
	for i in range(len(labels)):
		_data = data[i]
		label = labels[i]

		_data = np.sort(_data)
		probability = [(i + 1) / len(_data) for i in range(len(_data))]
		plt.plot(_data, probability, label=label)

	plt.xlim((0, xmax))
	plt.xlabel('Pert Amount Threshold')
	plt.ylim((0, 1))
	plt.ylabel('Attack Success Rate')
	plt.legend(loc=2)
	plt.grid()
	plt.show()
