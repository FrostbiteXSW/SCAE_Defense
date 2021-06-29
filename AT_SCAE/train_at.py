import os

import sonnet as snt
import tensorflow as tf
from tqdm import tqdm

from SCAE.attack_cw import AttackerCW
from SCAE.capsules import primary
from SCAE.capsules.attention import SetTransformer
from SCAE.capsules.models.scae import ImageAutoencoder, ImageCapsule
from SCAE.tools.model import _ModelCollector, ScaeBasement
from SCAE.tools.utilities import block_warnings
from utilities import *

# File paths
snapshot = './checkpoints/{}/model.ckpt'
dataset_path = '../SCAE/datasets/'
gtsrb_dataset_path = '../SCAE/datasets/GTSRB-for-SCAE_Attack/GTSRB/'


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
				pri_clean = tf.stop_gradient(res_clean.caps_presence_prob)
				pos_clean = tf.stop_gradient(res_clean.posterior_mixing_probs)
				self._loss += tf.nn.l2_loss(pri_clean - self._res.caps_presence_prob) + \
				              tf.nn.l2_loss(pos_clean - self._res.posterior_mixing_probs)

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


def build_from_config(
		config,
		batch_size,
		is_training=False,
		learning_rate=1e-4,
		scope='SCAE',
		use_lr_schedule=True,
		snapshot=None
):
	return ScaeAdvTrain(
		input_size=[batch_size, config['canvas_size'], config['canvas_size'], config['n_channels']],
		num_classes=config['num_classes'],
		n_part_caps=config['n_part_caps'],
		n_obj_caps=config['n_obj_caps'],
		colorize_templates=config['colorize_templates'],
		use_alpha_channel=config['use_alpha_channel'],
		prior_within_example_sparsity_weight=config['prior_within_example_sparsity_weight'],
		prior_between_example_sparsity_weight=config['prior_between_example_sparsity_weight'],
		posterior_within_example_sparsity_weight=config['posterior_within_example_sparsity_weight'],
		posterior_between_example_sparsity_weight=config['posterior_between_example_sparsity_weight'],
		template_size=config['template_size'],
		template_nonlin=config['template_nonlin'],
		color_nonlin=config['color_nonlin'],
		part_encoder_noise_scale=0.,
		obj_decoder_noise_type=None,
		obj_decoder_noise_scale=0.,
		set_transformer_n_layers=config['set_transformer_n_layers'],
		set_transformer_n_heads=config['set_transformer_n_heads'],
		set_transformer_n_dims=config['set_transformer_n_dims'],
		set_transformer_n_output_dims=config['set_transformer_n_output_dims'],
		part_cnn_strides=config['part_cnn_strides'],
		prep=config['prep'],
		is_training=is_training,
		learning_rate=learning_rate,
		scope=scope,
		use_lr_schedule=use_lr_schedule,
		snapshot=snapshot
	)


if __name__ == '__main__':
	block_warnings()

	config = Configs.config_mnist
	batch_size = 100
	max_train_steps = 50
	learning_rate = 3e-5
	snapshot = snapshot.format(config['dataset'])
	num_batches_per_adv_train = 2

	# Attack configuration
	optimizer_config = AttackerCW.OptimizerConfigs.FGSM_normal
	classifier = AttackerCW.Classifiers.PosL

	path = snapshot[:snapshot.rindex('/')]
	if not os.path.exists(path):
		os.makedirs(path)

	model = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=True,
		learning_rate=learning_rate,
		scope='SCAE',
		use_lr_schedule=True
	)

	attacker = AttackerCW(
		scae=model,
		optimizer_config=optimizer_config,
		classifier=classifier
	)

	model.finalize()

	trainset = DatasetHelper(config['dataset'], 'train', shape=[config['canvas_size']] * 2,
	                         file_path=dataset_path, save_after_load=True,
	                         batch_size=batch_size, shuffle=True, fill_batch=True,
	                         normalize=True if config['dataset'] == Configs.GTSRB else False,
	                         gtsrb_raw_file_path=gtsrb_dataset_path, gtsrb_classes=Configs.GTSRB_CLASSES)
	testset = DatasetHelper(config['dataset'], 'test', shape=[config['canvas_size']] * 2,
	                        file_path=dataset_path, save_after_load=True,
	                        batch_size=batch_size, fill_batch=True,
	                        normalize=True if config['dataset'] == Configs.GTSRB else False,
	                        gtsrb_raw_file_path=gtsrb_dataset_path, gtsrb_classes=Configs.GTSRB_CLASSES)

	model.simple_test(testset)

	n_batches = 0
	for epoch in range(max_train_steps):
		print('\n[Epoch {}/{}]'.format(epoch + 1, max_train_steps))

		for images, labels in tqdm(trainset, desc='Training'):
			n_batches += 1
			if n_batches != num_batches_per_adv_train:
				model.train_step(images, labels)
			else:
				n_batches = 0
				model.train_step(attacker(images, labels, nan_if_fail=False), labels, images)

		test_loss = 0.
		test_acc_prior = 0.
		test_acc_posterior = 0.
		for images, labels in tqdm(testset, desc='Testing'):
			test_pred_prior, test_pred_posterior, _test_loss = model.run(
				images=images,
				labels=labels,
				to_collect=[model._res.prior_cls_pred,
				            model._res.posterior_cls_pred,
				            model._loss]
			)
			test_loss += _test_loss
			test_acc_prior += (test_pred_prior == labels).sum()
			test_acc_posterior += (test_pred_posterior == labels).sum()
			assert not np.isnan(test_loss)

		print('loss: {:.6f}  prior acc: {:.6f}  posterior acc: {:.6f}'.format(
			test_loss / testset.dataset_size,
			test_acc_prior / testset.dataset_size,
			test_acc_posterior / testset.dataset_size
		))

		model.save_model(snapshot)