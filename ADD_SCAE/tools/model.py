import tensorflow as tf

from SCAE.tools.model import _ModelCollector, ScaeBasement, _stacked_capsule_autoencoder as _scae
from SCAE.tools.utilities import DatasetHelper


class ScaeAdvDefDist(_ModelCollector):
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
