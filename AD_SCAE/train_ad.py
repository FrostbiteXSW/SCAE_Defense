import tensorflow as tf
from tqdm import tqdm

from DD_SCAE.train_dd import snapshot_student, snapshot_teacher
from SCAE.attack_cw import AttackerCW
from SCAE.tools.model import ScaeBasement
from SCAE.tools.model import _ModelCollector, _stacked_capsule_autoencoder as _scae
from SCAE.tools.utilities import block_warnings
from SCAE.train import build_from_config
from utilities import *


class ScaeAdvDist(_ModelCollector):
	def __init__(
			self,
			scae: ScaeBasement,
			scope_teacher='SCAE',
			snapshot_teacher=None,
			loss_lambda=0.5
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

			self._input_stu = scae._input  # The adversarial input
			self._input_tch = tf.placeholder(tf.float32, scae._input_size)  # The clean input
			self._label = scae._label

			self._res_stu = scae._model({'image': self._input_stu, 'label': self._label, 'target': self._input_tch})
			self._res_tch = self._model({'image': self._input_tch})

			# alias
			self._input = self._input_tch
			self._res = self._res_tch
			self._is_training = False

			self._learning_rate = scae._learning_rate
			self._global_step = scae._global_step

			loss_pri = tf.nn.l2_loss(self._res_stu.caps_presence_prob
			                         - tf.stop_gradient(self._res_tch.caps_presence_prob))
			loss_pos = tf.nn.l2_loss(self._res_stu.posterior_mixing_probs
			                         - tf.stop_gradient(self._res_tch.posterior_mixing_probs))
			self._loss = (1 - loss_lambda) * scae._loss + loss_lambda * (loss_pri + loss_pos)
			self._train_step = scae._optimizer.minimize(self._loss, global_step=self._global_step,
			                                            var_list=tf.trainable_variables(scope=scae._scope))

			saver = tf.train.Saver(var_list=tf.trainable_variables(scope=scope_teacher))
			print('Restoring teacher from snapshot: {}'.format(snapshot_teacher))
			saver.restore(self._sess, snapshot_teacher)

	@property
	def global_step(self) -> int:
		return self._sess.run(self._global_step)

	@property
	def learning_rate(self) -> float:
		if isinstance(self._learning_rate, (int, float)):
			return self._learning_rate
		return self._sess.run(self._learning_rate)

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


if __name__ == '__main__':
	block_warnings()

	config = Configs.config_mnist
	batch_size = 100
	max_train_steps = 50
	learning_rate = 3e-5
	snapshot_teacher = snapshot_teacher.format(config['name'])
	snapshot_student = snapshot_student.format(config['name'])

	# Distillation configuration
	loss_lambda = 0.5
	num_batches_per_adv_train = 2

	# Attack configuration
	optimizer_config = AttackerCW.OptimizerConfigs.FGSM_normal
	classifier = AttackerCW.Classifiers.PosL

	# We are not going to use the embedded noise
	config['part_encoder_noise_scale'] = 0.
	config['obj_decoder_noise_type'] = None
	config['obj_decoder_noise_scale'] = 0.

	student = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=True,
		learning_rate=learning_rate,
		scope='STU',
		use_lr_schedule=True
	)

	teacher = ScaeAdvDist(
		scae=student,
		scope_teacher='SCAE',
		snapshot_teacher=snapshot_teacher,
		loss_lambda=loss_lambda
	)

	attacker = AttackerCW(
		scae=student,
		optimizer_config=optimizer_config,
		classifier=classifier
	)

	student.finalize()

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

	teacher.simple_test(testset)

	# Make snapshot directory
	path = snapshot_student[:snapshot_student.rindex('/')]
	remove_make_dirs(path)

	test_acc_prior_list = []
	test_acc_posterior_list = []
	n_batches = 0
	for epoch in range(max_train_steps):
		print('\n[Epoch {}/{}]'.format(epoch + 1, max_train_steps))

		tqdm_trainset = tqdm(trainset, desc='Training')
		for images, labels in tqdm_trainset:
			n_batches += 1
			if n_batches != num_batches_per_adv_train:
				teacher.train_step(images, images, labels)
			else:
				n_batches = 0
				teacher.train_step(images, attacker(images, labels, nan_if_fail=False), labels)
			tqdm_trainset.set_postfix_str(f'GS={teacher.global_step}, LR={teacher.learning_rate:.1e}')
		tqdm_trainset.close()

		test_loss = 0.
		test_acc_prior = 0.
		test_acc_posterior = 0.
		for images, labels in tqdm(testset, desc='Testing'):
			test_pred_prior, test_pred_posterior, _test_loss = student.run(
				images=images,
				labels=labels,
				to_collect=[student._res.prior_cls_pred,
				            student._res.posterior_cls_pred,
				            student._loss]
			)
			test_loss += _test_loss
			test_acc_prior += (test_pred_prior == labels).sum()
			test_acc_posterior += (test_pred_posterior == labels).sum()
			assert not np.isnan(test_loss)

		test_acc_prior /= testset.dataset_size
		test_acc_posterior /= testset.dataset_size
		test_acc_prior_list.append(test_acc_prior)
		test_acc_posterior_list.append(test_acc_posterior)

		print('loss: {:.6f}  prior acc: {:.6f}  posterior acc: {:.6f}'.format(
			test_loss / testset.dataset_size, test_acc_prior, test_acc_posterior))

		student.save_model(snapshot_student)

	draw_accuracy_variation(max_train_steps, ['Prior Acc', 'Posterior Acc'],
	                        [test_acc_prior_list, test_acc_posterior_list],
	                        title='Test Accuracy Variation', file_path=os.path.join(path, 'accuracy_variation_plot.png'))
