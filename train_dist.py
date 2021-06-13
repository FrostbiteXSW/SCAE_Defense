import os

import numpy as np
from tqdm import tqdm

from SCAE.attack_cw import AttackerCW
from SCAE.tools.model import ScaeBasement
from SCAE.tools.utilities import block_warnings, DatasetHelper
from utilities import ScaeDistTrain, Configs


def build_from_config(
	config,
	batch_size,
	is_training=False,
	learning_rate=1e-4,
	scope='SCAE',
	use_lr_schedule=True,
	snapshot=None
):
	return ScaeBasement(
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
	snapshot_student = './checkpoints/{}_dist/model.ckpt'.format(config['dataset'])
	snapshot_teacher = './checkpoints/{}/model.ckpt'.format(config['dataset'])
	num_batches_per_adv_train = 2

	# Attack configuration
	optimizer_config = AttackerCW.OptimizerConfigs.FGSM_fast
	classifier = AttackerCW.Classifiers.PosL

	path = snapshot_student[:snapshot_student.rindex('/')]
	if not os.path.exists(path):
		os.makedirs(path)

	student = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=True,
		learning_rate=learning_rate,
		scope='STU',
		use_lr_schedule=True
	)

	teacher = ScaeDistTrain(
		scae=student,
		scope_teacher='SCAE',
		snapshot_teacher=snapshot_teacher
	)

	attacker = AttackerCW(
		scae=student,
		optimizer_config=optimizer_config,
		classifier=classifier
	)

	student.finalize()

	trainset = DatasetHelper(config['dataset'], 'train', shape=[config['canvas_size']] * 2,
	                         file_path='./datasets', save_after_load=True,
	                         batch_size=batch_size, shuffle=True, fill_batch=True,
	                         normalize=True if config['dataset'] == Configs.GTSRB else False,
	                         gtsrb_raw_file_path=Configs.GTSRB_DATASET_PATH, gtsrb_classes=Configs.GTSRB_CLASSES)
	testset = DatasetHelper(config['dataset'], 'test', shape=[config['canvas_size']] * 2,
	                        file_path='./datasets', save_after_load=True,
	                        batch_size=batch_size, fill_batch=True,
	                        normalize=True if config['dataset'] == Configs.GTSRB else False,
	                        gtsrb_raw_file_path=Configs.GTSRB_DATASET_PATH, gtsrb_classes=Configs.GTSRB_CLASSES)

	teacher.simple_test(testset)

	n_batches = 0
	for epoch in range(max_train_steps):
		print('\n[Epoch {}/{}]'.format(epoch + 1, max_train_steps))

		for images, labels in tqdm(trainset, desc='Training'):
			n_batches += 1
			if n_batches != num_batches_per_adv_train:
				teacher.train_step(images, images, labels)
			else:
				n_batches = 0
				teacher.train_step(images, attacker(images, labels, nan_if_fail=False), labels)

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

		print('loss: {:.6f}  prior acc: {:.6f}  posterior acc: {:.6f}'.format(
			test_loss / testset.dataset_size,
			test_acc_prior / testset.dataset_size,
			test_acc_posterior / testset.dataset_size
		))

		student.save_model(snapshot_student)
