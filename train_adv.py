import os

import numpy as np
from tqdm import tqdm

from SCAE.attack_bim import AttackerBIM
from SCAE.tools.utilities import DatasetHelper
from SCAE.train import block_warnings
from utilities import ScaeAdvTrain, Configs


def build_from_config(
		config,
		batch_size,
		is_training=False,
		learning_rate=1e-4,
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
		part_encoder_noise_scale=config['part_encoder_noise_scale'] if is_training else 0.,
		obj_decoder_noise_type=config['obj_decoder_noise_type'] if is_training else None,
		obj_decoder_noise_scale=config['obj_decoder_noise_scale'] if is_training else 0.,
		set_transformer_n_layers=config['set_transformer_n_layers'],
		set_transformer_n_heads=config['set_transformer_n_heads'],
		set_transformer_n_dims=config['set_transformer_n_dims'],
		set_transformer_n_output_dims=config['set_transformer_n_output_dims'],
		part_cnn_strides=config['part_cnn_strides'],
		prep=config['prep'],
		is_training=is_training,
		learning_rate=learning_rate,
		scope='SCAE',
		use_lr_schedule=use_lr_schedule,
		snapshot=snapshot
	)


if __name__ == '__main__':
	block_warnings()

	config = Configs.config_mnist
	batch_size = 100
	max_train_steps = 300
	learning_rate = 3e-5
	snapshot = './checkpoints/{}/model.ckpt'.format(config['dataset'])
	num_adv_images = 100

	num_epochs = 5
	alpha = 0.4

	model = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=True,
		learning_rate=learning_rate,
		use_lr_schedule=True,
		snapshot=snapshot
	)
	attacker = AttackerBIM(
		scae=model,
		classifier=AttackerBIM.Classifiers.PosL,
		alpha=alpha
	)
	model.finalize()

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

	path = snapshot[:snapshot.rindex('/')]
	if not os.path.exists(path):
		os.makedirs(path)

	# Classification accuracy test
	model.simple_test(testset)

	# Train model
	for epoch in range(max_train_steps):
		print('\n[Epoch {}/{}]'.format(epoch + 1, max_train_steps))

		for images, labels in tqdm(trainset, desc='Training'):
			images[-num_adv_images:] = attacker(images[-num_adv_images:], labels[-num_adv_images:], num_epochs, False, False)
			model.train_step(images, labels)

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
		test_loss /= testset.dataset_size

		print('loss: {:.6f}  prior acc: {:.6f}  posterior acc: {:.6f}'.format(
			test_loss,
			test_acc_prior / testset.dataset_size,
			test_acc_posterior / testset.dataset_size
		))

		model.save_model(snapshot)
