from tqdm import tqdm

from DD_SCAE.train_dd import snapshot_student, snapshot_teacher, ScaeDefDist
from SCAE.attack_cw import AttackerCW
from SCAE.tools.utilities import block_warnings
from SCAE.train import build_from_config
from utilities import *

if __name__ == '__main__':
	block_warnings()

	config = Configs.config_mnist
	batch_size = 100
	max_train_steps = 100
	learning_rate = 3e-5
	snapshot_student = snapshot_student.format(config['dataset'])
	snapshot_teacher = snapshot_teacher.format(config['dataset'])

	# Distillation configuration
	loss_lambda = 0.5
	temperature = 1
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

	teacher = ScaeDefDist(
		scae=student,
		scope_teacher='SCAE',
		snapshot_teacher=snapshot_teacher,
		loss_lambda=loss_lambda,
		temperature=temperature
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
