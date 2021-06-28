from ADD_SCAE.train_add import *
from SCAE.tools.utilities import block_warnings

if __name__ == '__main__':
	block_warnings()

	config = Configs.config_fashion_mnist
	batch_size = 100
	max_train_steps = 50
	learning_rate = 3e-5
	snapshot_student = snapshot_student.format(config['dataset'])
	snapshot_teacher = snapshot_teacher.format(config['dataset'])

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

	teacher = ScaeDefDist(
		scae=student,
		scope_teacher='SCAE',
		snapshot_teacher=snapshot_teacher
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

	for epoch in range(max_train_steps):
		print('\n[Epoch {}/{}]'.format(epoch + 1, max_train_steps))

		for images, labels in tqdm(trainset, desc='Training'):
			teacher.train_step(images, images, labels)

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
