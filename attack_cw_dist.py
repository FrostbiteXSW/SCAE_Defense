import numpy as np

from SCAE.attack_cw import AttackerCW
from SCAE.tools.model import Attacker, KMeans
from SCAE.tools.utilities import block_warnings, DatasetHelper, ResultBuilder
from train_dist import Configs, build_from_config

if __name__ == '__main__':
	block_warnings()

	# Attack configuration
	config = Configs.config_mnist
	optimizer_config = AttackerCW.OptimizerConfigs.Adam_fast
	num_samples = 1000
	batch_size = 100
	classifier = Attacker.Classifiers.PosK
	use_mask = True
	pert_percentile = 0.9

	snapshot_stu = './checkpoints/{}_dist/model.ckpt'.format(config['dataset'])
	snapshot_kmeans_stu = './checkpoints/{}_dist/kmeans_{}/model.ckpt'.format(
		config['dataset'], 'pri' if classifier[:3].upper() == 'PRI' else 'pos')

	snapshot_tch = './SCAE/checkpoints/{}/model.ckpt'.format(config['dataset'])
	snapshot_kmeans_tch = './SCAE/checkpoints/{}/kmeans_{}/model.ckpt'.format(
		config['dataset'], 'pri' if classifier[:3].upper() == 'PRI' else 'pos')

	# Create student attack model
	model_stu = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=False,
		snapshot=snapshot_stu,
		scope='STU'
	)

	if classifier[-1].upper() == 'K':
		kmeans_stu = KMeans(
			scae=model_stu,
			kmeans_type=KMeans.KMeansTypes.Prior if classifier[:3].upper() == 'PRI' else KMeans.KMeansTypes.Posterior,
			is_training=False,
			scope='KMeans_Pri' if classifier[:3].upper() == 'PRI' else 'KMeans_Pos',
			snapshot=snapshot_kmeans_stu
		)

	attacker_stu = AttackerCW(
		scae=model_stu,
		optimizer_config=optimizer_config,
		classifier=classifier,
		kmeans_classifier=kmeans_stu if classifier[-1].upper() == 'K' else None
	)

	model_stu.finalize()

	# Create teacher attack model
	model_tch = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=False,
		snapshot=snapshot_tch,
		scope='SCAE'
	)

	if classifier[-1].upper() == 'K':
		kmeans_tch = KMeans(
			scae=model_tch,
			kmeans_type=KMeans.KMeansTypes.Prior if classifier[:3].upper() == 'PRI' else KMeans.KMeansTypes.Posterior,
			is_training=False,
			scope='KMeans_Pri' if classifier[:3].upper() == 'PRI' else 'KMeans_Pos',
			snapshot=snapshot_kmeans_tch
		)

	attacker_tch = AttackerCW(
		scae=model_tch,
		optimizer_config=optimizer_config,
		classifier=classifier,
		kmeans_classifier=kmeans_tch if classifier[-1].upper() == 'K' else None
	)

	model_tch.finalize()

	# Load dataset
	dataset = DatasetHelper(config['dataset'],
	                        'train' if config['dataset'] == Configs.GTSRB
	                                   or config['dataset'] == Configs.FASHION_MNIST else 'test',
	                        file_path='./SCAE/datasets', batch_size=batch_size, shuffle=True, fill_batch=True,
	                        normalize=True if config['dataset'] == Configs.GTSRB else False,
	                        gtsrb_raw_file_path=Configs.GTSRB_DATASET_PATH, gtsrb_classes=Configs.GTSRB_CLASSES)

	# Temporary results
	stu_source_images = []
	stu_pert_images = []
	tch_pert_amount = []
	tch_succeed_count = 0
	stu_classification_error_count = 0

	# Classification accuracy test
	model_stu.simple_test(dataset)
	model_tch.simple_test(dataset)

	# Start the attack on selected samples
	dataset = iter(dataset)
	remain = num_samples
	while remain > 0:
		images, labels = next(dataset)

		# Judge student classification
		if classifier[-1].upper() == 'K':
			right_classification_stu = kmeans_stu(images) == labels
		else:
			right_classification_stu = model_stu.run(
				images=images,
				to_collect=model_stu._res.prior_cls_pred if classifier == Attacker.Classifiers.PriL
				else model_stu._res.posterior_cls_pred
			) == labels

		# Judge teacher classification
		if classifier[-1].upper() == 'K':
			right_classification_tch = kmeans_tch(images) == labels
		else:
			right_classification_tch = model_tch.run(
				images=images,
				to_collect=model_tch._res.prior_cls_pred if classifier == Attacker.Classifiers.PriL
				else model_tch._res.posterior_cls_pred
			) == labels

		output_stu = attacker_stu(images, labels, nan_if_fail=True, verbose=True)
		output_tch = attacker_tch(images, labels, nan_if_fail=True, verbose=True)

		for i in range(batch_size):
			if right_classification_tch[i] and remain:
				remain -= 1
				stu_source_images.append(images[i])

				if True not in np.isnan(output_tch[i]):
					tch_succeed_count += 1
					tch_pert_amount.append(np.linalg.norm(output_tch[i] - images[i]))

				if right_classification_stu[i]:
					stu_pert_images.append(output_stu[i])
				else:
					stu_classification_error_count += 1
					stu_pert_images.append(images[i])

		print('Remain: {}, student classification error count: {}\n'.format(remain, stu_classification_error_count))

	# Compute pert threshold
	tch_pert_amount = np.sort(tch_pert_amount)
	pert_threshold = tch_pert_amount[int(tch_succeed_count * pert_percentile)]
	print('Pert threshold is {:.4f} (according to {} samples)\n'.format(pert_threshold, tch_succeed_count))

	# Variables to save the attack result
	succeed_count = 0
	succeed_pert_amount = []
	succeed_pert_robustness = []
	source_images = []
	pert_images = []

	# Judge success rate
	for i in range(num_samples):
		if True not in np.isnan(stu_pert_images[i]):
			# L2 distance between pert_image and source_image
			pert_amount = np.linalg.norm(stu_pert_images[i] - stu_source_images[i])

			if pert_amount <= pert_threshold:
				pert_robustness = pert_amount / np.linalg.norm(stu_source_images[i])

				succeed_count += 1
				succeed_pert_amount.append(pert_amount)
				succeed_pert_robustness.append(pert_robustness)

				source_images.append(stu_source_images[i])
				pert_images.append(stu_pert_images[i])

	# Change list into numpy array
	succeed_pert_amount = np.array(succeed_pert_amount, dtype=np.float32)
	succeed_pert_robustness = np.array(succeed_pert_robustness, dtype=np.float32)

	# Save the final result of complete attack
	result = ResultBuilder()
	result['Dataset'] = config['dataset']
	result['Classifier'] = classifier
	result['Num of samples'] = num_samples
	result['Pert threshold'] = pert_threshold

	# Attacker params
	result['Optimizer config'] = optimizer_config

	# Attack results
	result['Success rate on original model'] = tch_succeed_count / num_samples
	result['Success rate on robust model'] = succeed_count / num_samples
	result['Average pert amount'] = succeed_pert_amount.mean()
	result['Pert amount standard deviation'] = succeed_pert_amount.std()
	result['Average pert robustness'] = succeed_pert_robustness.mean()
	result['Pert robustness standard deviation'] = succeed_pert_robustness.std()

	# Print and save results
	print(result)
	path = result.save('./results/cw/')
	np.savez_compressed(path + 'source_images.npz', source_images=np.array(source_images, dtype=np.float32))
	np.savez_compressed(path + 'pert_images.npz', pert_images=np.array(pert_images, dtype=np.float32))
