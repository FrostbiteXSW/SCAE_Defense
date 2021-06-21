import numpy as np

from SCAE.attack_bim import AttackerBIM
from SCAE.tools.model import Attacker, KMeans
from SCAE.tools.utilities import block_warnings, DatasetHelper, ResultBuilder
from train_dist import Configs, build_from_config
from utilities import draw_pdf

if __name__ == '__main__':
	block_warnings()

	# Attack configuration
	config = Configs.config_mnist
	num_samples = 1000
	batch_size = 100
	classifier = Attacker.Classifiers.PosK
	num_iter = 100
	alpha = 0.05
	use_mask = True

	snapshot_rob = './checkpoints/{}_dist/model.ckpt'.format(config['dataset'])
	snapshot_kmeans_rob = './checkpoints/{}_dist/kmeans_{}/model.ckpt'.format(
		config['dataset'], 'pri' if classifier[:3].upper() == 'PRI' else 'pos')

	snapshot_ori = './SCAE/checkpoints/{}/model.ckpt'.format(config['dataset'])
	snapshot_kmeans_ori = './SCAE/checkpoints/{}/kmeans_{}/model.ckpt'.format(
		config['dataset'], 'pri' if classifier[:3].upper() == 'PRI' else 'pos')

	# Create original model
	model_ori = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=False,
		snapshot=snapshot_ori,
		scope='SCAE'
	)

	if classifier[-1].upper() == 'K':
		kmeans_ori = KMeans(
			scae=model_ori,
			kmeans_type=KMeans.KMeansTypes.Prior if classifier[:3].upper() == 'PRI' else KMeans.KMeansTypes.Posterior,
			is_training=False,
			scope='KMeans_Pri' if classifier[:3].upper() == 'PRI' else 'KMeans_Pos',
			snapshot=snapshot_kmeans_ori
		)

	attacker_ori = AttackerBIM(
		scae=model_ori,
		classifier=classifier,
		kmeans_classifier=kmeans_ori if classifier[-1].upper() == 'K' else None,
		alpha=alpha
	)

	model_ori.finalize()

	# Create robust model
	model_rob = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=False,
		snapshot=snapshot_rob,
		scope='STU'
	)

	if classifier[-1].upper() == 'K':
		kmeans_rob = KMeans(
			scae=model_rob,
			kmeans_type=KMeans.KMeansTypes.Prior if classifier[:3].upper() == 'PRI' else KMeans.KMeansTypes.Posterior,
			is_training=False,
			scope='KMeans_Pri' if classifier[:3].upper() == 'PRI' else 'KMeans_Pos',
			snapshot=snapshot_kmeans_rob
		)

	attacker_rob = AttackerBIM(
		scae=model_rob,
		classifier=classifier,
		kmeans_classifier=kmeans_rob if classifier[-1].upper() == 'K' else None,
		alpha=alpha
	)

	model_rob.finalize()

	# Load dataset
	dataset = DatasetHelper(config['dataset'],
	                        'train' if config['dataset'] == Configs.GTSRB
	                                   or config['dataset'] == Configs.FASHION_MNIST else 'test',
	                        file_path='./SCAE/datasets', batch_size=batch_size, shuffle=True, fill_batch=True,
	                        normalize=True if config['dataset'] == Configs.GTSRB else False,
	                        gtsrb_raw_file_path=Configs.GTSRB_DATASET_PATH, gtsrb_classes=Configs.GTSRB_CLASSES)

	# Variables to save the attack result
	ori_succeed_count = 0
	ori_succeed_pert_amount = []
	ori_succeed_pert_robustness = []
	ori_source_images = []
	ori_pert_images = []

	rob_succeed_count = 0
	rob_succeed_pert_amount = []
	rob_succeed_pert_robustness = []
	rob_source_images = []
	rob_pert_images = []

	# Variables to draw the plot
	ori_pert_amount = []
	rob_pert_amount = []

	# Classification accuracy test
	print('Testing original model...')
	model_ori.simple_test(dataset)
	print('\nTesting robust model...')
	model_rob.simple_test(dataset)

	# Start the attack on the original model
	print('\nStart the attack on the original model...')
	dataset = iter(dataset)
	remain = num_samples
	while remain > 0:
		images, labels = next(dataset)

		# Judge classification
		if classifier[-1].upper() == 'K':
			right_classification = kmeans_ori(images) == labels
		else:
			right_classification = model_ori.run(
				images=images,
				to_collect=model_ori._res.prior_cls_pred if classifier == Attacker.Classifiers.PriL
				else model_ori._res.posterior_cls_pred
			) == labels

		attacker_outputs = attacker_ori(images, labels, num_iter=num_iter, nan_if_fail=True, verbose=True)

		for i in range(len(attacker_outputs)):
			if right_classification[i] and remain:
				remain -= 1
				if True not in np.isnan(attacker_outputs[i]):
					# L2 distance between pert_image and source_image
					pert_amount = np.linalg.norm(attacker_outputs[i] - images[i])
					pert_robustness = pert_amount / np.linalg.norm(images[i])

					ori_succeed_count += 1
					ori_succeed_pert_amount.append(pert_amount)
					ori_succeed_pert_robustness.append(pert_robustness)

					ori_source_images.append(images[i])
					ori_pert_images.append(attacker_outputs[i])

					ori_pert_amount.append(pert_amount)
				else:
					ori_pert_amount.append(np.inf)

		print('Up to now: Success rate: {:.4f}. Average pert amount: {:.4f}. Remain: {}.'.format(
			ori_succeed_count / (num_samples - remain), np.array(ori_succeed_pert_amount, dtype=np.float32).mean(), remain))

	# Start the attack on the robust model
	print('\nStart the attack on the robust model...')
	dataset = iter(dataset)
	remain = num_samples
	while remain > 0:
		images, labels = next(dataset)

		# Judge classification
		if classifier[-1].upper() == 'K':
			right_classification = kmeans_rob(images) == labels
		else:
			right_classification = model_rob.run(
				images=images,
				to_collect=model_rob._res.prior_cls_pred if classifier == Attacker.Classifiers.PriL
				else model_rob._res.posterior_cls_pred
			) == labels

		attacker_outputs = attacker_rob(images, labels, num_iter=num_iter, nan_if_fail=True, verbose=True)

		for i in range(len(attacker_outputs)):
			if right_classification[i] and remain:
				remain -= 1
				if True not in np.isnan(attacker_outputs[i]):
					# L2 distance between pert_image and source_image
					pert_amount = np.linalg.norm(attacker_outputs[i] - images[i])
					pert_robustness = pert_amount / np.linalg.norm(images[i])

					rob_succeed_count += 1
					rob_succeed_pert_amount.append(pert_amount)
					rob_succeed_pert_robustness.append(pert_robustness)

					rob_source_images.append(images[i])
					rob_pert_images.append(attacker_outputs[i])

					rob_pert_amount.append(pert_amount)
				else:
					rob_pert_amount.append(np.inf)

		print('Up to now: Success rate: {:.4f}. Average pert amount: {:.4f}. Remain: {}.'.format(
			rob_succeed_count / (num_samples - remain), np.array(rob_succeed_pert_amount, dtype=np.float32).mean(), remain))

	# Draw plot
	draw_pdf(5, ['Original Model', 'Robust Model'], ori_pert_amount, rob_pert_amount)

	# Change list into numpy array
	ori_succeed_pert_amount = np.array(ori_succeed_pert_amount, dtype=np.float32)
	ori_succeed_pert_robustness = np.array(ori_succeed_pert_robustness, dtype=np.float32)
	rob_succeed_pert_amount = np.array(rob_succeed_pert_amount, dtype=np.float32)
	rob_succeed_pert_robustness = np.array(rob_succeed_pert_robustness, dtype=np.float32)

	# Save the final result of complete attack
	result = ResultBuilder()
	result['Dataset'] = config['dataset']
	result['Classifier'] = classifier
	result['Num of samples'] = num_samples

	# Attacker params
	result['Num of iter'] = num_iter
	result['Alpha'] = str(alpha)

	# Attack results
	result['[ORI]Success rate'] = ori_succeed_count / num_samples
	result['[ORI]Average pert amount'] = ori_succeed_pert_amount.mean()
	result['[ORI]Pert amount standard deviation'] = ori_succeed_pert_amount.std()
	result['[ORI]Average pert robustness'] = ori_succeed_pert_robustness.mean()
	result['[ORI]Pert robustness standard deviation'] = ori_succeed_pert_robustness.std()

	result['[ROB]Success rate'] = rob_succeed_count / num_samples
	result['[ROB]Average pert amount'] = rob_succeed_pert_amount.mean()
	result['[ROB]Pert amount standard deviation'] = rob_succeed_pert_amount.std()
	result['[ROB]Average pert robustness'] = rob_succeed_pert_robustness.mean()
	result['[ROB]Pert robustness standard deviation'] = rob_succeed_pert_robustness.std()

	# Print and save results
	print(result)
	path = result.save('./results/bim/')
	np.savez_compressed(path + 'ori_source_images.npz', source_images=np.array(ori_source_images, dtype=np.float32))
	np.savez_compressed(path + 'ori_pert_images.npz', pert_images=np.array(ori_pert_images, dtype=np.float32))
	np.savez_compressed(path + 'rob_source_images.npz', source_images=np.array(rob_source_images, dtype=np.float32))
	np.savez_compressed(path + 'rob_pert_images.npz', pert_images=np.array(rob_pert_images, dtype=np.float32))
