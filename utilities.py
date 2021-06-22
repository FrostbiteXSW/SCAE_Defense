import matplotlib.pyplot as plt
import numpy as np

from SCAE.tools.model import Attacker
from SCAE.tools.utilities import DatasetHelper
from SCAE.train import Configs

# File paths
snapshot_ori = '../SCAE/checkpoints/{}/model.ckpt'
snapshot_kmeans_ori = '../SCAE/checkpoints/{}/kmeans_{}/model.ckpt'
snapshot_rob = './checkpoints/{}/model.ckpt'
snapshot_kmeans_rob = './checkpoints/{}/kmeans_{}/model.ckpt'
dataset_path = 'SCAE/datasets/'
gtsrb_dataset_path = 'SCAE/datasets/GTSRB-for-SCAE_Attack/GTSRB/'


def attack(
		model,
		kmeans,
		attacker,
		classifier,
		dataset,
		num_samples,
		succeed_pert_amount_list,
		succeed_pert_robustness_list,
		source_image_list,
		pert_image_list,
		pert_amount_list
):
	succeed_count = 0

	dataset = iter(dataset)
	remain = num_samples
	while remain > 0:
		images, labels = next(dataset)

		# Judge classification
		if classifier[-1].upper() == 'K':
			right_classification = kmeans(images) == labels
		else:
			right_classification = model.run(
				images=images,
				to_collect=model._res.prior_cls_pred if classifier == Attacker.Classifiers.PriL
				else model._res.posterior_cls_pred
			) == labels

		attacker_outputs = attacker(images, labels, nan_if_fail=True, verbose=True)

		for i in range(len(attacker_outputs)):
			if right_classification[i] and remain:
				remain -= 1
				if True not in np.isnan(attacker_outputs[i]):
					# L2 distance between pert_image and source_image
					pert_amount = np.linalg.norm(attacker_outputs[i] - images[i])
					pert_robustness = pert_amount / np.linalg.norm(images[i])

					succeed_count += 1
					succeed_pert_amount_list.append(pert_amount)
					succeed_pert_robustness_list.append(pert_robustness)

					source_image_list.append(images[i])
					pert_image_list.append(attacker_outputs[i])

					pert_amount_list.append(pert_amount)
				else:
					pert_amount_list.append(np.inf)

		print('Up to now: Success rate: {:.4f}. Average pert amount: {:.4f}. Remain: {}.'.format(
			succeed_count / (num_samples - remain), np.array(succeed_pert_amount_list, dtype=np.float32).mean(), remain))

	return succeed_count


def load_dataset(config, batch_size):
	return DatasetHelper(config['dataset'],
	                     'train' if config['dataset'] == Configs.GTSRB
	                                or config['dataset'] == Configs.FASHION_MNIST else 'test',
	                     file_path='SCAE/datasets', batch_size=batch_size, shuffle=True, fill_batch=True,
	                     normalize=True if config['dataset'] == Configs.GTSRB else False,
	                     gtsrb_raw_file_path=gtsrb_dataset_path, gtsrb_classes=Configs.GTSRB_CLASSES)


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
