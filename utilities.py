import os
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np

from SCAE.tools.model import Attacker
from SCAE.tools.utilities import DatasetHelper, ResultBuilder
from SCAE.train import Configs

# File paths
snapshot_ori = '../SCAE/checkpoints/{}/model.ckpt'
snapshot_kmeans_ori = '../SCAE/checkpoints/{}/kmeans_{}/model.ckpt'
snapshot_rob = './checkpoints/{}/model.ckpt'
snapshot_kmeans_rob = './checkpoints/{}/kmeans_{}/model.ckpt'
dataset_path = '../SCAE/datasets/'
gtsrb_dataset_path = '../SCAE/datasets/GTSRB-for-SCAE_Attack/GTSRB/'
ori_pert_amount_file_path = '../ORI_SCAE/results/{}/{}/{}/ori_pert_amount.npz'


def attack(
		model,
		kmeans,
		attacker,
		classifier,
		dataset,
		num_samples,
		result=None,
		result_prefix=''
) -> (ResultBuilder, ...):
	succeed_pert_amount = []
	succeed_pert_robustness = []
	source_images = []
	pert_images = []

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
			if remain > 0:
				remain -= 1
				if not right_classification[i]:
					succeed_pert_amount.append(0)
					succeed_pert_robustness.append(0)
					source_images.append(images[i])
					pert_images.append(images[i])
				elif True not in np.isnan(attacker_outputs[i]):
					# L2 distance between pert_image and source_image
					pert_amount = np.linalg.norm(attacker_outputs[i] - images[i])
					pert_robustness = pert_amount / np.linalg.norm(images[i])

					succeed_pert_amount.append(pert_amount)
					succeed_pert_robustness.append(pert_robustness)
					source_images.append(images[i])
					pert_images.append(attacker_outputs[i])
			else:
				break

		print('Up to now: Success rate: {:.4f}. Average pert amount: {:.4f}. Remain: {}.'.format(
			len(succeed_pert_amount) / (num_samples - remain), np.array(succeed_pert_amount, dtype=np.float32).mean(),
			remain))

	# Change list into numpy array
	all_pert_amount: list = succeed_pert_amount.copy()
	all_pert_amount.extend([np.inf] * (num_samples - len(succeed_pert_amount)))
	succeed_pert_amount = np.array(succeed_pert_amount, dtype=np.float32)
	succeed_pert_robustness = np.array(succeed_pert_robustness, dtype=np.float32)

	# Save the final result of complete attack
	if result is None:
		result = ResultBuilder()
	result[f'{result_prefix}Success rate'] = len(succeed_pert_amount) / num_samples
	result[f'{result_prefix}Average pert amount'] = succeed_pert_amount.mean()
	result[f'{result_prefix}Pert amount standard deviation'] = succeed_pert_amount.std()
	result[f'{result_prefix}Average pert robustness'] = succeed_pert_robustness.mean()
	result[f'{result_prefix}Pert robustness standard deviation'] = succeed_pert_robustness.std()

	return result, source_images, pert_images, all_pert_amount


def remove_make_dirs(path):
	if os.path.exists(path):
		rmtree(path)
		print(f'INFO: Path {path} has been removed.')
	os.makedirs(path)


def load_dataset(config, batch_size):
	return DatasetHelper(config['dataset'],
	                     'train' if config['dataset'] == Configs.GTSRB
	                                or config['dataset'] == Configs.FASHION_MNIST else 'test',
	                     file_path=dataset_path, batch_size=batch_size, shuffle=True, fill_batch=True,
	                     normalize=True if config['dataset'] == Configs.GTSRB else False,
	                     gtsrb_raw_file_path=gtsrb_dataset_path, gtsrb_classes=Configs.GTSRB_CLASSES)


def draw_cumulative_distribution(xmax, labels, data, title=None, file_path=None):
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
	if title:
		plt.title(title)

	if file_path is not None:
		figure = plt.gcf()
		plt.show()
		figure.savefig(file_path, bbox_inches='tight')
	else:
		plt.show()


def draw_accuracy_variation(n_epoch, labels, data, title=None, file_path=None):
	r_epoch = range(0, n_epoch + 1)
	for i in range(len(labels)):
		plt.plot(r_epoch, [0, *data[i]], label=labels[i])

	plt.xlim((0, n_epoch))
	plt.xlabel('Epoch')
	plt.ylim((0, 1))
	plt.ylabel('Accuracy')
	plt.legend(loc=2)
	plt.grid()
	if title:
		plt.title(title)

	if file_path is not None:
		figure = plt.gcf()
		plt.show()
		figure.savefig(file_path, bbox_inches='tight')
	else:
		plt.show()
