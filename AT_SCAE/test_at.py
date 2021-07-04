import os

from SCAE.test import test
from utilities import *

# File paths
snapshot = './checkpoints/{}/model.ckpt'
snapshot_kmeans_pri = './checkpoints/{}/kmeans_pri/model.ckpt'
snapshot_kmeans_pos = './checkpoints/{}/kmeans_pos/model.ckpt'

if __name__ == '__main__':
	Configs.GTSRB_DATASET_PATH = gtsrb_dataset_path

	config = Configs.config_mnist
	snapshot = snapshot.format(config['dataset'])
	snapshot_kmeans_pri = snapshot_kmeans_pri.format(config['dataset'])
	snapshot_kmeans_pos = snapshot_kmeans_pos.format(config['dataset'])
	train_and_save_kmeans = not (os.path.exists(snapshot_kmeans_pri[:snapshot_kmeans_pri.rindex('/')])
	                             and os.path.exists(snapshot_kmeans_pos[:snapshot_kmeans_pos.rindex('/')]))

	model, kmeans_pri, kmeans_pos, trainset, testset = test(
		config=config,
		scope='SCAE',
		snapshot=snapshot,
		snapshot_kmeans_pri=snapshot_kmeans_pri,
		snapshot_kmeans_pos=snapshot_kmeans_pos,
		dataset_path=dataset_path,
		train_and_save_kmeans=train_and_save_kmeans
	)
