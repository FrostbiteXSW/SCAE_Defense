import os

from SCAE.test import test
from utilities import Configs

if __name__ == '__main__':
	config = Configs.config_mnist
	snapshot = './checkpoints/{}/model.ckpt'.format(config['dataset'])
	snapshot_kmeans_pri = './checkpoints/{}/kmeans_pri/model.ckpt'.format(config['dataset'])
	snapshot_kmeans_pos = './checkpoints/{}/kmeans_pos/model.ckpt'.format(config['dataset'])
	train_and_save_kmeans = not (os.path.exists(snapshot_kmeans_pri[:snapshot_kmeans_pri.rindex('/')])
	                             and os.path.exists(snapshot_kmeans_pos[:snapshot_kmeans_pos.rindex('/')]))

	model, kmeans_pri, kmeans_pos, trainset, testset = test(
		config=config,
		snapshot=snapshot,
		snapshot_kmeans_pri=snapshot_kmeans_pri,
		snapshot_kmeans_pos=snapshot_kmeans_pos,
		train_and_save_kmeans=train_and_save_kmeans
	)
