from SCAE.attack_cw import AttackerCW
from SCAE.tools.model import KMeans
from SCAE.tools.utilities import block_warnings
from SCAE.train import build_from_config
from utilities import *


# File paths
result_path = './results/cw/'


def build_all(
		config,
		classifier,
		batch_size,
		optimizer_config,
		const_init,
		scope,
		snapshot,
		snapshot_kmeans
):
	snapshot = snapshot.format(config['name'])
	snapshot_kmeans = snapshot_kmeans.format(config['name'], classifier[:3].lower())

	model = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=False,
		snapshot=snapshot,
		scope=scope
	)

	if classifier[-1].upper() == 'K':
		kmeans = KMeans(
			scae=model,
			kmeans_type=KMeans.KMeansTypes.Prior if classifier[:3].upper() == 'PRI' else KMeans.KMeansTypes.Posterior,
			is_training=False,
			scope='KMeans_Pri' if classifier[:3].upper() == 'PRI' else 'KMeans_Pos',
			snapshot=snapshot_kmeans
		)
	else:
		kmeans = None

	attacker = AttackerCW(
		scae=model,
		classifier=classifier,
		kmeans_classifier=kmeans,
		optimizer_config=optimizer_config,
		const_init=const_init
	)

	model.finalize()

	return model, kmeans, attacker


if __name__ == '__main__':
	block_warnings()

	# Attack configuration
	config = Configs.config_mnist
	optimizer_config = AttackerCW.OptimizerConfigs.Adam_fast
	num_samples = 1000
	batch_size = 100
	classifier = Attacker.Classifiers.PosK
	use_mask = True

	# Create original model
	model_ori, kmeans_ori, attacker_ori = build_all(config=config,
	                                                classifier=classifier,
	                                                batch_size=batch_size,
	                                                optimizer_config=optimizer_config,
	                                                const_init=1e2,
	                                                scope='SCAE',
	                                                snapshot=snapshot_ori,
	                                                snapshot_kmeans=snapshot_kmeans_ori)

	# Load dataset
	dataset = load_dataset(config, batch_size)

	# Classification accuracy test
	print('Testing original model...')
	model_ori.simple_test(dataset)

	# Start the attack on the original model
	print('\nStart the attack on the original model...')
	result, ori_source_images, ori_pert_images, ori_pert_amount = attack(
		model=model_ori, kmeans=kmeans_ori, attacker=attacker_ori, classifier=classifier, dataset=dataset,
		num_samples=num_samples, result_prefix='[ORI]')

	# Attack settings
	result['Dataset'] = config['name']
	result['Classifier'] = classifier
	result['Num of samples'] = num_samples

	# Attacker params
	result['Optimizer config'] = optimizer_config

	# Print and save results (without datetime)
	print(result)
	path = result.save(os.path.join(result_path, f'{config["name"]}/{classifier}/'), plus_time=False)
	np.savez_compressed(os.path.join(path, 'ori_source_images.npz'),
	                    source_images=np.array(ori_source_images, dtype=np.float32))
	np.savez_compressed(os.path.join(path, 'ori_pert_images.npz'),
	                    pert_images=np.array(ori_pert_images, dtype=np.float32))
	np.savez_compressed(os.path.join(path, 'ori_pert_amount.npz'),
	                    pert_amount=np.array(ori_pert_amount, dtype=np.float32))

	# Draw plot
	draw_cumulative_distribution(5, ['Original Model'], [ori_pert_amount],
	                             title='Pert Amount Cumulative Distribution',
	                             file_path=os.path.join(path, 'result_plot.png'))
