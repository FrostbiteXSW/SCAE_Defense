from ORI_SCAE.attack_cw_ori import build_all, result_path
from SCAE.attack_cw import AttackerCW
from SCAE.tools.utilities import block_warnings, load_npz
from utilities import *


if __name__ == '__main__':
	block_warnings()

	# Attack configuration
	config = Configs.config_mnist
	optimizer_config = AttackerCW.OptimizerConfigs.Adam_fast
	num_samples = 1000
	batch_size = 100
	classifier = Attacker.Classifiers.PosK
	use_mask = True

	# Create robust model
	model_rob, kmeans_rob, attacker_rob = build_all(config=config,
	                                                classifier=classifier,
	                                                batch_size=batch_size,
	                                                optimizer_config=optimizer_config,
	                                                const_init=1e2,
	                                                scope='STU',
	                                                snapshot=snapshot_rob,
	                                                snapshot_kmeans=snapshot_kmeans_rob)
	# Load dataset
	dataset = load_dataset(config, batch_size)

	# Classification accuracy test
	print('\nTesting robust model...')
	model_rob.simple_test(dataset)

	# Start the attack on the robust model
	print('\nStart the attack on the robust model...')
	result, rob_source_images, rob_pert_images, rob_pert_amount = attack(
		model=model_rob, kmeans=kmeans_rob, attacker=attacker_rob, classifier=classifier, dataset=dataset,
		num_samples=num_samples, result_prefix='[ROB]')

	# Attack settings
	result['Dataset'] = config['name']
	result['Classifier'] = classifier
	result['Num of samples'] = num_samples

	# Attacker params
	result['Optimizer config'] = optimizer_config

	# Print and save results
	print(result)
	path = result.save(result_path)
	np.savez_compressed(os.path.join(path, 'rob_source_images.npz'),
	                    source_images=np.array(rob_source_images, dtype=np.float32))
	np.savez_compressed(os.path.join(path, 'rob_pert_images.npz'),
	                    pert_images=np.array(rob_pert_images, dtype=np.float32))

	# Draw plot
	labels = ['Robust Model']
	data = [rob_pert_amount]
	format_path = ori_pert_amount_file_path.format('cw', config['name'], classifier)
	if os.path.exists(format_path):
		ori_pert_amount = load_npz(format_path)['pert_amount']
		labels.insert(0, 'Original Model')
		data.insert(0, ori_pert_amount)
	draw_cumulative_distribution(5, labels=labels, data=data, title='Pert Amount Cumulative Distribution',
	                             file_path=os.path.join(path, 'result_plot.png'))
