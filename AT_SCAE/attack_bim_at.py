from SCAE.attack_bim import AttackerBIM
from SCAE.tools.model import KMeans
from SCAE.tools.utilities import block_warnings, ResultBuilder
from SCAE.train import build_from_config
from utilities import *

# File paths
result_path = './results/bim/'


def build_all(
		config,
		classifier,
		batch_size,
		alpha,
		num_iter,
		scope,
		snapshot,
		snapshot_kmeans
):
	snapshot = snapshot.format(config['dataset'])
	snapshot_kmeans = snapshot_kmeans.format(config['dataset'], classifier[:3].lower())

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

	attacker = AttackerBIM(
		scae=model,
		classifier=classifier,
		kmeans_classifier=kmeans,
		alpha=alpha,
		num_iter=num_iter
	)

	model.finalize()

	return model, kmeans, attacker


if __name__ == '__main__':
	block_warnings()

	# Attack configuration
	config = Configs.config_mnist
	num_samples = 5000
	batch_size = 100
	classifier = Attacker.Classifiers.PosK
	num_iter = 100
	alpha = 0.05
	use_mask = True

	# Create original model
	model_ori, kmeans_ori, attacker_ori = build_all(config=config,
	                                                classifier=classifier,
	                                                batch_size=batch_size,
	                                                alpha=alpha,
	                                                num_iter=num_iter,
	                                                scope='SCAE',
	                                                snapshot=snapshot_ori,
	                                                snapshot_kmeans=snapshot_kmeans_ori)

	# Create robust model
	model_rob, kmeans_rob, attacker_rob = build_all(config=config,
	                                                classifier=classifier,
	                                                batch_size=batch_size,
	                                                alpha=alpha,
	                                                num_iter=num_iter,
	                                                scope='SCAE',
	                                                snapshot=snapshot_rob,
	                                                snapshot_kmeans=snapshot_kmeans_rob)
	# Load dataset
	dataset = load_dataset(config, batch_size)

	# Classification accuracy test
	print('Testing original model...')
	model_ori.simple_test(dataset)
	print('\nTesting robust model...')
	model_rob.simple_test(dataset)

	# Start the attack on the original model
	print('\nStart the attack on the original model...')
	ori_succeed_pert_amount = []
	ori_succeed_pert_robustness = []
	ori_source_images = []
	ori_pert_images = []
	ori_pert_amount = []
	ori_succeed_count = attack(model=model_ori,
	                           kmeans=kmeans_ori,
	                           attacker=attacker_ori,
	                           classifier=classifier,
	                           dataset=dataset,
	                           num_samples=num_samples,
	                           succeed_pert_amount_list=ori_succeed_pert_amount,
	                           succeed_pert_robustness_list=ori_succeed_pert_robustness,
	                           source_image_list=ori_source_images,
	                           pert_image_list=ori_pert_images,
	                           pert_amount_list=ori_pert_amount)

	# Start the attack on the robust model
	print('\nStart the attack on the robust model...')
	rob_succeed_pert_amount = []
	rob_succeed_pert_robustness = []
	rob_source_images = []
	rob_pert_images = []
	rob_pert_amount = []
	rob_succeed_count = attack(model=model_rob,
	                           kmeans=kmeans_rob,
	                           attacker=attacker_rob,
	                           classifier=classifier,
	                           dataset=dataset,
	                           num_samples=num_samples,
	                           succeed_pert_amount_list=rob_succeed_pert_amount,
	                           succeed_pert_robustness_list=rob_succeed_pert_robustness,
	                           source_image_list=rob_source_images,
	                           pert_image_list=rob_pert_images,
	                           pert_amount_list=rob_pert_amount)

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
	path = result.save(result_path)
	np.savez_compressed(path + 'ori_source_images.npz', source_images=np.array(ori_source_images, dtype=np.float32))
	np.savez_compressed(path + 'ori_pert_images.npz', pert_images=np.array(ori_pert_images, dtype=np.float32))
	np.savez_compressed(path + 'rob_source_images.npz', source_images=np.array(rob_source_images, dtype=np.float32))
	np.savez_compressed(path + 'rob_pert_images.npz', pert_images=np.array(rob_pert_images, dtype=np.float32))

	# Draw plot
	draw_pdf(5, ['Original Model', 'Robust Model'], [ori_pert_amount, rob_pert_amount], file_path=path + 'result_plot.png')
