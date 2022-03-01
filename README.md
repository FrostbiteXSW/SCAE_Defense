# **Towards Robust Stacked Capsule Autoencoder with Hybrid Adversarial Training**

This is the official source code of the paper: ["Towards Robust Stacked Capsule Autoencoder with Hybrid Adversarial Training"](https://arxiv.org/abs/2202.13755).

* **Author**: Jiazhu Dai, Siwei Xiong
* **Institution**: Shanghai University
* **Email**: daijz@shu.edu.cn (J. Dai)

Executable .py files in *_SCAE folders:

* *train\_\*.py*: Train the SCAE model with the specified defense method and save it under *_SCAE/checkpoints folder. Dataset cache files are saved under SCAE/datasets folder.
* *test\_\*.py*: Test the model under *_SCAE/checkpoints folder, generate and save the K-Means classifier at the same place.
* *attack_opt\_\*.py*: Launch the attack with OPT algorithm using the model under *_SCAE/checkpoints folder. Results are saved under *_SCAE/results/opt folder.

Please feel free to use it as you like.