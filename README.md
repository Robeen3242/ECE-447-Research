# ECE-447-Research

This goal of this repository is to simulate structured research experience in modern machine learning, specifically regarding the study on Group Equivariant Convolutional Networks by Taco S. Cohen and Max Welling.

## Our Repo
In the root folder are our All-CNN style CIFAR-10 models `cnngeneral.py`, `gcnngeneral.py`, and `gcnn_p4mgeneral.py` as well as the paper we based our experiment on. Our prototype/legacy code is in the `old_models` folder, which are the models we used to produce the results for our presentation. For our raw experiments, we chose to opt for the `___general.py` files instead. Their metrics and `.pth` files can be found in the `models` folder. 

All of the code for our experiments can be found in the `EXPERIMENTS` folder. Each experiment is setup at your convenience, including the `.pth` and `.py` folders for easy portability into collab, which is primarily where we ran our experiments on. Each experiment folder contains a `results` folder showcasing our outputs from running the `.ipynb` files, which incorporate the .pth and .py files into one place to produce our experiment results.

The instructions for installing the correct dependencies and setting up the environment is in the setup.md file.

## Differences from the original paper

- We did not use the paper's exact rotated-MNIST benchmark.
- We built a custom rotated-MNIST dataset from standard MNIST.
- Our rotated-MNIST test set had 10,000 samples, not 50,000.
- We used a 4-stage backbone instead of the paper's 7-layer model.
- We included p4m and an early-pooling ablation in our comparison.
- Our training setup differed in epochs, scheduling, and optimization details.
- Our use of batch normalization was not identical to the paper's setup.
- We used a simplified All-CNN-style CIFAR-10 model.
- We did not fully reproduce the paper's ResNet44 experiments.
- We used a 45,000/5,000 CIFAR-10 train-validation split.
- In code, our CIFAR-10 loaders use `ToTensor()` and `Normalize(...)` as the default transform, without reproducing the paper's flip and translation augmentations.
- We trained for fewer epochs than some of the paper's experiments.
- We added rotation-robustness and filter-visualization experiments.
- Our results reproduce the main idea, not the exact paper numbers.
