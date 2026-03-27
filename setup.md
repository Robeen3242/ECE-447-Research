# Environment Setup
 
The following instructions describe how to configure a local Python environment to run the experiments in this project.
 
## Prerequisites
 
Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your operating system.
 
## 1. Create and Activate a Conda Environment
 
```bash
conda create --name gcnn_env python=3.10
conda activate gcnn_env
```
 
Replace `gcnn_env` with your preferred environment name.
 
## 2. Install Dependencies
 
Once the environment is active, install the required packages:
 
```bash
pip install torch torchvision
pip install matplotlib
pip install ipywidgets
pip install seaborn
pip install plotly
```
 
> **Note:** If you intend to use GPU acceleration, install the appropriate CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before proceeding with the remaining packages.
 
## 3. Verify Installation
 
You can verify the core packages were installed correctly by running:
 
```bash
python -c "import torch; print(torch.__version__)"
```
 