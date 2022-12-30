# Polynomial Distribution using Transformer Encoder-Decoder Architecture

Programmer: Chris Vaisnor

Python: 3.8.13

PyTorch: 1.13.0

PyTorch Lightning: 1.7.7

# For a project overview, please see: 
* main_notebook.pdf (includes images for viewing in GitHub)
* or
* main_notebook.ipynb (images may not load in the GitHub browser. If you clone the repo, they will load in Jupyter Notebook).

The Docker image runs main_script.py which loads the pretrained model and predicts on a custom user input.
DockerHub Image: 705.603_chrisvaisnor:system_project

Link: https://hub.docker.com/repository/docker/cvaisnor/705.603_chrisvaisnor

# Dashboard:
The Weights and Biases (Wandb) dashboard can be seen here: https://wandb.ai/chrisvaisnor/system_project?workspace=user-chrisvaisnor. This displays a number of graphs for comparing model configurations as well as showing showing system performance information. 

The default model configuration is the RED label corresponding to (hid_dim=256, layers=3, batch_size=512).

# Compatibility:
Training and testing is only supported on a CUDA-enabled GPU. For comparison purposes, I tried training it on my 3.6Ghz Intel i9, 10 core CPU in the default model configuration and it takes almost 50 minutes for one epoch. This would lead to an approximate total training time of at **LEAST 7.5 hours.**

For comparison, my Nvidia RTX 3060 12GB GPU trains the model with **10 epochs in about 25 minutes.**

# To load the pretrained model and predict on a custom input:
```python
python3 main_script.py
```

# To train a model from scratch (NOT RECOMMENDED):
```python
python3 model.py --max_epochs=(Any int) # default is 10
```

# To visualize the PyTorch Lightning Trainer process:
```python
python3 model.py --fast_dev_run
```