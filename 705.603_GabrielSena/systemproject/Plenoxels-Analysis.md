# PLENOXELS
### About
Plenoxels represent a scene as a sparse 3D grid with spherical harmonics which then can be optimized from images via gradient methods and regularization without any neural components. 
Plenoxels train orders of magnitude faster than Neural Radiance Fields with barely any loss in realistic quality of 
rendered novel views.

_Plenoxels Github Project found [here](https://github.com/sxyu/svox2)_.

### Successes

Using Ubuntu 22.04 with NVDIA/Drivers I was able to have a successful training of the [Lego dataset](https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a).

The setup of the conda environment did not succeed out of the box so I had to do the following to get 
the project working:

##### Steps:
1. Install the svox2 library on a new conda environment using python==3.8 (i.e `$ conda create -n plenoxels python=3.9; pip install .`)
2. Inside the `plenoxels` conda environment install all dependencies from the `environment.yml` file.
3. Run the launch script and train `./launch.sh lego-experiment 0 ../data/nerf_synthetic/lego -c configs/syn.json`

_Example Run:_
``` 
(plenoxel) ubuntu@ubuntu-Alienware-15-R3:~/developer/projects/svox2/opt$ ./launch.sh lego-experiment 0 ../data/nerf_synthetic/lego -c configs/syn.json
Launching experiment lego-experiment
GPU 0
EXTRA ../data/nerf_synthetic/lego -c configs/syn.json
CKPT ckpt/lego-experiment
LOGFILE ckpt/lego-experiment/log
DETACH
```
From there, you can tail -f the ckpt/lego-experiment/log file.

**IMPORTANT**: 
- I faced out of memory errors when initially running the Lego dataset. 
To resolve this you can lower the resolution in the [syn.json](https://github.com/sxyu/svox2/blob/master/opt/configs/syn.json)
For me, I divided everything in half and ended up with:
``` 
{
    "reso": "[[128, 128, 128], [256, 256, 256]]",
    ...
```
- Another issue that I encountered was a libg1.so error. To fix, on Ubuntu, you can run the following:
`sudo apt-get update && sudo apt-get install libgl1`

### Failures
#### Limitations on Apple 1

Similar to the issues raised in [Light Field Neural Rendering](LightFieldNeuralRendering-Analysis.md), jax/iree
compatibility issues were present.

### Citations
``` 
@misc{https://doi.org/10.48550/arxiv.2112.05131,
  doi = {10.48550/ARXIV.2112.05131},
  url = {https://arxiv.org/abs/2112.05131},
  author = {Yu, Alex and Fridovich-Keil, Sara and Tancik, Matthew and Chen, Qinhong and Recht, Benjamin and Kanazawa, Angjoo},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Graphics (cs.GR), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Plenoxels: Radiance Fields without Neural Networks},
  publisher = {arXiv},
  year = {2021},  
  copyright = {Creative Commons Attribution 4.0 International}
}
```