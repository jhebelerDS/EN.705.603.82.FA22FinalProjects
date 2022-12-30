# System Project - Gabriel Sena

### Objective:

#### The goal of this project is to create an application that interfaces with view synthesis models. Mentioned in the following [blog post](https://dellaert.github.io/NeRF22/), there are more than 50 papers related to NeRF (Neural Radiance Fields) alone. With that, a lot of papers also contain open source code implementation of the concepts. As more view synthesis models arise (for various use cases) it will be imperative to have a simple design that can work with the various model design and build requirements.

---

This Project has tested the following models:
- [NeRF: Neural Radiance Fields](https://www.matthewtancik.com/nerf)
- [Plenoxels](https://github.com/sarafridov/plenoxels)
- [Local Light Field Fusion](https://github.com/Fyusion/LLFF)
- [Light Field Neural Rendering](https://github.com/google-research/google-research/tree/master/light_field_neural_rendering)
- [NeuMan: Neural Numan Radiance Field from a Single Video](https://machinelearning.apple.com/research/neural-human-radiance-field)

## Working Demo / System Project

The working demo is found [here](deployment). Please follow the [README.md](deployment/README.md) to ensure you have the appropriate setup.

## Data Collection

- _Personal Images_ 
    - [Mr Skeleton Man](data/mr_skeleton_man)
- _Online Datasets_
    - [NeRF Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
    - [NeRF Project Dir](https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/)
    - [NeuMan Video Dataset](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/dataset.zip)

Data collected has followed guidelines from [LLFF preprocessing](https://github.com/Fyusion/LLFF#1-recover-camera-poses) steps.
Online datasets follow similar setup where we have access to camera poses and metadata needed for training.

For my personal image, I used an iphone with each image being roughly 1.5mb each. I also chose to shoot in dim light
to see if the model can also generate successful novel views from my set.

## Other Models Analysis, Findings, and Limitations
The following pages are models that I have experimented with. Some I have success running while
others there are limitations which are described in detail.

- [NeuMan](NeuMan-Analysis.md)
- [Light Field Neural Rendering](LightFieldNeuralRendering-Analysis.md)
- [NeRF: Neural Radiance Fields](NeRF-Analysis.md)
- [Plenoxels](Plenoxels-Analysis.md)

## Citations
1. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. ImageNet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer  Vision and Pattern Recognition. 248–255. https://doi.org/10.1109/CVPR.2009.5206848
2. John Hart. 1995. Sphere Tracing: A Geometric Method for the Antialiased Ray Tracing of Implicit Surfaces. The Visual Computer 12 (06 1995). https://doi.org/10.1007/s003710050084
3. Jeffrey Ichnowski, Yahav Avigal, Justin Kerr, and Ken Goldberg. 2021. Dex- NeRF: Using a Neural Radiance Field to Grasp Transparent Objects. (2021). https://doi.org/10.48550/ARXIV.2110.14217
4. Wei Jiang, Kwang Moo Yi, Golnoosh Samei, Oncel Tuzel, and Anurag Ranjan. 2022. NeuMan: Neural Human Radiance Field from a Single Video. (2022). https://doi.org/10.48550/ARXIV.2203.12575
5. Abhijit Kundu, Kyle Genova, Xiaoqi Yin, Alireza Fathi, Caroline Pantofaru, Leonidas Guibas, Andrea Tagliasacchi, Frank Dellaert, and Thomas Funkhouser. 2022. Panoptic Neural Fields: A Semantic Object-Aware Neural Scene Representation. (2022). https://doi.org/10.48550/ARXIV.2205.04334
6. Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. 2015. SMPL: A Skinned Multi-Person Linear Model. 34, 6 (oct 2015), 248:1–248:16.
7. Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. (2020). https://doi.org/10.48550/ARXIV.2003.08934
8. Konstantinos Rematas, Andrew Liu, Pratul P. Srinivasan, Jonathan T. Barron, Andrea Tagliasacchi, Thomas Funkhouser, and Vittorio Ferrari. 2021. Urban Radiance Fields. (2021). https://doi.org/10.48550/ARXIV.2111.14643
9. Vincent Sitzmann, Michael Zollhöfer, and Gordon Wetzstein. 2019. Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations. (2019). https://doi.org/10.48550/ARXIV.1906.01618
10. Mohammed Suhail, Carlos Esteves, Leonid Sigal, and Ameesh Makadia. 2021. Light Field Neural Rendering. (2021). https://doi.org/10.48550/ARXIV.2112.09687
11. Alex Yu, Sara Fridovich-Keil, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. 2021. Plenoxels: Radiance Fields without Neural Networks. (2021). https://doi.org/10.48550/ARXIV.2112.05131
