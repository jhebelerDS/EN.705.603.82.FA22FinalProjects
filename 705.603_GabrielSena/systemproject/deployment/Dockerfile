FROM tf_colmap AS builder

# Image Arguments / Variables
ARG model_repo
ARG model_name
ARG scenes_base_dir=$model_name-scenes
ARG scene_name
ARG mpi_dir=$scenes_base_dir/mpis

# Image Environment Variables for reuse
ENV scene_dir $scenes_base_dir/$scene_name
ENV mpi_dir $mpi_dir

RUN mkdir /host
WORKDIR /host

# Clone model to preprocess
RUN git clone $model_repo 
WORKDIR /host/$model_name

# Script to run in image
ADD preprocess.sh preprocess.sh

# Pipe to dev null so we can continuously run new scenes
# and models by way of triggering commands on the image
ENTRYPOINT ["tail", "-f", "/dev/null"]
