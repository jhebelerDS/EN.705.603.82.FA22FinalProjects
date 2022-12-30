#!/bin/bash
# Preprequisite images
echo "Pulling required images..."
docker pull bmild/tf_colmap
docker tag bmild/tf_colmap tf_colmap

# Module repo that allows uses to preprocess, generate mpis, and render video
export MODEL_REPO=https://github.com/Fyusion/LLFF.git
export MODEL_NAME=LLFF
export SCENE_NAME=mr_skeleton_man
export IMAGE_NAME=viewsynthesis

# STEP 1: Trigger nvidia-docker to build image
echo "Building image: $IMAGE_NAME..."
nvidia-docker build -t $IMAGE_NAME \
	--build-arg model_repo=$MODEL_REPO \
	--build-arg model_name=$MODEL_NAME \
	--build-arg scene_name=$SCENE_NAME .

# STEP 2: Run image in the background
echo "Running $IMAGE_NAME in the background..."
nvidia-docker run -d --name $IMAGE_NAME \
  -v $PWD/scenes/:/host/$MODEL_NAME/$MODEL_NAME-scenes \
  -v $PWD/checkpoints/:/host/$MODEL_NAME/checkpoints $IMAGE_NAME

# STEP 3: Run command on running image
# NOTE: We can run multiple commands on the image now that
# we are running the image in the background
echo "Generating camera poses, mpis, and video from static images..."
nvidia-docker exec $IMAGE_NAME sh preprocess.sh
# Run NeRF, NeuMan, or other models etc.,

echo "Cleaning up image: $IMAGE_NAME..."
nvidia-docker stop $IMAGE_NAME
nvidia-docker rm $IMAGE_NAME
echo "PROCESS COMPLETE."