#!/bin/bash
python imgs2poses.py $scene_dir \
	&& python imgs2mpis.py $scene_dir $mpi_dir --height 360 \
	&& python imgs2renderpath.py $scene_dir $scene_dir/spiral_path.txt --spiral \
	&& python mpis2video.py $mpi_dir $scene_dir/spiral_path.txt $scene_dir/spiral_render.mp4 --crop_factor 0.8
