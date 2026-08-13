#!/bin/bash

experiment="${1:?usage: make_dirs.sh <experiment>}"
base_path="${OPS_BASE_PATH:?OPS_BASE_PATH is not set}"

full_path=${base_path}/${experiment}

if [ -d "$full_path" ]; then
    echo "The directory '$experiment' exists."
else
    echo "Creating directory '$experiment'."
    mkdir $full_path
    mkdir ${full_path}/0-convert
    mkdir ${full_path}/0-convert/live_imaging
    mkdir ${full_path}/0-convert/in_situ_sequencing

    mkdir ${full_path}/1-preprocess
    mkdir ${full_path}/1-preprocess/live_imaging
    mkdir ${full_path}/1-preprocess/live_imaging/reconstruction
    mkdir ${full_path}/1-preprocess/live_imaging/virtual_staining
    mkdir ${full_path}/1-preprocess/live_imaging/stitch
    mkdir ${full_path}/1-preprocess/live_imaging/segmentation
    mkdir ${full_path}/1-preprocess/in_situ_sequencing
    mkdir ${full_path}/1-preprocess/in_situ_sequencing/stitch
    mkdir ${full_path}/1-preprocess/in_situ_sequencing/segmentation
    mkdir ${full_path}/1-preprocess/in_situ_sequencing/base_calling

    mkdir ${full_path}/2-tracking
    
    mkdir ${full_path}/3-assembly


fi
