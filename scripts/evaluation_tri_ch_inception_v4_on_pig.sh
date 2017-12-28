#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes a ResNetV1-50 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_resnet_v1_50_on_typhoon.sh
set -e

# Where the pre-trained ResNetV1-50 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=~/dataset/checkpoints

# model name
MODEL_NAME=inception_v4_base

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=~/dataset/DA_PIG_models/inception_v4

# Where the dataset is saved to.
DATASET_DIR=~/dataset/DA_PIG

# Where the test dataset is saved to.
TESTSET_DIR=~/dataset/Pig

# Set which GPU to use
GPU_OPT=2


# Run evaluation.
while true
do
CUDA_VISIBLE_DEVICES=${GPU_OPT} python Tri_Ch_eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/eval \
  --dataset_name=pig \
  --dataset_split_name=validation \
  --dataset_dir=${TESTSET_DIR} \
  --model_name=${MODEL_NAME} \
  --eval_image_size=299
sleep 600
done
## Run evaluation.
#CUDA_VISIBLE_DEVICES=${GPU_OPT} python test_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR}/all \
#  --eval_dir=${TRAIN_DIR}/all \
#  --dataset_name=pig \
#  --dataset_split_name=test \
#  --dataset_dir=${TESTSET_DIR} \
#  --model_name=${MODEL_NAME}
