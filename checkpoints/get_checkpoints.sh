#!/bin/bash

base_url=https://neuralalignckpts.s3-us-west-1.amazonaws.com

for model in 18 50 50v2 101v2 152v2
do
  mkdir -p ./resnet${model}/symmetric
  mkdir -p ./resnet${model}/activation
  mkdir -p ./resnet${model}/information_tpe
  mkdir -p ./resnet${model}/mirror
  for alignment in symmetric/model.ckpt-450360 activation/model.ckpt-450360 information_tpe/model.ckpt-1000800
  do
    curl -fLo ./resnet${model}/${alignment}.meta ${base_url}/resnet${model}/${alignment}.meta
    curl -fLo ./resnet${model}/${alignment}.index ${base_url}/resnet${model}/${alignment}.index
    curl -fLo ./resnet${model}/${alignment}.data-00000-of-00001 ${base_url}/resnet${model}/${alignment}.data-00000-of-00001
  done
done

# For mirror we only have ResNet18
model=18
alignment=mirror/model.ckpt-126250
curl -fLo ./resnet${model}/${alignment}.meta ${base_url}/resnet${model}/${alignment}.meta
curl -fLo ./resnet${model}/${alignment}.index ${base_url}/resnet${model}/${alignment}.index
curl -fLo ./resnet${model}/${alignment}.data-00000-of-00001 ${base_url}/resnet${model}/${alignment}.data-00000-of-00001
