#!/bin/sh
PARTITION=Segmentation

GPU_ID=0
mode=Test
dataset=coco # pascal coco fss
exp_name=split0

arch=SAM_RSP
net=resnet50 # vgg resnet50 resnet101


#time=2023-10-28_13时14分20秒
now=$(date +"%Y-%m-%d_%X")
exp_path=exp/${mode}/${dataset}/${arch}/${exp_name}/${net}/${now}
snapshot_path=${exp_path}/snapshot
result_path=${exp_path}/result
show_path=${exp_path}/show
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml
mkdir -p ${snapshot_path} ${result_path} ${show_path}

cp test.sh test.py ${config} ${exp_path}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u test.py \
        --config=${config} \
        --arch=${arch} \
        --exp_path=${exp_path}\
        --snapshot_path=${snapshot_path} \
        --result_path=${result_path} \
        --show_path=${show_path}\
        2>&1 | tee ${result_path}/test-$now.log