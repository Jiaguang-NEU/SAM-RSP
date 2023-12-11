#!/bin/sh
PARTITION=Segmentation

GPU_ID=1
mode=Train
dataset=coco # pascal coco
exp_name=split0

arch=SAM_RSP
net=resnet50 # vgg resnet50
now=$(date +"%Y-%m-%d_%X")
exp_path=exp/${mode}/${dataset}/${arch}/${exp_name}/${net}/${now}
snapshot_path=${exp_path}/snapshot
result_path=${exp_path}/result
show_path=${exp_path}/show
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml
mkdir -p ${snapshot_path} ${result_path} ${show_path}
cp train.sh train.py ${config} ${exp_path}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=6543 train.py \
        --config=${config} \
        --arch=${arch} \
        --exp_path=${exp_path}\
        --snapshot_path=${snapshot_path} \
        --result_path=${result_path} \
        --show_path=${show_path}\
        2>&1 | tee ${result_path}/train-$now.log