import os
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
import math
from visdom import Visdom
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
from PIL import Image
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter

from model import SAM_RSP

from util import dataset
from util import transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, \
    get_logger, get_save_path, \
    is_same_model, fix_bn, sum_list, check_makedirs

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
val_manual_seed = 156
val_num = 5
setup_seed(val_manual_seed, False)
seed_array = np.random.randint(0, 1000, val_num)  # seed->[0,999]


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='SAM_RSP')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str,default='/home/data/coco_split0_vgg.yaml',help='config file')  # coco/coco_split0_resnet50.yaml
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,nargs=argparse.REMAINDER)
    parser.add_argument('--exp_path', type=str, default='/home/data/ljg/zw/SAM-RSP-main/exp/Test/pascal/SAM_RSP/split0/vgg')
    parser.add_argument('--snapshot_path', type=str,default='/home/data/ljg/zw/SAM-RSP-main/exp/Test/pascal/SAM_RSP/split0/vgg/snapshot')
    parser.add_argument('--result_path', type=str, default='/home/data/ljg/zw/SAM-RSP-main/exp/Test/pascal/SAM_RSP/split0/vgg/result')
    parser.add_argument('--show_path', type=str, default='/home/data/ljg/zw/SAM-RSP-main/exp/Test/pascal/SAM_RSP/split0/vgg/show')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model(args):
    model = eval(args.arch).OneModel(args)
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    model = model.cuda()

    # Resume
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)
    check_makedirs(args.show_path)

    if args.weight:
        weight_path = args.weight
        if os.path.isfile(weight_path):
            logger.info("=> loading checkpoint '{}'".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(weight_path))

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    print('Number of Parameters: %d' % (total_number))
    print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer

def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0

    logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    logger.info(model)

    # ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Val
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
            val_transform_tri = transform_tri.Compose([
                transform_tri.Resize(size=args.val_size),
                transform_tri.ToTensor(),
                transform_tri.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
            val_transform_tri = transform_tri.Compose([
                transform_tri.test_Resize(size=args.val_size),
                transform_tri.ToTensor(),
                transform_tri.Normalize(mean=mean, std=std)])
        if args.data_set == 'pascal' or args.data_set == 'coco':
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                       base_data_root=args.base_data_root, data_list=args.val_list, \
                                       transform=val_transform, transform_tri=val_transform_tri, mode='val', \
                                       data_set=args.data_set, use_split_coco=args.use_split_coco)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, sampler=None)

    # ----------------------  VAL  ----------------------
    start_time = time.time()
    FBIoU_array = np.zeros(val_num)
    mIoU_array = np.zeros(val_num)
    mIoU_array_b = np.zeros(val_num)
    pIoU_array = np.zeros(val_num)
    for val_id in range(val_num):
        val_seed = seed_array[val_id]
        print('Val: [{}/{}] \t Seed: {}'.format(val_id + 1, val_num, val_seed))
        fb_iou, miou, miou_b, piou = validate(val_loader, model, val_seed)
        FBIoU_array[val_id], mIoU_array[val_id], mIoU_array_b[val_id], pIoU_array[val_id] = \
            fb_iou, miou, miou_b, piou

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    print('\nTotal running time: {}'.format(total_time))
    print('Seed0: {}'.format(val_manual_seed))
    print('Seed:  {}'.format(seed_array))
    print('mIoU:  {}'.format(np.round(mIoU_array, 4)))
    print('mIoU_b:  {}'.format(np.round(mIoU_array_b, 4)))
    print('FBIoU: {}'.format(np.round(FBIoU_array, 4)))
    print('pIoU:  {}'.format(np.round(pIoU_array, 4)))
    print('-' * 43)
    print('Best_Seed_m: {} \t Best_Seed_F: {} \t Best_Seed_p: {}'.format(seed_array[mIoU_array.argmax()],
                                                                         seed_array[FBIoU_array.argmax()],
                                                                         seed_array[pIoU_array.argmax()]))
    print(
        'Best_mIoU: {:.4f} \t Best_mIoU_b: {:.4f} \t Best_FBIoU: {:.4f} \t  Best_pIoU: {:.4f}'.format(mIoU_array.max(),
                                                                                                      mIoU_array_b.max(),
                                                                                                      FBIoU_array.max(),
                                                                                                      pIoU_array.max()))
    print(
        'Mean_mIoU: {:.4f} \t Mean_mIoU_b: {:.4f} \t Mean_FBIoU: {:.4f} \t  Mean_pIoU: {:.4f}'.format(mIoU_array.mean(),
                                                                                                      mIoU_array_b.mean(),
                                                                                                      FBIoU_array.mean(),
                                                                                                      pIoU_array.mean()))


def validate(val_loader, model, val_seed):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter_b = AverageMeter()  # baseline
    union_meter_b = AverageMeter()
    target_meter_b = AverageMeter()

    intersection_meter = AverageMeter()  # final (second)
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    setup_seed(val_seed, args.seed_deterministic)

    if args.data_set == 'pascal':
        test_num = 1000
        split_gap = 5
    elif args.data_set == 'coco':
        test_num = 1000
        split_gap = 20

    class_intersection_meter_b = [0] * split_gap
    class_union_meter_b = [0] * split_gap
    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num / (len(val_loader) - args.batch_size_val))
    iter_num = 0

    for e in range(db_epoch):
        for i, (input, target, target_b, s_input, s_mask, subcls, ori_label, ori_label_b) in enumerate(val_loader):
            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_b = target_b.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            ori_label_b = ori_label_b.cuda(non_blocking=True)


            start_time = time.time()
            output1, output = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, y_b=target_b, cat_idx=subcls)
            model_time.update(time.time() - start_time)


            if args.ori_resize:  # 真值转化为方形
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside, device='cuda') * 255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output1 = F.interpolate(output1, size=target.size()[1:], mode='bilinear', align_corners=True)
            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)

            loss = criterion(output, target)

            output1 = output1.max(1)[1]
            output = output.max(1)[1]

            subcls = subcls[0].cpu().numpy()[0]

            intersection, union, new_target = intersectionAndUnionGPU(output1, target, args.classes, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter_b.update(intersection), union_meter_b.update(union), target_meter_b.update(new_target)
            class_intersection_meter_b[subcls] += intersection[1]
            class_union_meter_b[subcls] += union[1]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
            class_intersection_meter[subcls] += intersection[1]
            class_union_meter[subcls] += union[1]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % round((test_num / 100)) == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

    val_time = time.time() - val_start

    iou_class_b = intersection_meter_b.sum / (union_meter_b.sum + 1e-10)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)

    mIoU_b = np.mean(iou_class_b)
    mIoU = np.mean(iou_class)

    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class_b = []
    # class_iou_class_1 = []
    class_iou_class = []
    class_miou_b = 0
    # class_miou_1 = 0
    class_miou = 0

    for i in range(len(class_intersection_meter)):
        class_iou_b = class_intersection_meter_b[i] / (class_union_meter_b[i] + 1e-10)
        class_iou_class_b.append(class_iou_b)
        class_miou_b += class_iou_b
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou

    class_miou_b = class_miou_b * 1.0 / len(class_intersection_meter_b)
    class_miou = class_miou * 1.0 / len(class_intersection_meter)

    logger.info('meanIoU---Val result: mIoU_b {:.4f}.'.format(class_miou_b))
    logger.info('meanIoU---Val result: mIoU_f {:.4f}.'.format(class_miou))

    logger.info('FBIoU---Val result: FBIoU_f {:.4f}.'.format(mIoU))

    logger.info('<<<<<<< Novel Results <<<<<<<')
    for i in range(split_gap):
        logger.info('Class_{} Result: iou_b {:.4f}.'.format(i + 1, class_iou_class_b[i]))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou_f {:.4f}.'.format(i + 1, class_iou_class[i]))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

    return mIoU, class_miou, class_miou_b, iou_class[1]


if __name__ == '__main__':
    main()