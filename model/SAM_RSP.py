import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm
import os
import os.path as osp
import csv
import numpy as np
import random
import time
import cv2
from model import BAM
from model.image_encoder import ImageEncoderViT
from model.image_encoder import Block
import math


def create_diff_mask(output_bin):
    cosine_eps = 1e-7
    bsize, ch, h, w = output_bin.size()[:]
    diff = (output_bin[:, 1, :, :].sub(output_bin[:, 0, :, :])).reshape(bsize, -1)
    diff_max = torch.max(diff, dim=1, keepdim=True)[0]
    diff_min = torch.min(diff, dim=1, keepdim=True)[0]
    diff = (diff - diff_min) / (diff_max - diff_min + cosine_eps)
    diff = diff.reshape(bsize, -1, h, w)
    return diff


class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()
        # rough segment prompt generator
        self.RSPG_weight_path = args.RSPG_weight_path
        self.Bam = BAM.OneModel(args, cls_type='Base')
        if os.path.isfile(self.RSPG_weight_path):
            checkpoint = torch.load(self.RSPG_weight_path, map_location=torch.device('cpu'))
            new_param = checkpoint['state_dict']
            try:
                self.Bam.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                self.Bam.load_state_dict(new_param)

        # vit_L
        self.SAM_pretrained_path = args.SAM_pretrained_path
        self.feature_extractor = ImageEncoderViT(img_size=1024,
                                                 patch_size=16,
                                                 depth=24,
                                                 num_heads=16,
                                                 embed_dim=1024,
                                                 use_abs_pos=True,
                                                 use_rel_pos=True,
                                                 rel_pos_zero_init=True,
                                                 global_attn_indexes=[5, 11, 17, 23],
                                                 window_size=14,
                                                 )
        state_dict = torch.load(self.SAM_pretrained_path)
        model_dict = self.feature_extractor.state_dict()
        for k, v in state_dict.items():
            if 'image_encoder' in k:
                n_k = k.replace('image_encoder.', '')
                model_dict[n_k] = v
        self.feature_extractor.load_state_dict(model_dict)

        reduce_dim = 256
        mask_add_num = 1
        classes = 2

        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + mask_add_num * 3, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.depth = args.depth
        self.blocks = nn.ModuleList()
        for i in range(self.depth):
            block = Block(
                dim=256,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
                use_rel_pos=True,
                rel_pos_zero_init=True,
                window_size=14,
                input_size=(64, 64),  # 取整除法
            )
            self.blocks.append(block)
        self.inner_cls = []
        self.beta_conv = []
        # for bin in range(((self.depth+1) // 2) - 1):
        for bin in range(self.depth - 1):
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim + 3, reduce_dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            ))
        self.inner_cls = nn.ModuleList(self.inner_cls)
        self.beta_conv = nn.ModuleList(self.beta_conv)

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [
                {'params': model.init_merge.parameters()},
                {'params': model.inner_cls.parameters()},
                {'params': model.beta_conv.parameters()},
                {'params': model.blocks.parameters()},
                {'params': model.res2.parameters()},
                {'params': model.cls.parameters()}
            ],
            lr=LR, momentum=args.momentum, weight_decay=args.weight_decay
        )
        return optimizer

    def freeze_modules(self, model):
        for param in model.Bam.parameters():
            param.requires_grad = False
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    # que_img, sup_img, sup_mask, que_mask(meta), que_mask(base), cat_idx(meta)
    def forward(self, x, s_x, s_y, y_m, y_b, cat_idx=None):
        bs, ch, h, w = x.size()[:]
        with torch.no_grad():
            # rough segment mask generator
            outputBam = self.Bam(x, s_x, s_y, y_m, y_b, cat_idx)

            # rough segment mask
            outputBam_bin = F.interpolate(outputBam, size=(64, 64), mode='bilinear', align_corners=True)
            diff_mask = create_diff_mask(outputBam_bin)

            # query feat
            x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=True)
            query_feat_sam = self.feature_extractor(x)

        # pixel-level prototype
        query_pro_list = []
        bsize, ch_sz, sp_sz, _ = query_feat_sam.size()[:]
        cosine_eps = 1e-7
        tmp_query = query_feat_sam
        tmp_query = tmp_query.reshape(bsize, ch_sz, -1)  # ch h*w

        query_feat_sam_extract = query_feat_sam.reshape(bsize, ch_sz, -1)
        query_feat_sam_extract = query_feat_sam_extract.permute(0, 2, 1)  # bs 4096 ch
        with torch.no_grad():
            query_self_correlation = torch.bmm(query_feat_sam_extract, tmp_query) / math.sqrt(ch_sz)
            min_vals, _ = torch.min(query_self_correlation, dim=2, keepdim=True)
            max_vals, _ = torch.max(query_self_correlation, dim=2, keepdim=True)
            query_self_correlation = (query_self_correlation - min_vals) / (max_vals - min_vals + cosine_eps)

            diff = (outputBam_bin[:, 1, :, :].sub(outputBam_bin[:, 0, :, :])).reshape(bsize, 1, -1)
            temp_diff = torch.clamp(diff, min=0)

            query_self_correlation = query_self_correlation * temp_diff
            query_self_correlation = query_self_correlation.to(torch.float)
            query_self_correlation = F.threshold(query_self_correlation, 0, -1e7)
            query_self_correlation = F.softmax(query_self_correlation, dim=-1)
            query_feat_sam_clone = query_feat_sam.reshape(bsize, ch_sz, -1, 1)
            for i in range(bsize):
                query_pro = query_self_correlation[i] @ query_feat_sam_clone[i]
                query_pro_list.append(query_pro)
            query_pro = torch.cat(query_pro_list, dim=0)
            query_pro = query_pro.reshape(bsize, -1, sp_sz, sp_sz)

        #
        merge_feat = torch.cat([query_feat_sam, query_pro, outputBam_bin, diff_mask], 1)

        # decoder
        out_list = []
        merge_feat = self.init_merge(merge_feat)
        merge_feat = merge_feat.permute(0, 2, 3, 1)
        for bin, blk in zip(range(self.depth), self.blocks):
            merge_feat = blk(merge_feat)
            if bin < (self.depth - 1):
                merge_feat = merge_feat.permute(0, 3, 1, 2)
                inner_out_bin = self.inner_cls[bin](merge_feat)
                inner_out_diff_mask = create_diff_mask(inner_out_bin)
                out_list.append(inner_out_bin)
                merge_feat = torch.cat([merge_feat, inner_out_bin, inner_out_diff_mask], 1)
                merge_feat = self.beta_conv[bin](merge_feat)
                merge_feat = merge_feat.permute(0, 2, 3, 1)
        merge_feat = merge_feat.permute(0, 3, 1, 2)
        query_feat = self.res2(merge_feat) + merge_feat
        out = self.cls(query_feat)

        #   Output Part
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y_m.long())
            aux_loss = torch.zeros_like(main_loss).cuda()
            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y_m.long())
            if len(out_list) != 0:
                aux_loss = aux_loss / len(out_list)

            return out.max(1)[1], aux_loss, main_loss
        else:
            return outputBam, out
