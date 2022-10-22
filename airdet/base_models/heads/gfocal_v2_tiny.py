
from functools import partial

import torch
import numpy as np
import functools
import torch.nn as nn
import torch.nn.functional as F

from ..core.base_ops import BaseConv, DWConv
from ..core.ota_assigner import SimOTAAssigner
from ..core.weight_init import normal_init, bias_init_with_prob
from ..core.bbox_calculator import bbox_overlaps, multiclass_nms
from ..core.utils import multi_apply, unmap, reduce_mean, images_to_levels, Scale

from ..losses.gfocal_loss import GIoULoss, DistributionFocalLoss, QualityFocalLoss
from airdet.utils import postprocess_gfocal as postprocess

def xyxy2CxCywh(xyxy, size=None):
    x1 = xyxy[..., 0]
    y1 = xyxy[..., 1]
    x2 = xyxy[..., 2]
    y2 = xyxy[..., 3]

    cx = (x1+x2) /2
    cy = (y1+y2) /2

    w = x2 - x1
    h = y2 - y1
    if size is not None:
    	w = w.clamp(min=0, max=size[1])
    	h = h.clamp(min=0, max=size[0])
    return torch.stack([cx, cy, w, h], axis=-1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)

def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        b, nb, ne, _ = x.size()
        x = x.reshape(b*nb*ne, self.reg_max+1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, nb, 4)
        return x


class GFocalHead_Tiny(nn.Module):
    """Ref to Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4, # 4
                 feat_channels=256,
                 reg_max=12,
                 reg_topk=4,
                 reg_channels=64,
                 strides=[8,16,32],
                 add_mean=True,
                 norm='gn',
                 act='relu',
                 start_kernel_size=3,
                 conv_groups=1,
                 conv_type='BaseConv',
                 simOTA_cls_weight=1.0,
                 simOTA_iou_weight=3.0,
                 octbase=8,
                 **kwargs):
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.strides = strides
        self.feat_channels = feat_channels if isinstance(feat_channels, list) \
                                else [feat_channels] * len(self.strides)

        self.cls_out_channels = num_classes + 1 # add 1 for keep consistance with former models
                                                # and will be deprecated in future.
        self.stacked_convs = stacked_convs
        self.conv_groups = conv_groups
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        self.start_kernel_size = start_kernel_size
        self.decode_in_inference = True # will be set as False, when trying to convert onnx models

        self.norm = norm
        self.act = act
        self.conv_module = DWConv if conv_type=='DWConv' else BaseConv

        if add_mean:
            self.total_dim += 1

        self.assigner = SimOTAAssigner(
            center_radius=2.5,
            cls_weight=simOTA_cls_weight,
            iou_weight=simOTA_iou_weight)

        super(GFocalHead_Tiny, self).__init__()
        self.integral = Integral(self.reg_max)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.loss_cls = QualityFocalLoss(use_sigmoid=False, beta=2.0, loss_weight=1.0)
        self.loss_bbox = GIoULoss(loss_weight=2.0)

        self._init_layers()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        self.relu = nn.ReLU(inplace=True)
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else self.start_kernel_size
            cls_convs.append(
                    self.conv_module(
                        chn,
                        feat_channels,
                        kernel_size,
                        stride=1,
                        groups=self.conv_groups,
                        norm=self.norm,
                        act=self.act))
            reg_convs.append(
                    self.conv_module(
                        chn,
                        feat_channels,
                        kernel_size,
                        stride=1,
                        groups=self.conv_groups,
                        norm=self.norm,
                        act=self.act))

        conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]
        reg_conf = nn.Sequential(*conf_vector)

        return cls_convs, reg_convs, reg_conf

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_confs = nn.ModuleList()

        for i in range(len(self.strides)):
            cls_convs, reg_convs, reg_conf = self._build_not_shared_convs(
                                                self.in_channels[i],
                                                self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.reg_confs.append(reg_conf)

        self.gfl_cls = nn.ModuleList(
                            [nn.Conv2d(
                                self.feat_channels[i],
                                self.cls_out_channels,
                                3,
                                padding=1) for i in range(len(self.strides))])

        self.gfl_reg = nn.ModuleList(
                            [nn.Conv2d(
                                self.feat_channels[i],
                                4 * (self.reg_max + 1),
                                3,
                                padding=1) for i in range(len(self.strides))])

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for cls_conv in self.cls_convs:
            for m in cls_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conv in self.reg_convs:
            for m in reg_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conf in self.reg_confs:
            for m in reg_conf:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

    def forward(self, xin, labels=None, imgs=None, conf_thre=0.05, nms_thre=0.7):

        # prepare labels during training
        b, c, h, w = xin[0].shape
        if labels is not None:
            gt_bbox_list = []
            gt_cls_list = []
            for label in labels:
                gt_bbox_list.append(label.bbox)
                gt_cls_list.append((label.get_field("labels") - 1).long()) # labels starts from 1

        # prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(
                xin[i].shape[0],
                xin[i].shape[-2:],
                stride,
                dtype=torch.float32,
                device=xin[0].device)
                for i, stride in enumerate(self.strides)]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)

        # forward for bboxes and classification prediction
        cls_scores, bbox_preds = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.reg_confs,
            self.scales,
            )
        flatten_cls_scores = torch.cat(cls_scores, dim=1)
        flatten_bbox_preds = torch.cat(bbox_preds, dim=1)

        # calculating losses or bboxes decoded
        if self.training:
            loss = self.loss(
                flatten_cls_scores,
                flatten_bbox_preds,
                gt_bbox_list,
                gt_cls_list,
                mlvl_priors)
            return loss
        else:
            output = self.get_bboxes(
                flatten_cls_scores,
                flatten_bbox_preds,
                mlvl_priors)
            if self.decode_in_inference:
                output = postprocess(output, self.num_classes, conf_thre, nms_thre, imgs)
            return output

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, reg_conf, scale):
        """Forward feature of a single scale level.

        """
        cls_feat = x
        reg_feat = x

        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(gfl_reg(reg_feat)).float()
        N, C, H, W = bbox_pred.size()
        prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max+1, H, W), dim=2)
        prob_topk, _ = prob.topk(self.reg_topk, dim=2)

        if self.add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                             dim=2)
        else:
            stat = prob_topk

        quality_score = reg_conf(stat.reshape(N, 4*self.total_dim, H, W))
        cls_score = gfl_cls(cls_feat).sigmoid() * quality_score

        flatten_cls_score = cls_score.flatten(start_dim=2).transpose(1, 2)
        flatten_bbox_pred = bbox_pred.flatten(start_dim=2).transpose(1, 2)
        return flatten_cls_score, flatten_bbox_pred

    def get_single_level_center_priors(self,
                                       batch_size,
                                       featmap_size,
                                       stride,
                                       dtype,
                                       device):

        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype, device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype, device=device)) * stride

        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)

        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0], ), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)

        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             mlvl_center_priors,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        """
        device = cls_scores[0].device

        # get decoded bboxes for label assignment
        dis_preds = self.integral(bbox_preds) * mlvl_center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)
        cls_reg_targets = self.get_targets(cls_scores,
                                           decoded_bboxes,
                                           gt_bboxes,
                                           mlvl_center_priors,
                                           gt_labels_list=gt_labels)
        if cls_reg_targets is None:
            return None

        (labels_list, label_scores_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, dfl_targets_list, num_pos) = cls_reg_targets

        num_total_pos = max(
            reduce_mean(torch.tensor(num_pos).type(torch.float).to(device)).item(), 1.0)

        labels = torch.cat(labels_list, dim=0)
        label_scores = torch.cat(label_scores_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        dfl_targets = torch.cat(dfl_targets_list, dim=0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        bbox_preds = bbox_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)

        loss_qfl = self.loss_cls(
            cls_scores, (labels, label_scores), avg_factor=num_total_pos)

        pos_inds = torch.nonzero(
                (labels >= 0) & (labels < self.num_classes), as_tuple=False).squeeze(1)

        temp_scores = bbox_overlaps(decoded_bboxes[pos_inds], bbox_targets[pos_inds], is_aligned=True)
        if len(pos_inds) > 0:
            weight_targets = cls_scores.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            norm_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=1.0 * norm_factor,
                )
            loss_dfl = self.loss_dfl(
                bbox_preds[pos_inds].reshape(-1, self.reg_max+1),
                dfl_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * norm_factor,
                )

        else:
            loss_bbox = bbox_preds.sum() * 0.0
            loss_dfl = bbox_preds.sum() * 0.0

        total_loss = loss_qfl + loss_bbox + loss_dfl

        return dict(
                total_loss=total_loss,
                loss_cls=loss_qfl,
                loss_bbox=loss_bbox,
                loss_dfl=loss_dfl,
                )

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    mlvl_center_priors,
                    gt_labels_list=None,
                    unmap_outputs=True):
        """Get targets for GFL head.

        """
        num_imgs = mlvl_center_priors.shape[0]

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_dfl_targets, all_pos_num) = multi_apply(
             self.get_target_single,
             mlvl_center_priors,
             cls_scores,
             bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
        )
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        all_pos_num = sum(all_pos_num)

        return (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
            all_bbox_weights, all_dfl_targets, all_pos_num)

    def get_target_single(self,
                           center_priors,
                           cls_scores,
                           bbox_preds,
                           gt_bboxes,
                           gt_labels,
                           unmap_outputs=True,
                           gt_bboxes_ignore=None):
        """Compute regression, classification targets for anchors in a single
        image.

        """
        # assign gt and sample anchors

        num_valid_center = center_priors.shape[0]

        labels = center_priors.new_full((num_valid_center, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = center_priors.new_zeros(num_valid_center, dtype=torch.float)
        label_scores = center_priors.new_zeros(num_valid_center, dtype=torch.float)

        bbox_targets = torch.zeros_like(center_priors)
        bbox_weights = torch.zeros_like(center_priors)
        dfl_targets = torch.zeros_like(center_priors)

        if gt_labels.size(0) == 0:

            return (labels, label_scores, label_weights,
                   bbox_targets, bbox_weights,  dfl_targets, 0)

        assign_result = self.assigner.assign(
            cls_scores.detach(),
            center_priors,
            bbox_preds.detach(),
            gt_bboxes,
            gt_labels)

        pos_inds, neg_inds, pos_bbox_targets, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dfl_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2]/center_priors[pos_inds, None, 2], pos_bbox_targets/center_priors[pos_inds, None, 2], self.reg_max)
            )
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # map up to original set of anchors

        return (labels, label_scores, label_weights, bbox_targets, bbox_weights,
                dfl_targets, pos_inds.size(0))

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def get_bboxes(self, cls_preds, reg_preds, mlvl_center_priors, img_meta=None):

        device = cls_preds.device
        batch_size = cls_preds.shape[0]
        dis_preds = self.integral(reg_preds) * mlvl_center_priors[..., 2, None]
        bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)

        bboxes = xyxy2CxCywh(bboxes) if not getattr(self,'end2end',False) else bboxes
        obj = torch.ones_like(cls_preds[..., 0:1])
        res = torch.cat([bboxes, obj, cls_preds[..., 0:self.num_classes]], dim=-1)

        return res
