import torch
import torch.nn as nn
from ..image_classfication.darknet import DarkNet_19
import numpy as np
from .tools import *


def iou_score(bboxes_a, bboxes_b):
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    return area_i / (area_a + area_b - area_i)


def loss(pred_conf, pred_cls, pred_txtytwth, label, num_classes, obj_loss_f='mse'):
    if obj_loss_f == 'bce':
        # In yolov3, we use bce as conf loss_f
        conf_loss_function = BCELoss(reduction='mean')
        obj = 1.0
        noobj = 1.0
    elif obj_loss_f == 'mse':
        # In yolov2, we use mse as conf loss_f.
        conf_loss_function = MSELoss(reduction='mean')
        obj = 5.0
        noobj = 1.0

    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    pred_conf = torch.sigmoid(pred_conf[:, :, 0])
    pred_cls = pred_cls.permute(0, 2, 1)
    txty_pred = pred_txtytwth[:, :, :2]
    twth_pred = pred_txtytwth[:, :, 2:]

    gt_conf = label[:, :, 0].float()
    gt_obj = label[:, :, 1].float()
    gt_cls = label[:, :, 2].long()
    gt_txtytwth = label[:, :, 3:-1].float()
    gt_box_scale_weight = label[:, :, -1]
    gt_mask = (gt_box_scale_weight > 0.).float()

    # objectness loss
    pos_loss, neg_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)
    conf_loss = obj * pos_loss + noobj * neg_loss

    # class loss
    cls_loss = torch.mean(torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_mask, 1))

    # box loss
    txty_loss = torch.mean(
        torch.sum(torch.sum(txty_loss_function(txty_pred, gt_txtytwth[:, :, :2]), 2) * gt_box_scale_weight * gt_mask,
                  1))
    twth_loss = torch.mean(
        torch.sum(torch.sum(twth_loss_function(twth_pred, gt_txtytwth[:, :, 2:]), 2) * gt_box_scale_weight * gt_mask,
                  1))

    txtytwth_loss = txty_loss + twth_loss

    total_loss = conf_loss + cls_loss + txtytwth_loss

    return conf_loss, cls_loss, txtytwth_loss, total_loss

class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1, leakyReLU=False):
        super(Conv2d, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class reorg_layer(nn.Module):
    """
    把x进行重构，每个stride构成的平面重构成到不同的channel中
    """
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride

        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x


class YOLOv2(nn.Module):
    def __init__(self, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.5,
                 anchor_size=None, hr=False):
        super(YOLOv2, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.anchor_number = len(anchor_size)
        self.stride = 32
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy()).float()

        # backbone darknet-19
        self.backbone = DarkNet_19()

        # detection head
        self.convsets_1 = nn.Sequential(
            Conv2d(1024, 1024, 3, 1, leakyReLU=True),
            Conv2d(1024, 1024, 3, 1, leakyReLU=True)
        )

        self.route_layer = Conv2d(512, 64, 1, leakyReLU=True)
        self.reorg = reorg_layer(stride=2)

        self.convsets_2 = Conv2d(1280, 1024, 3, 1, leakyReLU=True)

        # prediction layer
        self.pred = nn.Conv2d(1024, self.anchor_number * (1 + 4 + self.num_classes), 1)

    def create_grid(self, input_size):
        w, h = input_size[1], input_size[0]
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)  # the size of bbox
        order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order

        keep = []  # store the final bounding boxes
        while order.size > 0:
            i = order[0]  # the index of the bbox with highest confidence
            keep.append(i)  # save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf):
        """
        bbox_pred: (HxW*anchor_n, 4), bsize = 1
        prob_pred: (HxW*anchor_n, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bbox_pred, scores, cls_inds

    def forward(self, x, target=None):
        # backbone
        _, fp_1, fp_2 = self.backbone(x)

        # head
        fp_2 = self.convsets_1(fp_2)

        # route from 16th layer in darknet
        fp_1 = self.reorg(self.route_layer(fp_1))

        # route concatenate
        fp = torch.cat([fp_1, fp_2], dim=1)
        fp = self.convsets_2(fp)
        prediction = self.pred(fp)

        B, abC, H, W = prediction.size()

        # [B, anchor_n * C, N, M] -> [B, N, M, anchor_n * C] -> [B, N*M, anchor_n*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # Divide prediction to conf_pred, txtytwth_pred and cls_pred
        # [B, H*W*anchor_n, 1]
        conf_pred = prediction[:, :, :1 * self.anchor_number].contiguous().view(B, H * W * self.anchor_number, 1)
        # [B, H*W, anchor_n, num_cls]
        cls_pred = prediction[:, :,
                   1 * self.anchor_number: (1 + self.num_classes) * self.anchor_number].contiguous().view(B,
                                                                                                          H * W * self.anchor_number,
                                                                                                          self.num_classes)
        # [B, H*W, anchor_n, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.anchor_number:].contiguous()

        # test
        if not self.trainable:
            txtytwth_pred = txtytwth_pred.view(B, H * W, self.anchor_number, 4)
            with torch.no_grad():
                # batch size = 1
                all_obj = torch.sigmoid(conf_pred)[0]  # 0 is because that these is only 1 batch.
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_obj)
                # separate box pred and class conf
                all_obj = all_obj.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                return bboxes, scores, cls_inds

        else:
            txtytwth_pred = txtytwth_pred.view(B, H * W, self.anchor_number, 4)
            # decode bbox, and remember to cancel its grad since we set iou as the label of objectness.
            with torch.no_grad():
                x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.scale_torch).view(-1, 4)

            txtytwth_pred = txtytwth_pred.view(B, H * W * self.anchor_number, 4)

            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

            # 计算iou
            iou = iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, H * W * self.anchor_number, 1)
            # print(iou.min(), iou.max())

            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            target = torch.cat([iou, target[:, :, :7]], dim=2)

            conf_loss, cls_loss, txtytwth_loss, total_loss = loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=target,
                                                                        num_classes=self.num_classes)

            return conf_loss, cls_loss, txtytwth_loss, total_loss