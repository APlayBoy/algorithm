import numpy as np
import torch
import torch.nn as nn


ignore_thresh = 0.6

class BCELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, mask):
        pos_id = (mask == 1.0).float()
        neg_id = (mask == 0.0).float()
        pos_loss = -pos_id * (targets * torch.log(inputs + 1e-14) + (1 - targets) * torch.log(1.0 - inputs + 1e-14))
        neg_loss = -neg_id * torch.log(1.0 - inputs + 1e-14)
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss


class MSELoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, mask):
        # We ignore those whose tarhets == -1.0.
        pos_id = (mask == 1.0).float()
        neg_id = (mask == 0.0).float()
        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * (inputs) ** 2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss


class BCE_focal_loss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(BCE_focal_loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        pos_id = (targets == 1.0).float()
        neg_id = (1 - pos_id).float()
        pos_loss = -pos_id * (1.0 - inputs) ** self.gamma * torch.log(inputs + 1e-14)
        neg_loss = -neg_id * (inputs) ** self.gamma * torch.log(1.0 - inputs + 1e-14)

        if self.reduction == 'mean':
            return torch.mean(torch.sum(pos_loss + neg_loss, 1))
        else:
            return pos_loss + neg_loss


def generate_anchor(input_size, stride, anchor_scale, anchor_aspect):
    assert len(anchor_scale) == len(anchor_aspect)
    h, w = input_size
    hs, ws = h // stride, w // stride
    S_fmap = hs * ws
    total_anchor_size = []
    for ab_scale, aspect_ratio in zip(anchor_scale, anchor_aspect):
        for a in aspect_ratio:
            S_ab = S_fmap * ab_scale
            ab_w = np.floor(np.sqrt(S_ab))
            ab_h = ab_w * a
            total_anchor_size.append([ab_w, ab_h])
    return total_anchor_size


def compute_iou(anchor_boxes, gt_box):
    # compute the iou between anchor box and gt box
    # First, change [c_x_s, c_y_s, anchor_w, anchor_h] ->  [xmin, ymin, xmax, ymax]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]

    # gt_box :
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily.
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2  # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2  # ymin
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2  # xmax
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2  # ymin
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U

    return IoU


def set_anchors(anchor_size):
    anchor_number = len(anchor_size)
    anchor_boxes = np.zeros([anchor_number, 4])
    for index, size in enumerate(anchor_size):
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])

    return anchor_boxes


def generate_txtytwth(gt_label, w, h, s, all_anchor_size):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1e-28 or box_h < 1e-28:
        # print('A dirty data !!!')
        return False

    c_x_s = c_x / s
    c_y_s = c_y / s
    box_ws = box_w / s
    box_hs = box_h / s

    # the grid cell location
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # generate anchor boxes
    anchor_boxes = set_anchors(all_anchor_size)
    gt_box = np.array([[0, 0, box_ws, box_hs]])
    # compute the IoU
    iou = compute_iou(anchor_boxes, gt_box)
    # We consider those anchor boxes whose IoU is more than ignore thresh,
    iou_mask = (iou > ignore_thresh)

    result = []
    if iou_mask.sum() == 0:
        # We assign the anchor box with highest IoU score.
        index = np.argmax(iou)
        p_w, p_h = all_anchor_size[index]
        tx = c_x_s - grid_x
        ty = c_y_s - grid_y
        tw = np.log(box_ws / p_w)
        th = np.log(box_hs / p_h)
        weight = 2.0 - (box_w / w) * (box_h / h)

        result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])

        return result

    else:
        best_index = np.argmax(iou)
        for index, iou_m in enumerate(iou_mask):
            if iou_m:
                if index == best_index:
                    p_w, p_h = all_anchor_size[index]
                    tx = c_x_s - grid_x
                    ty = c_y_s - grid_y
                    tw = np.log(box_ws / p_w)
                    th = np.log(box_hs / p_h)
                    weight = 2.0 - (box_w / w) * (box_h / h)

                    result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])
                else:
                    result.append([index, grid_x, grid_y, 0., 0., 0., 0., -1.0, 0., 0., 0., 0.])

        return result


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



if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10],
                             [0.0, 0.0, 4, 4],
                             [0.0, 0.0, 8, 8],
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)