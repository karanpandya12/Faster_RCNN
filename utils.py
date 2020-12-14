import numpy as np
import torch
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))


def IOU(boxA, boxB):
    ##################################
    # TODO compute the IOU between the boxA, boxB boxes
    ##################################
    # This function computes the IOU between two set of boxes
    # Input: boxA :(n,4); boxB: (m,4)
    # Output: iou:(n,m)

    x_top_left = torch.max(boxA[:, 0].view(-1, 1), boxB[:, 0].view(1, -1))
    y_top_left = torch.max(boxA[:, 1].view(-1, 1), boxB[:, 1].view(1, -1))
    x_bottom_right = torch.min(boxA[:, 2].view(-1, 1), boxB[:, 2].view(1, -1))
    y_bottom_right = torch.min(boxA[:, 3].view(-1, 1), boxB[:, 3].view(1, -1))

    intersection_w = torch.max(torch.tensor([0.], device=device), x_bottom_right - x_top_left)
    intersection_h = torch.max(torch.tensor([0.], device=device), y_bottom_right - y_top_left)

    intersection_area = intersection_h * intersection_w

    union_area = ((boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])).view(-1, 1) \
                 + ((boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])).view(1, -1) - intersection_area

    iou = intersection_area / (union_area + 0.0001)

    return iou


def output_decoding(flatten_out,flatten_anchors):
    # This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
    # into box coordinates where it return the upper left and lower right corner of the bbox
    # Input:
    #       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
    #       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
    # Output:
    #       box: (total_proposals,4) ([x1,y1,x2,y2] format)

    box = torch.zeros(flatten_anchors.shape, device=device)
    x_s = flatten_out[:,0]*flatten_anchors[:,2] + flatten_anchors[:,0]
    y_s = flatten_out[:,1]*flatten_anchors[:,3] + flatten_anchors[:,1]
    w_s = torch.exp(flatten_out[:,2])*flatten_anchors[:,2]
    h_s = torch.exp(flatten_out[:,3])*flatten_anchors[:,3]

    box[:,0] = x_s - w_s/2
    box[:,1] = y_s - h_s/2
    box[:,2] = x_s + w_s/2
    box[:,3] = y_s + h_s/2

    return box


def corner_to_center(corner_coords):

    center_coords = torch.zeros(corner_coords.shape, device=device)
    center_coords[:,0] = (corner_coords[:,0] + corner_coords[:,2])*0.5
    center_coords[:,1] = (corner_coords[:,1] + corner_coords[:,3])*0.5
    center_coords[:,2] = corner_coords[:,2] - corner_coords[:,0]
    center_coords[:,3] = corner_coords[:,3] - corner_coords[:,1]

    return center_coords
