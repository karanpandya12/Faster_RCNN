import torch
import torch.nn.functional as F
from torch import nn
from utils import *
import torchvision
import numpy as np
from sklearn import metrics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        super(BoxHead, self).__init__()

        self.C=Classes
        self.P=P
        self.gt_dict = {}
        # TODO initialize BoxHead

        self.inter1 = nn.Linear(256*self.P*self.P, 1024)
        self.inter2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()

        self.classifier = nn.Linear(1024, self.C+1)

        self.regressor = nn.Linear(1024, 4*self.C)
        self.zero_int = torch.tensor([0], device=device)
        self.one_int = torch.tensor([1], device=device)

        nn.init.normal_(self.inter1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.inter2.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.regressor.weight, mean=0.0, std=0.01)

        nn.init.constant_(self.inter1.bias, 0)
        nn.init.constant_(self.inter2.bias, 0)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.regressor.bias, 0)

    def create_batch_truth(self, proposals, gt_labels, bbox, index_list):
        #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
        #  Input:
        #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
        #       gt_labels: list:len(bz) {(n_obj)}
        #       bbox: list:len(bz){(n_obj, 4)}
        #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
        #       labels: (total_proposals,1) (the class that the proposal is assigned)
        #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)

        label_list, target_list = list(zip(*map(self.create_ground_truth, proposals, gt_labels, bbox, index_list)))
        labels = torch.cat(label_list)
        regressor_target = torch.cat(target_list)

        return labels,regressor_target

    def create_ground_truth(self, img_proposals, img_gt_labels, img_bbox, index):
        key = str(index)
        if key in self.gt_dict and len(self.gt_dict[key])==3:
            img_labels, img_regressor_target = self.gt_dict[key][1:3]
            return img_labels.long(), img_regressor_target.float()

        img_bbox=img_bbox.float()
        img_gt_labels=img_gt_labels.long()
        iou_matrix = IOU(img_proposals, img_bbox)

        max_values, max_ids = torch.max(iou_matrix, dim=1)

        img_labels = torch.where(max_values>0.5, img_gt_labels[max_ids], torch.tensor([0], device=device))

        # prop_gt = torch.where(max_values>0.5, img_bbox[max_ids], torch.tensor([0,0,0,0], device=device))
        prop_gt = img_bbox[max_ids]

        prop_center = corner_to_center(img_proposals)

        prop_gt_center = corner_to_center(prop_gt)

        img_regressor_target = torch.zeros(prop_gt.shape, device=device)

        img_regressor_target[:,0] = (prop_gt_center[:,0] - prop_center[:,0])/prop_center[:,2]
        img_regressor_target[:,1] = (prop_gt_center[:,1] - prop_center[:,1])/prop_center[:,3]
        img_regressor_target[:,2] = torch.log(prop_gt_center[:,2]/prop_center[:,2])
        img_regressor_target[:,3] = torch.log(prop_gt_center[:,3]/prop_center[:,3])

        del prop_center, prop_gt_center
        torch.cuda.empty_cache()

        self.gt_dict[key] = (self.gt_dict[key], img_labels.byte(), img_regressor_target.half())

        return img_labels, img_regressor_target

    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,index_list):
        # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
        # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
        # Input:
        #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
        #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
        #      P: scalar
        # Output:
        #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)

        fpn_feat_list = list(zip(*fpn_feat_list))

        feature_vectors_list = list(zip(*MultiApply(self.img_RoiAlign, fpn_feat_list, proposals, index_list, P=self.P)))

        feature_vectors_list = [item for subtuple in feature_vectors_list for item in subtuple]

        feature_vectors = torch.stack(feature_vectors_list)
        return feature_vectors

    def img_RoiAlign(self, img_fpn_feats, proposals, index, P):

        key = str(index)
        if key in self.gt_dict:
            img_feature_vectors = self.gt_dict[key][0]
            return img_feature_vectors.float()

        prop_center = corner_to_center(proposals)
        k = torch.floor(4 + torch.log2(torch.sqrt(prop_center[:,2]*prop_center[:,3])/224))
        del prop_center
        torch.cuda.empty_cache()
        k = torch.clamp(k, min=2, max=5)-2

        k = k.int().tolist()
        proposals=list(proposals)

        img_feature_vectors_list = MultiApply(self.prop_RoiAlign, proposals, k, fpn_feats = img_fpn_feats, P=P)

        img_feature_vectors = torch.stack(img_feature_vectors_list[0])

        self.gt_dict[key] = img_feature_vectors.half()
        return img_feature_vectors

    def prop_RoiAlign(self, proposal, k, fpn_feats, P):

        scale = fpn_feats[k].shape[2]/1088

        feature_vector = torchvision.ops.roi_align(fpn_feats[k].unsqueeze(0), boxes=[(proposal*scale).view(1,-1)], output_size=(P,P))

        return  feature_vector.view(1,-1)

    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50, NMS=True):
        # This function does the post processing for the results of the Box Head for a batch of images
        # Use the proposals to distinguish the outputs from each image
        # Input:
        #       class_logits: (total_proposals,(C+1))
        #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
        #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
        #       conf_thresh: scalar
        #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
        #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
        # Output:
        #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
        #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
        #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)

        pred_score, pred_class = torch.max(class_logits, dim=1)

        pred_box=box_regression[pred_class!=0]
        pred_score = pred_score[pred_class!=0]
        total_proposals = torch.cat(proposals)
        total_proposals = total_proposals[pred_class != 0]
        pred_class = pred_class[pred_class!=0]

        col_index = torch.stack([4*pred_class-4, 4*pred_class-3, 4*pred_class-2, 4*pred_class-1]).T
        pred_box = pred_box[torch.arange(pred_class.shape[0]).view(-1,1),col_index]

        decoded_boxes = output_decoding(pred_box, total_proposals)

        pre_NMS_boxes = decoded_boxes[pred_score>conf_thresh]
        pre_NMS_class = pred_class[pred_score>conf_thresh]
        pre_NMS_score = pred_score[pred_score>conf_thresh]


        indices = torch.argsort(pre_NMS_score, descending=True)
        top_scores = pre_NMS_score[indices[:keep_num_preNMS]]
        top_class = pre_NMS_class[indices[:keep_num_preNMS]]
        top_boxes = pre_NMS_boxes[indices[:keep_num_preNMS]]

        top_boxes[:,[0,2]]=torch.clamp(top_boxes[:,[0,2]], min=0, max=1087)
        top_boxes[:,[1,3]]=torch.clamp(top_boxes[:,[1,3]], min=0, max=799)

        if NMS==False:
            return top_boxes, top_scores, top_class

        if len(top_scores[top_class==1])>0:
           decay_scores_1 = self.NMS(top_scores[top_class==1],top_boxes[top_class==1])
        else:
            decay_scores_1 = torch.tensor([], device=device)

        if len(top_scores[top_class==2])>0:
            decay_scores_2 = self.NMS(top_scores[top_class==2],top_boxes[top_class==2])
        else:
            decay_scores_2 = torch.tensor([], device=device)

        if len(top_scores[top_class==3])>0:
            decay_scores_3 = self.NMS(top_scores[top_class==3],top_boxes[top_class==3])
        else:
            decay_scores_3 = torch.tensor([], device=device)

        post_NMS_boxes = torch.cat((top_boxes[top_class==1], top_boxes[top_class==2], top_boxes[top_class==3]))
        post_NMS_scores = torch.cat((decay_scores_1,decay_scores_2,decay_scores_3))
        post_NMS_class = torch.cat((top_class[top_class==1], top_class[top_class==2], top_class[top_class==3]))

        indices = torch.argsort(post_NMS_scores, descending=True)
        post_NMS_scores = post_NMS_scores[indices[:keep_num_postNMS]]
        post_NMS_boxes = post_NMS_boxes[indices[:keep_num_postNMS]]
        post_NMS_class = post_NMS_class[indices[:keep_num_postNMS]]

        return post_NMS_boxes, post_NMS_scores, post_NMS_class


    def NMS(self,clas,prebox,method='gauss',gauss_sigma=0.5):
        ##################################
        # TODO perform NMS
        ##################################
        # Input:
        #       clas: (top_k_boxes) (scores of the top k boxes)
        #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
        # Output:
        #       nms_clas: (Post_NMS_boxes)
        #       nms_prebox: (Post_NMS_boxes,4)

        ious = IOU(prebox,prebox).triu(diagonal=1)

        ious_cmax, ids = torch.max(ious, dim=0)
        ious_cmax = ious_cmax.expand(ious_cmax.shape[0], ious_cmax.shape[0]).T

        if method == 'gauss':
            decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)

        decay, ids = torch.min(decay, dim=0)

        nms_clas = clas*decay

        return nms_clas


    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):
        # Compute the total loss of the classifier and the regressor
        # Input:
        #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
        #      box_preds: (total_proposals,4*C)      (as outputed from forward)
        #      labels: (total_proposals,1)
        #      regression_targets: (total_proposals,4)
        #      l: scalar (weighting of the two losses)
        #      effective_batch: scalar
        # Outpus:
        #      loss: scalar
        #      loss_class: scalar
        #      loss_regr: scalar

        fg_check = (labels != 0).squeeze()
        bg_check = (labels == 0).squeeze()

        fg_labels = labels[fg_check]
        bg_labels = labels[bg_check]

        fg_class_pred = class_logits[fg_check]
        bg_class_pred = class_logits[bg_check]

        fg_box_pred = box_preds[fg_check]
        fg_box_target = regression_targets[fg_check]

        fg_num = min(int(effective_batch *0.75), fg_labels.shape[0])
        bg_num = effective_batch - fg_num

        fg_sample_ids = np.random.choice(fg_labels.shape[0], fg_num, replace=False)
        bg_sample_ids = np.random.choice(bg_labels.shape[0], bg_num, replace=False)

        pred_class = torch.cat((fg_class_pred[fg_sample_ids], bg_class_pred[bg_sample_ids]))
        gt_class = torch.cat((fg_labels[fg_sample_ids], bg_labels[bg_sample_ids])).long()

        criterion_ce = nn.CrossEntropyLoss()
        loss_c = criterion_ce(pred_class, gt_class.squeeze())

        gt_box = fg_box_target[fg_sample_ids]

        # col_index = np.linspace(start=(fg_labels[fg_sample_ids] - 1) * 4, stop=fg_labels[fg_sample_ids] * 4 - 1,num=4).T

        col_index = torch.stack([4*fg_labels[fg_sample_ids]-4, 4*fg_labels[fg_sample_ids]-3, 4*fg_labels[fg_sample_ids]-2, 4*fg_labels[fg_sample_ids]-1]).T
        pred_box = fg_box_pred[fg_sample_ids.reshape(-1, 1), col_index]

        criterion_l1 = nn.SmoothL1Loss(reduction='sum')
        loss_r = criterion_l1(pred_box, gt_box)/effective_batch

        loss = loss_c + l * loss_r

        return loss, loss_c, loss_r


    def forward(self, feature_vectors):
        # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
        # Input:
        #        feature_vectors: (total_proposals, 256*P*P)
        # Outputs:
        #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
        #                                               CrossEntropyLoss you should not pass the output through softmax here)
        #        box_pred:     (total_proposals,4*C)

        x = self.relu(self.inter2(self.relu(self.inter1(feature_vectors))))

        class_logits = self.classifier(x)

        box_pred = self.regressor(x)

        return class_logits, box_pred

    def evaluation(self, scores_img, pred_label_img, pred_box_img, gt_label_img, gt_box_img):
        # Input:
        # scores_img, tensor, (keep_instance,)
        # pred_label_img, tensor, each (keep_instance,)
        # pred_box_img, tensor, each (keep_instance, 4)
        # gt_label_img, tensor, each (n_obj,)

        # 0 - pedestrian
        # 1 - traffic light
        # 2 - car
        match = {'0': [], '1': [], '2': []}
        scores = {'0': [], '1': [], '2': []}
        trues = {'0': 0, '1': 0, '2': 0}

        # print(gt_label_img)
        # print(pred_label_img.shape)
        # print(pred_box_img.shape)

        for i in range(3):
            # print(i)
            pred_label_class = pred_label_img[pred_label_img == i+1]
            gt_label_class = gt_label_img[gt_label_img == i + 1]
            # print("Number of objects:")
            # print(gt_label_class.shape)

            if torch.numel(pred_label_class) > 0:
                # print('Prediction')
                pred_box_class = pred_box_img[pred_label_img == i+1]

                if torch.numel(gt_label_class) > 0:
                    # print("object")
                    gt_box_class = gt_box_img[gt_label_img == i + 1]

                    ious = IOU(pred_box_class, gt_box_class)
                    # print(ious)

                    matches_check = ious > 0.5
                    matches = torch.sum(matches_check, dim=1)
                    matches = torch.where(matches >= 1, self.one_int, self.zero_int)
                    matches = matches.tolist()

                    num_obj_class = gt_label_class.shape[0]

                elif torch.numel(gt_label_class) == 0:
                    # print("no object")
                    num_obj_class = 0
                    matches = [0] * pred_label_class.shape[0]

                match[str(i)].extend(matches)
                trues[str(i)] += num_obj_class
                scores[str(i)].extend(scores_img[pred_label_img == i+1])

            elif torch.numel(pred_label_class) == 0:
                # print("no predcition")
                if torch.numel(gt_label_class) > 0:
                    num_obj_class = gt_label_class.shape[0]

                elif torch.numel(gt_label_class) == 0:
                    num_obj_class = 0

                trues[str(i)] += num_obj_class

        return match, scores, trues

    def average_precision(self, match_values, score_values, total_trues):

        match_values = np.array(match_values)
        score_values = np.array(score_values)

        max_score = max(score_values).cpu()
        ln = np.linspace(0.5, max_score, 100)
        precision_values = np.zeros((100))
        recall_values = np.zeros((100))

        for i, thresh in enumerate(ln):
            matches = match_values[score_values > thresh]
            true_positive = sum(matches)
            total_positive = matches.shape[0]

            precision = 1
            if total_positive > 0:
                precision = true_positive / total_positive

            recall = 1
            if total_trues > 0:
                recall = true_positive / total_trues

            precision_values[i] = precision
            recall_values[i] = recall

        pass

        sorted_indices = np.argsort(recall_values)
        sorted_recall = recall_values[sorted_indices]
        sorted_precision = precision_values[sorted_indices]

        area = metrics.auc(sorted_recall, sorted_precision)

        return area


