import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *
import tqdm
import os
import time
from BoxHead import *
from pretrained_models import *
from torchvision.models.detection.image_list import ImageList
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import torchvision

if __name__=="__main__":

    # Put the path were you save the given pretrained model
    pretrained_path='/content/drive/My Drive/CIS 680/data/checkpoint680.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone, rpn = pretrained_models_680(pretrained_path)

    # we will need the ImageList from torchvision

    imgs_path = "/content/drive/My Drive/CIS 680/data/hw3_mycocodata_img_comp_zlib.h5"
    labels_path = "/content/drive/My Drive/CIS 680/data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "/content/drive/My Drive/CIS 680/data/hw3_mycocodata_bboxes_comp_zlib.npy"

    paths = [imgs_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # Standard Dataloaders Initialization
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 1
    print("batch size:", batch_size)

    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    train_loader = train_build_loader.loader()
    print(len(train_loader))

    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    # Here we keep the top 20, but during training you should keep around 200 boxes from the 1000 proposals
    keep_topK = 200

    path = os.path.join('.', 'drive', 'My Drive', 'CIS 680', 'Faster_RCNN', 'Trial 1', 'saved_epochs',
                        'frcnn_epoch_39')  # Set this to where the checkpoint is saved

    # anchors_param = dict(ratio=torch.tensor([0.7852]), scale=torch.tensor([360]), grid_size=(50, 68), stride=16)

    box_head = BoxHead(Classes=3, P=7)
    box_head.to(device)

    # avg_precision_values = torch.zeros(num_epochs)
    # learning_rate = (0.01 / 16) * batch_size
    learning_rate = 0.0007

    effective_batch = batch_size*16
    l = 20
    conf_thresh = 0.5
    keep_num_preNMS = 50
    keep_num_post_NMS = 3
    NMS = True

    checkpoint = torch.load(path)
    box_head.load_state_dict(checkpoint['model_state_dict'])
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    rpn.load_state_dict(checkpoint['rpn_state_dict'])

    box_head.eval()
    color_list = ['red','blue', 'green']

    match_values = {'0': [], '1': [], '2': []}
    score_values = {'0': [], '1': [], '2': []}
    total_trues = {'0': 0, '1': 0, '2': 0}

    with torch.no_grad():

        # box_head.gt_dict.clear()
        # gt_path = os.path.join('.', 'drive', 'My Drive', 'CIS 680', 'Faster_RCNN', 'saved_dicts3',
        #                        'gt_val_dict.h5')
        # box_head.gt_dict = torch.load(gt_path)

        for j, data in enumerate(test_loader, 0):

            images, label, bbox, index = data

            # Take the features from the backbone
            backout = backbone(images.float())

            # The RPN implementation takes as first argument the following image list
            im_lis = ImageList(images.float(), [(800, 1088)] * images.shape[0])
            # Then we pass the image list and the backbone output through the rpn
            rpnout = rpn(im_lis, backout)

            # The final output is
            # A list of proposal tensors: list:len(bz){(keep_topK,4)}
            proposals = [proposal[0:keep_topK, :] for proposal in rpnout[0]]
            # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}

            fpn_feat_list = list(backout.values())

            del backout, rpnout, im_lis
            torch.cuda.empty_cache()

            feature_vectors = box_head.MultiScaleRoiAlign(fpn_feat_list, proposals, index)

            del fpn_feat_list
            torch.cuda.empty_cache()

            class_pred, box_pred = box_head.forward(feature_vectors)

            boxes_to_plot, scores, labels_to_plot = box_head.postprocess_detections(class_pred, box_pred, proposals, conf_thresh, keep_num_preNMS, keep_num_post_NMS, NMS)

            match_list, scores_list, trues_list = box_head.evaluation(scores, labels_to_plot, boxes_to_plot, label[0], bbox[0])

            for k in range(3):
                match_values[str(k)].extend(match_list[str(k)])
                score_values[str(k)].extend(scores_list[str(k)])
                total_trues[str(k)] += trues_list[str(k)]

            # print(match_values)
            # print(score_values)
            # print(total_trues)

            # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
            images = transforms.functional.normalize(images[0],
                                                    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                    [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
            images = images.cpu().float()
            plt.figure(j)
            ax1 = plt.gca()
            plt.imshow(images.permute(1, 2, 0))
            title = 'Pre NMS Top ' + str(keep_num_preNMS)
            if NMS:
                title = 'Post NMS Top ' + str(keep_num_post_NMS)

            plt.title(title)

            for elem in range(boxes_to_plot.shape[0]):
                coord = boxes_to_plot[elem, :].view(-1)

                col = color_list[labels_to_plot[elem]-1]
                rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
                                         color=col)
                ax1.add_patch(rect)

            # plt.savefig("/content/drive/My Drive/CIS 680/Faster_RCNN/pre_NMS_images/prop_" + str(j) + ".png")
            plt.savefig("/content/drive/My Drive/CIS 680/Faster_RCNN/Trial 1/post_NMS_images/prop_" + str(j) + ".png")
            plt.close('all')

            del feature_vectors
            torch.cuda.empty_cache()

            del proposals, index, label, bbox
            torch.cuda.empty_cache()

            del class_pred, box_pred, images
            # del fpn_feat_list,cate_pred_list,ins_pred_list,ins_gts_list,\
            # ins_ind_gts_list,cate_gts_list,cate_loss,mask_loss,total_loss
            torch.cuda.empty_cache()

            if j == 20:
                break

            pass

        AP = 0
        count = 0
        mAP = 0

        for k in range(3):
            if len(match_values[str(k)]) > 0:
                area = box_head.average_precision(match_values[str(k)], score_values[str(k)], total_trues[str(k)])
                print("Average precision for class ", str(k+1), area)
                AP += area
                count += 1

        if count >0:
            mAP = AP/count

        print("Mean Average Precision:", mAP)

        end_time = time.time()
