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

    batch_size = 2
    print("batch size:", batch_size)

    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    train_loader = train_build_loader.loader()

    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    # Here we keep the top 20, but during training you should keep around 200 boxes from the 1000 proposals
    keep_topK = 200

    path = os.path.join('.', 'drive', 'My Drive', 'CIS 680', 'Faster_RCNN', 'Trial 1', 'saved_epochs',
                        'frcnn_epoch_')  # Set this to where the checkpoint is saved
    resume = False  # set this True if you want to resume training from a checkpoint

    box_head = BoxHead(Classes=3, P=7)
    box_head.to(device)

    num_epochs = 40  ## intialize this, atleast 36 epoch required for training
    training_total_loss = torch.zeros(num_epochs)
    training_class_loss = torch.zeros(num_epochs)
    training_regr_loss = torch.zeros(num_epochs)

    # avg_precision_values = torch.zeros(num_epochs)
    # learning_rate = (0.01 / 16) * batch_size
    learning_rate = 0.0007
    start_epoch = 0

    # Intialize optimizer
    optimizer = torch.optim.Adam(box_head.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[11,18,25], gamma=0.1)

    effective_batch = batch_size * 10
    l = 15

    # Trial 1
    # 0.0007,[11,18,25](0.1),10,15

    # Trial 2
    # 0.0007,[8, 12, 16, 20, 24, 28](0.3),10,15

    if resume == True:
        checkpoint = torch.load(path)
        box_head.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        training_total_loss = checkpoint['training_total_loss']
        training_class_loss = checkpoint['training_class_loss']
        training_regr_loss = checkpoint['training_regr_loss']
        scheduler.load_state_dict(checkpoint['scheduler'])
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        rpn.load_state_dict(checkpoint['rpn_state_dict'])

    for epoch in tqdm.tqdm(range(start_epoch, num_epochs)):
        start_time = time.time()
        print("Epoch %d/%d" % (epoch + 1, num_epochs))

        box_head.train()

        # if epoch > 0:
        #     box_head.gt_dict.clear()
        #     gt_path = os.path.join('.', 'drive', 'My Drive', 'CIS 680', 'Faster_RCNN', 'saved_dicts3',
        #                            'gt_dict.h5')
        #     box_head.gt_dict = torch.load(gt_path)

        for i, data in enumerate(train_loader, 0):
            images, label, bbox, index = data

            with torch.no_grad():
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

            del images, backout, rpnout, im_lis
            torch.cuda.empty_cache()

            feature_vectors = box_head.MultiScaleRoiAlign(fpn_feat_list, proposals, index)

            del fpn_feat_list
            torch.cuda.empty_cache()

            class_pred, box_pred = box_head.forward(feature_vectors)

            del feature_vectors
            torch.cuda.empty_cache()

            class_gt, box_gt = box_head.create_batch_truth(proposals, label, bbox, index)

            del proposals, index, label, bbox
            torch.cuda.empty_cache()

            total_loss, class_loss, regr_loss = box_head.compute_loss(class_pred, box_pred, class_gt, box_gt, l, effective_batch)

            # del bbox,label,mask
            # torch.cuda.empty_cache()

            if (i + 1) % 400 == 0 or (i+1) == len(train_loader):
                print("Batch: ", i + 1, "\tClass_loss: ", class_loss, "\tRegression_loss: ", regr_loss, "\tTotal_loss: ",
                      total_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            training_total_loss[epoch] += total_loss.item()
            training_class_loss[epoch] += class_loss.item()
            training_regr_loss[epoch] += regr_loss.item()

            del class_pred, box_pred, class_gt, box_gt, class_loss, regr_loss, total_loss

            # del fpn_feat_list,cate_pred_list,ins_pred_list,ins_gts_list,\
            # ins_ind_gts_list,cate_gts_list,cate_loss,mask_loss,total_loss
            torch.cuda.empty_cache()

            pass

        # if epoch == 0:
        #     gt_path = os.path.join('.', 'drive', 'My Drive', 'CIS 680', 'Faster_RCNN', 'saved_dicts3',
        #                            'gt_dict.h5')
        #     torch.save(box_head.gt_dict, gt_path)
        #     box_head.gt_dict.clear()

        path = os.path.join('.', 'drive', 'My Drive', 'CIS 680', 'Faster_RCNN', 'Trial 1', 'saved_epochs',
                            'frcnn_epoch_' + str(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': box_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_total_loss': training_total_loss,
            'training_class_loss': training_class_loss,
            'training_regr_loss': training_regr_loss,
            'scheduler': scheduler.state_dict(),
            'backbone_state_dict': backbone.state_dict(),
            'rpn_state_dict': rpn.state_dict()
        }, path)

        end_time = time.time()
        print("Time taken for epoch " + str(epoch + 1), end_time - start_time)

    # Dividing by number of batches
    training_total_loss = training_total_loss / (i + 1)
    training_class_loss = training_class_loss / (i + 1)
    training_regr_loss = training_regr_loss / (i + 1)
    print("Total loss: ", training_total_loss)
    print("Class loss: ", training_class_loss)
    print("Regression loss: ", training_regr_loss)

    plt.figure(1)
    plt.title("Total Loss Curve")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Total Loss per Batch")
    plt.plot(np.arange(num_epochs) + 1, training_total_loss)
    plt.savefig("/content/drive/My Drive/CIS 680/Faster_RCNN/Trial 1/total_loss.png")

    plt.figure(2)
    plt.title("Class Loss Curve")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Class Loss per Batch")
    plt.plot(np.arange(num_epochs) + 1, training_class_loss)
    plt.savefig("/content/drive/My Drive/CIS 680/Faster_RCNN/Trial 1/class_loss.png")

    plt.figure(3)
    plt.title("Regression Loss Curve")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Regression Loss per Batch")
    plt.plot(np.arange(num_epochs) + 1, training_regr_loss)
    plt.savefig("/content/drive/My Drive/CIS 680/Faster_RCNN/Trial 1/regr_loss.png")

    plt.show()
