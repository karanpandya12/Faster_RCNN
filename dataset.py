import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        #############################################
        # TODO Initialize  Dataset
        #############################################

        image_set = h5py.File(paths[0], 'r')
        self.labels = np.load(paths[1], allow_pickle=True)
        self.bboxes = np.load(paths[2], allow_pickle=True)

        self.images = image_set['data']

        j = 0
        for i in range(self.images.shape[0]):
            num_obj = self.labels[i].shape[0]
            j += num_obj

        pass

    def __getitem__(self, index):
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        ################################
        # In this function for given index we rescale the image and the corresponding  masks, boxes
        # and we return them as output
        # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index

        transed_img, transed_bbox = self.pre_process_batch(self.images[index], self.bboxes[index])
        label = self.labels[index]
        label = torch.tensor(label, device=device, dtype=torch.uint8)

        # assert transed_img.shape == (3,800,1088)
        # assert transed_bbox.shape[0] == transed_mask.shape[0]

        return transed_img, label, transed_bbox, index

    def pre_process_batch(self, img, bbox):
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes
        ######################################
        # This function preprocess the given image, mask, box by rescaling them appropriately
        # output:
        #        img: (3,800,1088)
        #        mask: (n_box,800,1088)
        #        box: (n_box,4)

        img = img / 255.0

        img = torch.tensor(img, device=device, dtype=torch.float16)
        bbox = torch.tensor(bbox, device=device, dtype=torch.float16)

        img = F.interpolate(img.unsqueeze(dim=0), size=(800, 1066))
        img = img.squeeze(dim=0)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = transforms.functional.normalize(img, mean, std)

        img = F.pad(img, (11, 11))

        bbox[:, [0, 2]] = ((bbox[:, [0, 2]] / 400) * 1066) + 11
        bbox[:, [1, 3]] = (bbox[:, [1, 3]] / 300) * 800

        # assert img.squeeze(0).shape == (3, 800, 1088)
        # assert bbox.shape[0] == mask.squeeze(0).shape[0]

        return img, bbox

    def __len__(self):
        return len(self.images)


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def collect_fn(self, batch):
        # output:
        #  dict{images: (bz, 3, 800, 1088)
        #       labels: list:len(bz)
        #       masks: list:len(bz){(n_obj, 800,1088)}
        #       bbox: list:len(bz){(n_obj, 4)}
        #       index: list:len(bz)

        transed_img_list = []
        label_list = []
        transed_bbox_list = []
        index_list = []

        for transed_img, label, transed_bbox, index in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_bbox_list.append(transed_bbox)
            index_list.append(index)

        return (torch.stack(transed_img_list, dim=0)), label_list, transed_bbox_list, index_list

    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)