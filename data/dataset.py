import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
from config import DATA_ROOT


class PedestrainDataset(data.Dataset):

    def __init__(self, root, transform=None, dataset_name='train'):
        self.root = root
        self.transform = transform
        self.name = dataset_name
        self.train_anno_list = open(os.path.join(DATA_ROOT, 'train_annotations.txt')).read().split('\n')
        # self.train_ignore_list = open(os.path.join(DATA_ROOT, 'pedestrian_ignore_part_train')).read().split('\n')
        self.train_img_list = open(os.path.join(DATA_ROOT, 'train.txt')).read().split('\n')
        self.train_img_path = os.path.join(DATA_ROOT, 'Images')
        # self.imgs = os.listdir(self.img_path)
        index = 0
        for idx in range(len(self.train_img_list)):
            box_list = self.train_anno_list[index].split(' ')
            if len(box_list) == 1:
                self.train_img_list.remove(self.train_img_list[index])
                self.train_anno_list.remove(self.train_anno_list[index])
                index = index - 1
            index = index + 1
        self.length = len(self.train_img_list)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return self.length

    def pull_item(self, index):
        img_path = os.path.join(self.train_img_path, self.train_img_list[index])
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        # print(self.img_list[index])

        anno = self.train_anno_list[index].split(" ")
        target = []
        object_num = int((len(anno) - 1) / 5)
        # print(object_num)
        res = []
        for p in range(object_num):
            if p >= 0:
                bndbox = []
                idx = 1 + 5 * p
                bndbox.append((int(anno[idx + 1]) - 1) / width)
                bndbox.append((int(anno[idx + 2]) - 1) / height)
                bndbox.append((int(anno[idx + 1]) + int(anno[idx + 3]) - 1) / width)
                bndbox.append((int(anno[idx + 2]) + int(anno[idx + 4]) - 1) / height)
                label = int(anno[idx]) - 1
                bndbox.append(label)
                res += [bndbox]
        if self.transform is not None:
            target = np.array(res)
            # print(target.shape)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_path = os.path.join(self.train_img_path, self.train_img_list[index])
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        anno = self.train_anno_list[index].split(" ")
        object_num = int((len(anno) - 1) / 5)
        res = []
        for p in range(object_num):
            bndbox = []
            idx = 1 + 5 * p
            bndbox.append(int(anno[idx + 1]) - 1)
            bndbox.append(int(anno[idx + 2]) - 1)
            bndbox.append(int(anno[idx + 1]) + int(anno[idx + 3]) - 1)
            bndbox.append(int(anno[idx + 2]) + int(anno[idx + 4]) - 1)
            label = int(anno[idx]) - 1
            bndbox.append(label)
            res += [bndbox]
        id = index
        return id, res

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
