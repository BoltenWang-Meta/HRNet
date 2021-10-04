import torch
from    torch import nn
from    torchvision import transforms
from    torch.utils.data import DataLoader, Dataset
from    scipy.stats import multivariate_normal

import json, os, csv
import cv2 as cv
import numpy as np

def normalize_ImgandLabel(img, label, target):
    img = cv.imread(img)
    ori_hit, ori_wid, _ = img.shape
    if ori_wid < ori_hit:
        scale = target / ori_hit
        scaled_wid = int(ori_wid * scale)
        label[1::2] = scale * label[1::2]
        label[0::2] = scale * label[0::2]
        img = cv.resize(img, (scaled_wid, target))
        delta_wid = target - scaled_wid
        left_wid = int(delta_wid / 2)
        right_wid = delta_wid - left_wid
        img = cv.copyMakeBorder(img, 0, 0, left_wid, right_wid, cv.BORDER_CONSTANT, value=[255, 255, 255])
        label[0::2] = label[0::2] + np.ones_like(label[0::2]) * right_wid // 2

    else:
        scale = target / ori_wid
        scaled_hit = int(ori_hit * scale)
        label[1::2] = scale * label[1::2]
        label[0::2] = scale * label[0::2]
        img = cv.resize(img, (target, scaled_hit))
        delta_hit = target - scaled_hit
        up_hit = int(delta_hit / 2)
        down_hit = delta_hit - up_hit
        img = cv.copyMakeBorder(img, up_hit, down_hit, 0, 0, cv.BORDER_CONSTANT, value=[255, 255, 255])
        label[1::2] = label[1::2] + np.ones_like(label[1::2]) * up_hit // 2

    temp_hit, temp_wid, _ = img.shape
    return img, label



def guassian_kernel(center_x, center_y, sgm=1, size_w=128, size_h=128):
    x, y = np.mgrid[0: size_w, 0: size_h]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([center_x, center_y])
    sigma = np.array([sgm, sgm])
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)
    delta = np.max(z) + 0.0001
    z = z / delta
    return z
def generate_group(center_x, center_y, people, size_w=128, size_h=128):
    original = np.zeros((size_w, size_h))
    original[center_y][center_x] = people
    return original

class CoCo(Dataset):
    def __init__(self, mode, img_size=512, map_size=128):
        super(CoCo, self).__init__()
        self.size, self.map_size = img_size, map_size
        imgs, labels = self.read_csv()
        scale = int(0.9 * len(imgs))
        if mode == 'train':
            self.x, self.y = imgs[: scale], labels[: scale]
        else:
            self.x, self.y = imgs[scale: ], labels[scale: ]


    def read_csv(self):
        imgs, labels = [], []
        with open('coco.csv', mode='r') as f:
            lines = csv.reader(f)
            for num, line in enumerate(lines):
                # if num >= 5:
                #     break
                imgs.append(line[0])
                num_person = len(line[1:]) // 34
                temp_label = []
                for person in range(num_person):
                    temp_label.append(np.array(line[34 * person + 3: 34 * person + 15], dtype=int))

                temp_label = np.array(temp_label).flatten()

                labels.append(temp_label)
        return imgs, labels

    def imgshow(self, img, label):
        img = cv.imread(img)
        for p in range(0, len(label), 2):
            cv.circle(img, (label[p], label[p+1]), 4, (0, 0, 255), -1)
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        img, label = self.x[item], self.y[item]
        Img, _ = normalize_ImgandLabel(img, label, self.size)
        img_map, label = normalize_ImgandLabel(img, label, self.map_size)
        num_person = len(label) // 12
        gt = torch.zeros((12, self.map_size, self.map_size))
        for kp in range(0, 12, 2):
            temp_person = torch.zeros_like(gt[0])
            temp_tag = torch.zeros_like(gt[0])
            for p in range(num_person):
                target = guassian_kernel(label[kp + p * 6], label[kp + p * 6 + 1])
                target = torch.tensor(target).unsqueeze(0)
                temp_person = temp_person + target

                if label[kp + p * 6] >= 128:
                    print(label[kp + p * 6], cv.imread(img).shape)
                    label[kp + p * 6] = 127
                if label[kp + p * 6 + 1] >= 128:
                    print(label[kp + p * 6 + 1], cv.imread(img).shape)
                    label[kp + p * 6 + 1] = 127
                target = generate_group(label[kp + p * 6], label[kp + p * 6 + 1], (p+1)/num_person)
                target = torch.tensor(target).unsqueeze(0)
                temp_tag = temp_tag + target

            gt[kp] = temp_person
            gt[kp // 2 + 6] = temp_tag

        tf = transforms.ToTensor()
        Img = tf(Img)

        return Img, gt


def test():
    test_db = CoCo('test')
    test_loader = DataLoader(test_db, batch_size=16, shuffle=False, num_workers=2)
    for x, y in test_loader:
        print(x.shape, y.shape)




if __name__ == '__main__':
    test()