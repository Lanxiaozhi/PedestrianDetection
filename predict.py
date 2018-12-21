import argparse
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from model import build_ssd
from config import pedestrian

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def make_predict(model_path, data_path, res_file):
    net = build_ssd('test', 300, pedestrian['num_classes'])
    net.load_weights(model_path)
    net = net.cuda()
    images = os.listdir(data_path)
    with open(res_file, "w") as f:
        for image in images:
            img = cv2.imread(os.path.join(data_path, image), cv2.IMREAD_COLOR)

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = cv2.resize(img, (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            x = torch.from_numpy(x).permute(2, 0, 1)
            xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
            if torch.cuda.is_available():
                xx = xx.cuda()
            y = net(xx)
            from data import DATA_CLASSES as labels
            top_k = 10
            detections = y.data
            scale = torch.Tensor(rgb_img.shape[1::-1]).repeat(2)
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.01:
                    score = detections[0, i, j, 0]
                    label_name = labels[i - 1]
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                    j += 1
                    f.write("{} {} {} {} {} {}\n".format(image, score, pt[0], pt[1], pt[2], pt[3]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--result")
    args = parser.parse_args()
    make_predict(args.model, args.data, args.result)
