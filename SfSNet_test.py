# coding=utf-8
from __future__ import absolute_import, division, print_function
import glob
import os
import numpy as np
import cv2
import torch
from config import M, LANDMARK_PATH, DATA_DIR, CSV_DIR, DEST_DIR
from src.functions import create_shading_recon
from src.mask import MaskGenerator
from src.model import SfSNet
from src.utils import convert
from pathlib import Path
import pandas as pd
from tqdm import tqdm



def _test():
    # define a SfSNet
    net = SfSNet()
    # set to eval mode
    net.eval()
    # load weights
    # net.load_weights_from_pkl('SfSNet-Caffe/weights.pkl')
    net.load_state_dict(torch.load('data/SfSNet.pth'))
    # define a mask generator
    mg = MaskGenerator(LANDMARK_PATH)

    # get image list
    data_info = pd.read_csv(CSV_DIR, header=None)
    num_img = len(data_info)

    for image_idx in tqdm(range(num_img)):
        # read image
        img_path = str(os.path.join(DATA_DIR, data_info.iloc[image_idx, 0]))
        im = cv2.imread(img_path)
        # resize
        im = cv2.resize(im, (M, M))
        # normalize to (0, 1.0)
        im = np.float32(im) / 255.0
        # from (128, 128, 3) to (1, 3, 128, 128)
        im = np.transpose(im, [2, 0, 1])
        im = np.expand_dims(im, 0)

        # get the normal, albedo and light parameter
        normal, _, _ = net(torch.from_numpy(im))

        # get numpy array
        n_out = normal.detach().numpy()
        # -----------add by wang-------------
        # from [1, 3, 128, 128] to [128, 128, 3]
        n_out = np.squeeze(n_out, 0)
        n_out = np.transpose(n_out, [1, 2, 0])
        # transform
        n_out2 = n_out[:, :, (2, 1, 0)]
        # print 'n_out2 shape', n_out2.shape
        n_out2 = 2 * n_out2 - 1  # [-1 1]
        nr = np.sqrt(np.sum(n_out2 ** 2, axis=2))  # nr=sqrt(sum(n_out2.^2,3))
        nr = np.expand_dims(nr, axis=2)
        n_out2 = n_out2 / np.repeat(nr, 3, axis=2)
        n_out2 = cv2.resize(n_out2, (256, 256), interpolation=cv2.INTER_NEAREST)
        dest_path = os.path.join(DEST_DIR, data_info.iloc[image_idx, 0]).replace("crop.jpg", "predict.npy")
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(dest_path, n_out2)


if __name__ == '__main__':
    _test()
