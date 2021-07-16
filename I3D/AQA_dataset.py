import torch
import torch.utils.data as data_utl
from scipy.io import loadmat
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os.path
import os, sys, inspect
import GPUtil

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import cv2


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):

    frames = []

    if int(vid) < 100:
        if int(vid) < 10:
            vid = str('00'+vid)
        else:
            vid = str('0'+vid)

    for i in range(start, start + num):
        img = cv2.imread(os.path.join(image_dir, vid, str(i + 1) + '.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset_AQA_Diving(split_file, root, mode, cfg, Train_or_Test):
    dataset = []

    # split_file_train = np.asarray(loadmat(split_file_train))
    if Train_or_Test:
        split_file = np.asarray(loadmat(split_file)['consolidated_train_list'])
    else:
        split_file = np.asarray(loadmat(split_file)['consolidated_test_list'])

    index_field = cfg.Dataset.Index_Field

    data = np.delete(split_file[split_file[:, 0] == index_field], 0, 1)


    i = 0
    for video_ind in range(np.shape(data)[0]):

        num_frames = cfg.Dataset.Num_Frames
        if mode == 'flow':
            num_frames = num_frames // 2
        if num_frames < 66:
            continue
        label = np.float32(data[video_ind,-1])

        dataset.append((str(int(data[video_ind,0])), label))
        i += 1

    return dataset


class AQA_dataset(data_utl.Dataset):

    def __init__(self, split_file, root, mode, cfg,  transforms=None, Train_or_Test=1):

        self.data = make_dataset_AQA_Diving(split_file, root, mode, cfg=cfg, Train_or_Test=Train_or_Test)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.n_frames = cfg.Dataset.Num_Frames
        self.flag_direct_pose_heatmap_numpy = 1
        self.cfg = cfg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label = self.data[index]
        pose_heatmaps = None
        features_figure_skating = 0
        imgs = 0

        if self.mode == 'rgb':
            if self.flag_direct_pose_heatmap_numpy:
                pose_heatmaps = load_pose_heatmaps(self.root, vid, self.cfg)
            else:
                pass
            if (self.cfg.Dataset.Index_Field == 7) or (self.cfg.Dataset.Index_Field == 8):
                features_figure_skating = load_figure_skating_features(self.root, vid, self.cfg)
            else:
                imgs = load_rgb_frames(self.root, vid, 0, self.n_frames)
                imgs = self.transforms(imgs)
                imgs = video_to_tensor(imgs)
        else:
            imgs = load_flow_frames(self.root, vid, 0, self.n_frames)
            imgs = self.transforms(imgs)
            imgs = video_to_tensor(imgs)

        if (self.cfg.Dataset.Index_Field == 7) or (self.cfg.Dataset.Index_Field == 8):
            return torch.from_numpy(pose_heatmaps).permute(1,0,2,3).float(), features_figure_skating, label
        else:
            return imgs, torch.from_numpy(pose_heatmaps).permute(1,0,2,3).float(), label


    def __len__(self):
        return len(self.data)


def load_pose_heatmaps(image_dir, vid, cfg):

    pose_dir = '%s' % image_dir + '_heatmaps'
    if int(vid) < 100:
        if int(vid) < 10:
            vid = str('00'+vid)
        else:
            vid = str('0'+vid)

    if ((cfg.Dataset.Index_Field != 7) and (cfg.Dataset.Index_Field != 8)):
        pred_array = 'heatmap_comp_' + vid + '.npz'
    else:
        pred_array = vid + '.npz'
    heatmap = np.load('%s/%s'%(pose_dir,pred_array))['heat']

    return heatmap


def load_figure_skating_features(image_dir, vid, cfg):

    features_dir = '%s' % image_dir + '_features'
    if int(vid) < 100:
        if int(vid) < 10:
            vid = str('00'+vid)
        else:
            vid = str('0'+vid)

    pred_array = vid + '.npz'
    features = np.load('%s/%s'%(features_dir,pred_array))['i3dfeatures']

    return features