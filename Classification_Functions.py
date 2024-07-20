import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io
import matplotlib.image as img
import numpy as np
from sklearn import metrics
import cv2
import torchxrayvision as xrv
import skimage
import random
import torchvision.transforms.functional as trans
from PIL import Image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Lung_cls(Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = csv_file
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = (image - image.min()) / (image.max() - image.min())
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        image_fused = np.zeros((3, 224, 224))
        image_fused[0] = image
        image_fused[1] = image
        image_fused[2] = image
        image_fused = torch.from_numpy(image_fused)

        return image_fused, label


class Lung_cls_embed(Dataset):
    def __init__(self, csv_file, embed_file):
        self.annotations = csv_file
        self.embed = embed_file

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        name = self.annotations.iloc[index, 0]
        embed = torch.from_numpy(np.array(self.embed.iloc[np.where(self.embed.iloc[:, 0] == name)[0][0], 1:]).astype(np.float64))
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        return embed, label


# Refer to https://github.com/google-research/medical-ai-research-foundations/blob/main/colab
# /REMEDIS_finetuning_example.ipynb
class Lung_cls_remedis(Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = csv_file
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (448, 448))
        image = (image - image.min()) / (image.max() - image.min())
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        image_fused = np.zeros((448, 448, 3))
        image_fused[:, :, 0] = image
        image_fused[:, :, 1] = image
        image_fused[:, :, 2] = image

        return image_fused, label


# Refer to https://mlmed.org/torchxrayvision/
class Lung_cls_xrv(Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = csv_file
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = skimage.io.imread(img_path)
        image = xrv.datasets.normalize(image, 255)
        # image = image.mean(2)[None, ...]
        transform = torchvision.transforms.Compose([xrv.datasets.XRayResizer(224)])
        image = transform(image)
        image = torch.from_numpy(image)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        return image, label


class Lung_cls_imagenet(Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = csv_file
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = (image - image.min()) / (image.max() - image.min())
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        image_fused = np.zeros((3, 224, 224))
        image_fused[0, :, :] = image
        image_fused[1, :, :] = image
        image_fused[2, :, :] = image
        image = trans.normalize(torch.from_numpy(image_fused), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, label


def train_cls(cls_model, train_loader, criterion, optimizer, device):
    cls_model.train()
    for batch_idx, (sample, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        sample = sample.to(device=device).float()
        targets = targets.to(device=device)

        # forward
        scores = cls_model(sample)
        loss = criterion(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


def binary_cross_entropy(t, p):
    t = np.float_(t)
    p = np.float_(p)
    # binary cross-entropy loss
    return -np.average(t * np.log(p) + (1 - t) * np.log(1 - p))


def valid_cls_prc(cls_model, val_loader, device):
    prob_list = []
    label_list = []
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].float().to(device), data[1].to(device)
            outputs = cls_model(images)
            label_list = np.concatenate((label_list, labels.cpu().numpy()), axis=None)
            prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)
    precision, recall, thresholds = metrics.precision_recall_curve(label_list, prob_list)
    return metrics.auc(recall, precision)


def create_sampled_instance(train_csv, random_seed, dataset_path, sample_times, min_number, max_number, step):

    for sample_id in range(sample_times):
        sample_selected = np.array([])
        sample_left = range(len(train_csv))

        for idx in range(min_number, max_number, step):
            np.random.seed(random_seed*sample_id)
            sample_tmp = np.random.choice(sample_left, step, replace=False)
            sample_selected = np.append(sample_selected, sample_tmp)
            sample_left = np.setdiff1d(sample_left, sample_tmp)
            assert len(sample_selected) == idx
            assert len(sample_selected) == len(np.unique(sample_selected))
            train_csv.iloc[sample_selected].to_csv(f"{dataset_path}\\train_{sample_id}_{idx}.csv", index=False)

