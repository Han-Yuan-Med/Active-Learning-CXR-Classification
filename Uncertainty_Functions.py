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
from tqdm import tqdm
from collections import defaultdict


class Lung_reg(Dataset):
    def __init__(self, csv_file, img_dir, embed_file):
        self.annotations = csv_file
        self.img_dir = img_dir
        self.embed = embed_file

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = (image - image.min()) / (image.max() - image.min())
        embed = torch.from_numpy(np.array(self.embed.iloc[
                                          np.where(self.embed.iloc[:, 0] ==
                                                   self.annotations.iloc[index, 0])[0][0], 1:]).astype(np.float64))

        image_fused = np.zeros((3, 224, 224))
        image_fused[0] = image
        image_fused[1] = image
        image_fused[2] = image
        image_fused = torch.from_numpy(image_fused)

        return image_fused, embed


def train_reg(reg_model, train_loader, criterion, optimizer, device):
    reg_model.train()
    # deactivate dropout for convergence stability
    deactivate_dropout(reg_model)
    for batch_idx, (sample, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        sample = sample.to(device=device).float()
        targets = targets.to(device=device).float()

        # forward
        scores = reg_model(sample)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


def mean_square_error(t, p):
    t = np.float_(t)
    p = np.float_(p)
    # binary cross-entropy loss
    return ((t - p)**2).mean()


def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()


def deactivate_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.eval()


def dropout_calculate(reg_model, train_csv, train_loader_reg_loss, mc_times, output_path, device):
    reg_model.eval()
    enable_dropout(reg_model)
    for each_module in reg_model.modules():
        print(each_module.training)

    for inference_idx in range(1, mc_times + 1):
        pos_probabilities = []
        for data in tqdm(train_loader_reg_loss):
            images = data[0].float().to(device)
            output = reg_model(images)
            pos_probabilities.append(output.squeeze().cpu().detach().tolist())
        pd.DataFrame(np.column_stack((train_csv.iloc[:, 0], pos_probabilities))) \
            .to_csv(output_path + f"\\output_{inference_idx}.csv", header=False, index=False)


def variance_calculate(output_path, summary_path):
    monte_carlo_files = os.listdir(output_path)
    filename_probabilities = defaultdict(list)
    for idx, monte_carlo_file in enumerate(monte_carlo_files):
        csv_tmp = pd.read_csv(output_path + f"\\{monte_carlo_file}", header=None)
        for i in range(len(csv_tmp)):
            filename_probabilities[csv_tmp.iloc[i, 0]].append(csv_tmp.iloc[i, 1:].apply(lambda x: float(x)))

    variance_probabilities = []
    filenames = list(filename_probabilities.keys())
    for idx, filename in enumerate(filenames):
        variance_probabilities.append([filename, np.sum(np.var(np.array(filename_probabilities[filename]), axis=0))])
    pd.DataFrame(variance_probabilities).to_csv(summary_path, header=False, index=False)
