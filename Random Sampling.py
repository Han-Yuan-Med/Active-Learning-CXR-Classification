import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
import pandas as pd
from Classification_Functions import *
from Bootstrap_Functions import *
from tqdm import tqdm

setup_seed(2024)
train_csv = pd.read_csv(f"covid_train_50.csv")
create_sampled_instance(train_csv=train_csv, random_seed=2024, dataset_path="Sample_random",
                        sample_times=100, min_number=10, max_number=51, step=10)
