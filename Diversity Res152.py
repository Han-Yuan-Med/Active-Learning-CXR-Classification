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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

setup_seed(2024)
image_path = "Images"

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

embed_file = pd.read_csv('Foundation_embeddings\\pakistan_res152_embedding.csv')
train_csv = pd.read_csv(f"covid_train_50.csv")

feature_list = []
for i in range(len(train_csv)):
    features = embed_file.iloc[np.where(embed_file.iloc[:, 0] == train_csv.iloc[i, 0])[0][0], 1:]
    feature_list.append(np.array(features))

feature_list = np.array(feature_list)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_list)


for cluster_num in range(10, 51, 10):
    print(f"Current cluster number: {cluster_num}")
    # kmeans = KMeans(init="random", n_clusters=round(len(train_csv) * idx), random_state=2024, n_init=10)
    kmeans = KMeans(init="random", n_clusters=cluster_num, random_state=2024, n_init=10)
    kmeans.fit_predict(scaled_features)

    id_array = np.zeros(cluster_num)
    for i in range(cluster_num):
        id_array[i] = np.argsort(kmeans.transform(scaled_features)[:, i])[0]
    train_csv.iloc[np.concatenate([id_array])]. \
        to_csv(f"Sample_diversity\\train_res152_{cluster_num}.csv", index=False)
