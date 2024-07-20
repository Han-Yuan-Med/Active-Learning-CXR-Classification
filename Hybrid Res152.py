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
from cxr_foundation import embeddings_data

setup_seed(2024)
image_path = "Images"

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

embed_file = pd.read_csv('Foundation_embeddings\\pakistan_res152_embedding.csv')
uncertainty_file = pd.read_csv('Monte_carlo\\summary\\res152.csv', header=None)
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
        # id_tmp retrieves most uncertain sample id in cluster i
        id_tmp = np.argsort(-np.array(uncertainty_file.iloc[np.where(kmeans.labels_ == i)[0], 1]))[0]
        # retrieve if of sample id_tmp in the whole sample
        id_array[i] = np.where(kmeans.labels_ == i)[0][id_tmp]
    train_csv.iloc[np.concatenate([id_array])]. \
        to_csv(f"Sample_hybrid\\train_res152_{cluster_num}.csv", index=False)
