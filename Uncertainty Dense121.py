import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
import pandas as pd
from Classification_Functions import *
from Uncertainty_Functions import *
from Bootstrap_Functions import *
from tqdm import tqdm
from collections import defaultdict

setup_seed(2024)
image_path = "Images"

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

embed_file = pd.read_csv('Foundation_embeddings\\pakistan_dense121_embedding.csv')
train_csv = pd.read_csv(f"covid_train_50.csv")

# Calculate uncertainty
train_set_reg_tmp = Lung_reg(csv_file=train_csv, img_dir=image_path, embed_file=embed_file)
train_loader_reg_tmp = torch.utils.data.DataLoader(train_set_reg_tmp, batch_size=10, shuffle=True)
train_loader_reg_loss = torch.utils.data.DataLoader(train_set_reg_tmp, batch_size=1, shuffle=False)
reg_model = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT)
reg_model.classifier[6] = nn.Linear(4096, 1024)
reg_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, reg_model.parameters()),
                            lr=0.0001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

epoch_num = 200
loss_past = 10086
patience = 0
early_stopping = 0

for epoch in range(epoch_num):
    train_reg(train_loader=train_loader_reg_tmp, device=device, reg_model=reg_model,
              criterion=criterion, optimizer=optimizer)
    print(f"Finish training at epoch {epoch + 1}")

    prob_list = []
    label_list = []
    with torch.no_grad():
        for data in tqdm(train_loader_reg_loss):
            images, labels = data[0].float().to(device), data[1].cpu().numpy()
            outputs = reg_model(images)
            label_list = np.concatenate((label_list, labels), axis=None)
            prob_list = np.concatenate((prob_list, outputs.squeeze().detach().cpu().numpy()),
                                       axis=None)
    loss_tmp = mean_square_error(label_list, prob_list)

    print(f"MSE on current set is {loss_tmp} \n"
          f"Previous optimal MSE is {loss_past}")

    if loss_tmp < loss_past:
        loss_past = loss_tmp
        PATH = f"Regressor_uncertainty\\VGG11-dense121.pt"
        torch.save(reg_model, PATH)
        print("Update regression model\n")
        patience = 0
        early_stopping = 0
    else:
        patience += 1
        early_stopping += 1
        print(f"Not update regression model\n"
              f"Add patience index to {int(patience)} \n"
              f"Add early stop index to {int(early_stopping)} \n")

    if patience == 10:
        # break
        print("No improvement in the last 10 epochs; \n Decrease learning rate")
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1} SGD lr {before_lr} -> {after_lr} \n")
        patience = 0

    if early_stopping == 20:
        print("No improvement in the last 20 epochs; \n Stop training")
        break


mc_times = 100
output_path = "Monte_carlo\\dense121"
summary_path = "Monte_carlo\\summary\\dense121.csv"
reg_model = torch.load(f"Regressor_uncertainty\\VGG11-dense121.pt")
dropout_calculate(reg_model=reg_model, train_csv=train_csv, train_loader_reg_loss=train_loader_reg_loss,
                  mc_times=mc_times, output_path=output_path, device=device)
print("Finish drop out calculation")
variance_calculate(output_path=output_path, summary_path=summary_path)
print("Finish variance calculation")

del train_loader_reg_tmp
del reg_model
del criterion
del optimizer
del scheduler

uncertainty_csv = pd.read_csv("Monte_carlo\\summary\\dense121.csv", header=None)
for sample_num in range(10, 51, 10):
    print(f"Current sample number: {sample_num}")
    id_array = np.argsort(-uncertainty_csv.iloc[:, 1])[:sample_num]
    train_csv.iloc[np.concatenate([id_array])]. \
        to_csv(f"Sample_uncertainty\\train_dense121_{sample_num}.csv", index=False)
