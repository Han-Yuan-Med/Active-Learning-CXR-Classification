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
image_path = "Images"

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
embed_file = pd.read_csv('Foundation_embeddings\\pakistan_remedis_embedding.csv')

# Initialization budget
bug = 10

for sub_bug in range(10, 41, 10):

    if sub_bug == 10:
        train_csv_pre = pd.read_csv(f"Sample_uncertainty\\train_remedis_10.csv")
        cls_model_pre = torch.load(f"Classifier_uncertainty\\MLP3-uncertainty-remedis-10.pt")
    else:
        train_csv_pre = pd.read_csv(f"Sample_uncertainty\\train_remedis_{bug}_{sub_bug-10}_MLP3.csv")
        cls_model_pre = torch.load(f"Classifier_uncertainty\\MLP3-uncertainty-remedis-{bug}-{sub_bug-10}.pt")

    train_csv = pd.read_csv(f"covid_train_50.csv")
    train_csv_available = pd.concat([train_csv_pre, train_csv]).drop_duplicates(keep=False)
    train_loader_available = torch.utils.data.DataLoader(
        Lung_cls_embed(csv_file=train_csv_available, embed_file=embed_file), batch_size=1, shuffle=False)
    train_set_cls_tmp = Lung_cls_embed(csv_file=train_csv_pre, embed_file=embed_file)
    train_loader_cls_loss = torch.utils.data.DataLoader(train_set_cls_tmp, batch_size=1, shuffle=False)

    cls_model_pre.to(device)
    cls_model_pre.eval()
    uncertainty_list = np.zeros(len(train_loader_available))
    kk = 0

    for data in tqdm(train_loader_available):
        images = data[0].float().to(device)
        uncertainty_list[kk] = torch.sigmoid(cls_model_pre(images))[:, 1].cpu().detach()
        kk += 1

    prob_list = []
    label_list = []
    with torch.no_grad():
        for data in tqdm(train_loader_cls_loss):
            images, labels = data[0].float().to(device), data[1].cpu().numpy()
            outputs = cls_model_pre(images)
            label_list = np.concatenate((label_list, labels), axis=None)
            prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)
    fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
    thres_val = round(thresholds[np.argmax(tpr - fpr)], 3)

    id_array = np.argsort(abs(uncertainty_list-thres_val))[:10]
    del cls_model_pre
    del train_loader_available

    train_csv_now = pd.concat([train_csv_pre, train_csv_available.iloc[np.concatenate([id_array])]]).\
        drop_duplicates().reset_index(drop=True)
    train_csv_now.to_csv(f"Sample_uncertainty\\train_remedis_{bug}_{sub_bug}_MLP3.csv", index=False)

    train_set_cls_tmp = Lung_cls_embed(csv_file=train_csv_now, embed_file=embed_file)
    train_loader_cls_tmp = torch.utils.data.DataLoader(train_set_cls_tmp, batch_size=10, shuffle=True)
    train_loader_cls_loss = torch.utils.data.DataLoader(train_set_cls_tmp, batch_size=1, shuffle=False)

    cls_model = nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Linear(256, 2),
    )
    cls_model.to(device)

    weights = [len(np.where(train_csv_now.iloc[:, 1] == 1)[0]) / len(train_csv_now),
               len(np.where(train_csv_now.iloc[:, 1] == 0)[0]) / len(train_csv_now)]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, cls_model.parameters()),
                                lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    epoch_num = 200
    loss_past = 10086
    patience = 0
    early_stopping = 0

    for epoch in range(epoch_num):
        train_cls(train_loader=train_loader_cls_tmp, device=device, cls_model=cls_model,
                  criterion=criterion, optimizer=optimizer)
        print(f"Finish training at epoch {epoch + 1}")

        cls_model.eval()
        prob_list = []
        label_list = []
        with torch.no_grad():
            for data in tqdm(train_loader_cls_loss):
                images, labels = data[0].float().to(device), data[1].cpu().numpy()
                outputs = cls_model(images)
                label_list = np.concatenate((label_list, labels), axis=None)
                prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()),
                                           axis=None)
        loss_tmp = binary_cross_entropy(label_list, prob_list)

        print(f"CE on current set is {loss_tmp} \n"
              f"Previous optimal CE is {loss_past}")

        if loss_tmp < loss_past:
            loss_past = loss_tmp
            PATH = f"Classifier_uncertainty\\MLP3-uncertainty-remedis-{bug}-{sub_bug}.pt"
            torch.save(cls_model, PATH)
            print("Update classification model\n")
            patience = 0
            early_stopping = 0
        else:
            patience += 1
            early_stopping += 1
            print(f"Not update classification model\n"
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

    del cls_model
    del criterion
    del optimizer
    del scheduler

test_csv = pd.read_csv("covid_test_50.csv")
test_set_cls = Lung_cls_embed(csv_file=test_csv, embed_file=embed_file)
test_loader_cls = torch.utils.data.DataLoader(test_set_cls, batch_size=1, shuffle=False)

results_df = []
for sub_bug in range(10, 41, 10):

    cls_model = torch.load(f"Classifier_uncertainty\\MLP3-uncertainty-remedis-{bug}-{sub_bug}.pt")
    cls_model.eval()
    prob_list = []
    label_list = []
    threshold = 0.5

    with torch.no_grad():
        for data in tqdm(test_loader_cls):
            images, labels = data[0].float().to(device), data[1].cpu().numpy()
            outputs = cls_model(images)
            label_list = np.concatenate((label_list, labels), axis=None)
            prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)
    auc_std, prc_std, acc_std, bac_std, f1s_std, sen_std, spe_std, ppv_std, npv_std = \
        bootstrap_cls_mul(prob_list=prob_list, label_list=label_list, threshold=threshold, times=100)

    auc_l, auc_u = np.percentile(auc_std, 2.5), np.percentile(auc_std, 97.5)
    prc_l, prc_u = np.percentile(prc_std, 2.5), np.percentile(prc_std, 97.5)
    acc_l, acc_u = np.percentile(acc_std, 2.5), np.percentile(acc_std, 97.5)
    bac_l, bac_u = np.percentile(bac_std, 2.5), np.percentile(bac_std, 97.5)
    f1s_l, f1s_u = np.percentile(f1s_std, 2.5), np.percentile(f1s_std, 97.5)
    sen_l, sen_u = np.percentile(sen_std, 2.5), np.percentile(sen_std, 97.5)
    spe_l, spe_u = np.percentile(spe_std, 2.5), np.percentile(spe_std, 97.5)
    ppv_l, ppv_u = np.percentile(ppv_std, 2.5), np.percentile(ppv_std, 97.5)
    npv_l, npv_u = np.percentile(npv_std, 2.5), np.percentile(npv_std, 97.5)

    results_df.append([f"{sub_bug}",
                       f"{format(np.mean(auc_std), '.3f')} ({format((auc_u - auc_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(prc_std), '.3f')} ({format((prc_u - prc_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(acc_std), '.3f')} ({format((acc_u - acc_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(bac_std), '.3f')} ({format((bac_u - bac_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(f1s_std), '.3f')} ({format((f1s_u - f1s_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(sen_std), '.3f')} ({format((sen_u - sen_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(spe_std), '.3f')} ({format((spe_u - spe_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(ppv_std), '.3f')} ({format((ppv_u - ppv_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(npv_std), '.3f')} ({format((npv_u - npv_l) / (2 * 1.96), '.3f')})",
                       ])
results_df = pd.DataFrame(results_df)
results_df.columns = ['Sample Number', 'AUROC', 'AUPRC', 'Accuracy', 'Balanced Accuracy',
                      'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
results_df.to_csv(f"Classifier_performance\\MLP3-uncertainty-remedis-{bug}.csv", index=False, encoding="cp1252")
