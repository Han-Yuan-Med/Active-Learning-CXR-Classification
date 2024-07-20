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
embed_file = pd.read_csv('Foundation_embeddings\\pakistan_dense121_embedding.csv')

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Model initialization using samples randomly selected
for sample_id in range(100):
    for idx in range(10, 51, 10):
        train_csv_tmp = pd.read_csv(f"Sample_random\\train_{sample_id}_{idx}.csv")
        print(f"Current sample number: {len(train_csv_tmp)}")
        train_set_cls_tmp = Lung_cls_embed(csv_file=train_csv_tmp, embed_file=embed_file)
        train_loader_cls_tmp = torch.utils.data.DataLoader(train_set_cls_tmp, batch_size=10, shuffle=True)
        train_loader_cls_loss = torch.utils.data.DataLoader(train_set_cls_tmp, batch_size=1, shuffle=False)

        cls_model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 2),
        )
        cls_model.to(device)

        weights = [len(np.where(train_csv_tmp.iloc[:, 1] == 1)[0]) / len(train_csv_tmp),
                   len(np.where(train_csv_tmp.iloc[:, 1] == 0)[0]) / len(train_csv_tmp)]
        # if weights[1] == 0:
        #     continue
        # if weights[0] == 0:
        #     continue

        print(f"Current weights are {weights}")
        class_weights = torch.FloatTensor(weights).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, cls_model.parameters()),
                                    lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        epoch_num = 200
        loss_past = 10086
        patience = 0
        early_stopping = 0

        for epoch in range(epoch_num):
            train_cls(train_loader=train_loader_cls_tmp, device=device, cls_model=cls_model,
                      criterion=criterion, optimizer=optimizer)
            print(f"Finish training at epoch {epoch + 1}")

            prob_list = []
            label_list = []
            cls_model.eval()
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
                PATH = f"Classifier_random\\dense121-random-{sample_id}-{idx}.pt"
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

        del train_loader_cls_tmp
        del cls_model
        del criterion
        del optimizer
        del scheduler

test_csv = pd.read_csv("covid_test_50.csv")
test_set_cls = Lung_cls_embed(csv_file=test_csv, embed_file=embed_file)
test_loader_cls = torch.utils.data.DataLoader(test_set_cls, batch_size=1, shuffle=False)

results_df = []
for idx in range(10, 51, 10):
    bootstrap_auc = []
    bootstrap_prc = []
    bootstrap_acc = []
    bootstrap_bac = []
    bootstrap_f1s = []
    bootstrap_sen = []
    bootstrap_spe = []
    bootstrap_ppv = []
    bootstrap_npv = []
    for sample_id in range(100):
        if not os.path.exists(f"Classifier_random\\dense121-random-{sample_id}-{idx}.pt"):
            auc_std = [0.5] * 100
            prc_std = [sum(test_csv.iloc[:, 1]) / len(test_csv)] * 100
            acc_std = [sum(test_csv.iloc[:, 1]) / len(test_csv)] * 100
            bac_std = [0.5] * 100
            f1s_std = [(2 * sum(test_csv.iloc[:, 1]) / len(test_csv)) / (
                    sum(test_csv.iloc[:, 1]) / len(test_csv) + 1)] * 100
            sen_std = [1] * 100
            spe_std = [0] * 100
            ppv_std = [sum(test_csv.iloc[:, 1]) / len(test_csv)] * 100
            npv_std = [0] * 100
        else:
            cls_model = torch.load(f"Classifier_random\\dense121-random-{sample_id}-{idx}.pt")
            cls_model.eval()

            prob_list = []
            label_list = []
            threshold = 0.5

            with torch.no_grad():
                for data in tqdm(test_loader_cls):
                    images, labels = data[0].float().to(device), data[1].cpu().numpy()
                    outputs = cls_model(images)
                    label_list = np.concatenate((label_list, labels), axis=None)
                    prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()),
                                               axis=None)
            auc_std, prc_std, acc_std, bac_std, f1s_std, sen_std, spe_std, ppv_std, npv_std = \
                bootstrap_cls_mul(prob_list=prob_list, label_list=label_list, threshold=threshold, times=100)

        bootstrap_auc = bootstrap_auc + auc_std
        bootstrap_prc = bootstrap_prc + prc_std
        bootstrap_acc = bootstrap_acc + acc_std
        bootstrap_bac = bootstrap_bac + bac_std
        bootstrap_f1s = bootstrap_f1s + f1s_std
        bootstrap_sen = bootstrap_sen + sen_std
        bootstrap_spe = bootstrap_spe + spe_std
        bootstrap_ppv = bootstrap_ppv + ppv_std
        bootstrap_npv = bootstrap_npv + npv_std

    auc_l, auc_u = np.percentile(bootstrap_auc, 2.5), np.percentile(bootstrap_auc, 97.5)
    prc_l, prc_u = np.percentile(bootstrap_prc, 2.5), np.percentile(bootstrap_prc, 97.5)
    acc_l, acc_u = np.percentile(bootstrap_acc, 2.5), np.percentile(bootstrap_acc, 97.5)
    bac_l, bac_u = np.percentile(bootstrap_bac, 2.5), np.percentile(bootstrap_bac, 97.5)
    f1s_l, f1s_u = np.percentile(bootstrap_f1s, 2.5), np.percentile(bootstrap_f1s, 97.5)
    sen_l, sen_u = np.percentile(bootstrap_sen, 2.5), np.percentile(bootstrap_sen, 97.5)
    spe_l, spe_u = np.percentile(bootstrap_spe, 2.5), np.percentile(bootstrap_spe, 97.5)
    ppv_l, ppv_u = np.percentile(bootstrap_ppv, 2.5), np.percentile(bootstrap_ppv, 97.5)
    npv_l, npv_u = np.percentile(bootstrap_npv, 2.5), np.percentile(bootstrap_npv, 97.5)

    results_df.append([f"{idx}",
                       f"{format(np.mean(bootstrap_auc), '.3f')} ({format((auc_u - auc_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(bootstrap_prc), '.3f')} ({format((prc_u - prc_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(bootstrap_acc), '.3f')} ({format((acc_u - acc_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(bootstrap_bac), '.3f')} ({format((bac_u - bac_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(bootstrap_f1s), '.3f')} ({format((f1s_u - f1s_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(bootstrap_sen), '.3f')} ({format((sen_u - sen_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(bootstrap_spe), '.3f')} ({format((spe_u - spe_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(bootstrap_ppv), '.3f')} ({format((ppv_u - ppv_l) / (2 * 1.96), '.3f')})",
                       f"{format(np.mean(bootstrap_npv), '.3f')} ({format((npv_u - npv_l) / (2 * 1.96), '.3f')})",
                       ])

results_df = pd.DataFrame(results_df)
results_df.columns = ['Budget', 'AUROC', 'AUPRC', 'Accuracy', 'Balanced Accuracy',
                      'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
results_df.to_csv("Classifier_performance\\dense121-random.csv", index=False, encoding="cp1252")
