import numpy as np
import pandas as pd

# Data split with positive and negative segmentation
csv_train = pd.read_csv("covid_train.csv")
csv_test = pd.read_csv("covid_test.csv")
csv_pos = pd.concat([csv_train.iloc[np.where(csv_train.iloc[:, 1] == 1)[0]],
                     csv_test.iloc[np.where(csv_test.iloc[:, 1] == 1)[0]]]).reset_index(drop=True)
csv_neg = pd.concat([csv_train.iloc[np.where(csv_train.iloc[:, 1] == 0)[0]],
                     csv_test.iloc[np.where(csv_test.iloc[:, 1] == 0)[0]]]).reset_index(drop=True)

np.random.seed(12345)
test_random_pos = np.random.choice(range(len(csv_pos)), np.int32(len(csv_pos) * 0.5), replace=False)
train_random_pos = np.setdiff1d(range(len(csv_pos)), test_random_pos)
test_random_neg = np.random.choice(range(len(csv_neg)), np.int32(len(csv_neg) * 0.5), replace=False)
train_random_neg = np.setdiff1d(range(len(csv_neg)), test_random_neg)

test_dataset = pd.concat([csv_pos.iloc[test_random_pos], csv_neg.iloc[test_random_neg]]).reset_index().drop("index",
                                                                                                            axis=1)
test_dataset.to_csv("covid_test_50.csv", index=False)
print(f'Positive ratio in test set is {sum(test_dataset.iloc[:, -1]) / len(test_dataset)}')

train_dataset = pd.concat([csv_pos.iloc[train_random_pos], csv_neg.iloc[train_random_neg]]).reset_index().drop("index",
                                                                                                               axis=1)
train_dataset.to_csv("covid_train_50.csv", index=False)
print(f'Positive ratio in train set contains {sum(train_dataset.iloc[:, -1]) / len(train_dataset)}')
